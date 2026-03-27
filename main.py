from fastapi import FastAPI
from fastapi import Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import threading
import math
import re
from typing import List

from database import engine, SessionLocal
from models import Base, Bill, Chunk
from scraper.scheduler import run_scheduler, process_new_pdfs
from scraper.downloader import download_pdf
from scraper.fetch_links import get_pdf_links

from AI.extractor import process_pdf_to_chunks
from AI.embedder import get_embedding
from AI.summarizer import generate_summary
from AI.qa import answer_question

app = FastAPI()


def _normalize_text(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()
    return re.sub(r"\s+", " ", cleaned)


def _token_set(value: str) -> set[str]:
    return {t for t in _normalize_text(value).split(" ") if len(t) > 2}


def _lexical_score(query: str, candidate: str) -> float:
    qn = _normalize_text(query)
    cn = _normalize_text(candidate)
    if not qn or not cn:
        return 0.0

    if qn in cn:
        return 1.0

    q_tokens = _token_set(query)
    c_tokens = _token_set(candidate)
    if not q_tokens or not c_tokens:
        return 0.0

    overlap = len(q_tokens & c_tokens)
    if overlap == 0:
        return 0.0

    # Bias toward recall of query tokens for user-entered phrases.
    return overlap / len(q_tokens)


def _bill_has_chunks(db, bill_id: int) -> bool:
    return db.query(Chunk.id).filter(Chunk.bill_id == bill_id).first() is not None

# Allow the frontend (e.g. Vercel) to call this API.
# You can tighten allow_origins later to just your Vercel URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ create tables
Base.metadata.create_all(bind=engine)


@app.on_event("startup")
def start_app() -> None:
    print("🚀 Starting app...")

    def bootstrap_pipeline() -> None:
        """Run heavy initialization work without blocking API startup."""

        try:
            # 🟢 Step 1: ingest only recent bill pages to keep startup light
            process_new_pdfs(max_bill_pages=15, source="startup")

            # 🟢 Step 2: download + process PDFs from FIRST 5 bill pages
            priority_items = get_pdf_links(max_bill_pages=5)
            priority_urls = {item["pdf_url"] for item in priority_items}

            if priority_urls:
                db = SessionLocal()
                try:
                    bills = db.query(Bill).filter(Bill.pdf_url.in_(priority_urls)).all()

                    for bill in bills:
                        # download if needed
                        if not bill.local_path:
                            print("⬇️ Initial download:", bill.title)
                            path = download_pdf(bill.pdf_url)
                            if not path:
                                # Skip processing if download failed
                                continue
                            bill.local_path = path

                        # process if needed
                        if not bill.processed:
                            print("⚙️ Initial processing:", bill.title)

                            original_chunks, compressed_chunks = process_pdf_to_chunks(bill.local_path)

                            # Generate summary once for dashboard
                            if bill.summary is None and compressed_chunks:
                                summary = generate_summary(compressed_chunks)
                                if summary:
                                    bill.summary = summary

                            if original_chunks:
                                for orig, comp in zip(original_chunks, compressed_chunks):
                                    embedding = get_embedding(comp)
                                    # Store chunks even if embedding is None; text-based retrieval
                                    # will still work for answering.
                                    db.add(
                                        Chunk(
                                            bill_id=bill.id,
                                            original_text=orig,
                                            compressed_text=comp,
                                            embedding=embedding,
                                        )
                                    )

                                bill.processed = True

                    db.commit()

                finally:
                    db.close()

        except Exception as exc:
            # Keep server alive even if bootstrap fails.
            print("❌ Bootstrap pipeline error:", exc)
        finally:
            # 🔁 Start background scheduler (link-only ingestion for new PDFs)
            run_scheduler()

    thread = threading.Thread(target=bootstrap_pipeline, daemon=True)
    thread.start()


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def home() -> FileResponse:
    return FileResponse("static/index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    # Quiet browser favicon probing when no icon file is shipped.
    return Response(status_code=204)


@app.get("/fetch-bill/{query}")
def fetch_bill(query: str):
    db = SessionLocal()

    try:
        # Step 0: semantic search over already-processed bills using embeddings
        query_embedding = get_embedding(query)
        if query_embedding:
            best_score: float | None = None
            best_bill: Bill | None = None

            # Only look at chunks for bills that are already processed
            chunks = db.query(Chunk).join(Bill).filter(Bill.processed.is_(True)).all()

            def cosine_similarity(a: list[float], b: list[float]) -> float:
                dot = sum(x * y for x, y in zip(a, b))
                norm_a = math.sqrt(sum(x * x for x in a))
                norm_b = math.sqrt(sum(y * y for y in b))
                if norm_a == 0 or norm_b == 0:
                    return 0.0
                return dot / (norm_a * norm_b)

            for chunk in chunks:
                emb = chunk.embedding
                # emb can be a numpy/pgvector array; avoid boolean context
                if emb is None:
                    continue

                score = cosine_similarity(query_embedding, list(emb))
                if best_score is None or score > best_score:
                    best_score = score
                    best_bill = chunk.bill

            # If we found a reasonably similar bill, return it immediately
            if best_bill is not None and best_score is not None and best_score >= 0.4:
                return {
                    "message": "Bill ready (semantic match)",
                    "pdf_url": best_bill.pdf_url,
                    "local_path": best_bill.local_path,
                    "processed": best_bill.processed,
                    "similarity": best_score,
                }

        # Step 0.5: semantic search over bill titles using title embeddings
        if query_embedding:
            title_best_score: float | None = None
            title_best_bill: Bill | None = None

            bills_for_titles = db.query(Bill).all()

            def cosine_similarity(a: list[float], b: list[float]) -> float:  # reuse
                dot = sum(x * y for x, y in zip(a, b))
                norm_a = math.sqrt(sum(x * x for x in a))
                norm_b = math.sqrt(sum(y * y for y in b))
                if norm_a == 0 or norm_b == 0:
                    return 0.0
                return dot / (norm_a * norm_b)

            for bill in bills_for_titles:
                emb = bill.title_embedding
                # emb can be a numpy/pgvector array; avoid boolean context
                if emb is None:
                    continue

                score = cosine_similarity(query_embedding, list(emb))
                if title_best_score is None or score > title_best_score:
                    title_best_score = score
                    title_best_bill = bill

            if (
                title_best_bill is not None
                and title_best_score is not None
                and title_best_score > 0.5
            ):
                bill = title_best_bill
                print("🔍 Match found (title semantic):", bill.title)

                # Same lazy download + processing flow as keyword match
                if not bill.local_path:
                    print("⬇️ Lazy downloading (title semantic)...")
                    path = download_pdf(bill.pdf_url)
                    if not path:
                        return {"message": "Failed to download existing bill PDF"}
                    bill.local_path = path
                    db.commit()

                if not bill.processed:
                    print("⚙️ Lazy processing (title semantic)...")

                    original_chunks, compressed_chunks = process_pdf_to_chunks(bill.local_path)

                    if bill.summary is None and compressed_chunks:
                        summary = generate_summary(compressed_chunks)
                        if summary:
                            bill.summary = summary

                    if original_chunks:
                        for orig, comp in zip(original_chunks, compressed_chunks):
                            embedding = get_embedding(comp)
                            if embedding is None:
                                continue

                            db.add(
                                Chunk(
                                    bill_id=bill.id,
                                    original_text=orig,
                                    compressed_text=comp,
                                    embedding=embedding,
                                )
                            )

                        bill.processed = True
                        db.commit()

                return {
                    "message": "Bill ready (title semantic match)",
                    "pdf_url": bill.pdf_url,
                    "local_path": bill.local_path,
                    "processed": bill.processed,
                    "similarity": title_best_score,
                }

        # Step 1: try to find a matching bill already in the DB
        bills = db.query(Bill).all()
        lexical_best_bill: Bill | None = None
        lexical_best_score = 0.0

        for bill in bills:
            title_score = _lexical_score(query, bill.title or "")
            url_score = _lexical_score(query, bill.pdf_url or "")
            score = max(title_score, url_score)
            if score > lexical_best_score:
                lexical_best_score = score
                lexical_best_bill = bill

        if lexical_best_bill is not None and lexical_best_score >= 0.4:
            bill = lexical_best_bill
            print(f"🔍 Match found (DB lexical={lexical_best_score:.2f}):", bill.title)

            # 🔴 Lazy download
            if not bill.local_path:
                print("⬇️ Lazy downloading...")
                path = download_pdf(bill.pdf_url)
                if not path:
                    return {"message": "Failed to download existing bill PDF"}
                bill.local_path = path
                db.commit()

            # 🔴 Lazy processing
            has_chunks = _bill_has_chunks(db, bill.id)
            if not bill.processed or not has_chunks:
                print("⚙️ Lazy processing...")

                original_chunks, compressed_chunks = process_pdf_to_chunks(bill.local_path)

                if bill.summary is None and compressed_chunks:
                    summary = generate_summary(compressed_chunks)
                    if summary:
                        bill.summary = summary

                if original_chunks:
                    for orig, comp in zip(original_chunks, compressed_chunks):
                        embedding = get_embedding(comp)
                        # Store chunks even if embedding is None; text-based retrieval
                        # will still work for answering.
                        db.add(
                            Chunk(
                                bill_id=bill.id,
                                original_text=orig,
                                compressed_text=comp,
                                embedding=embedding,
                            )
                        )

                    bill.processed = True
                    db.commit()

            return {
                "message": "Bill ready",
                "pdf_url": bill.pdf_url,
                "local_path": bill.local_path,
                "processed": bill.processed,
            }

        # Step 2: not in DB → search source site for a matching PDF
        print("🔎 Bill not in DB, scanning source site...")
        # Keep live query scraping bounded to recent listings.
        links = get_pdf_links(max_bill_pages=50)

        source_best = None
        source_best_score = 0.0
        for item in links:
            title = item["title"] or ""
            pdf_url = item["pdf_url"]

            score = max(_lexical_score(query, title), _lexical_score(query, pdf_url))
            if score > source_best_score:
                source_best_score = score
                source_best = item

        if source_best is not None and source_best_score >= 0.4:
            title = source_best["title"] or ""
            pdf_url = source_best["pdf_url"]
            print("✅ Match found on source:", title)

            bill = db.query(Bill).filter_by(pdf_url=pdf_url).first()
            if not bill:
                bill = Bill(
                    title=title,
                    pdf_url=pdf_url,
                    local_path=None,
                    processed=False,
                )
                db.add(bill)
                db.commit()
                db.refresh(bill)

            # On-demand download
            if not bill.local_path:
                print("⬇️ On-demand downloading...")
                path = download_pdf(bill.pdf_url)
                if not path:
                    return {"message": "Failed to download PDF from source"}
                bill.local_path = path
                db.commit()

            # On-demand processing
            if not bill.processed:
                print("⚙️ On-demand processing...")

                original_chunks, compressed_chunks = process_pdf_to_chunks(bill.local_path)

                if bill.summary is None and compressed_chunks:
                    summary = generate_summary(compressed_chunks)
                    if summary:
                        bill.summary = summary

                if original_chunks:
                    for orig, comp in zip(original_chunks, compressed_chunks):
                        embedding = get_embedding(comp)
                        # Store chunks even if embedding is None; text-based retrieval
                        # will still work for answering.
                        db.add(
                            Chunk(
                                bill_id=bill.id,
                                original_text=orig,
                                compressed_text=comp,
                                embedding=embedding,
                            )
                        )

                    bill.processed = True
                    db.commit()

            return {
                "message": "Bill ready (fetched from source)",
                "pdf_url": bill.pdf_url,
                "local_path": bill.local_path,
                "processed": bill.processed,
            }

        return {"message": "Bill not found"}

    finally:
        db.close()


@app.get("/dashboard")
def dashboard() -> List[dict[str, str]]:
    db = SessionLocal()
    try:
        bills = (
            db.query(Bill)
            .filter(Bill.processed.is_(True), Bill.summary.isnot(None))
            .order_by(Bill.id.desc())
            .all()
        )
        return [
            {
                "title": bill.title,
                "summary": bill.summary,
                "pdf_url": bill.pdf_url,
            }
            for bill in bills
        ]
    finally:
        db.close()


@app.get("/ask")
def ask(query: str, pdf_url: str | None = None):
    db = SessionLocal()

    try:
        target_bill: Bill | None = None
        if pdf_url:
            target_bill = db.query(Bill).filter(Bill.pdf_url == pdf_url).first()

            if target_bill is None:
                return {
                    "answer": "I could not find that bill in the database yet.",
                    "pdf_url": pdf_url,
                }

            if not target_bill.processed:
                seed_query = target_bill.title or target_bill.pdf_url
                fetch_bill(seed_query)
                db.expire_all()
                target_bill = (
                    db.query(Bill)
                    .filter(Bill.pdf_url == pdf_url, Bill.processed.is_(True))
                    .first()
                )

            if target_bill is None:
                return {
                    "answer": "I found the bill and started processing it. Please try again shortly.",
                    "pdf_url": pdf_url,
                }

        # For explicit bill-name queries, prefer lexical title match first.
        if target_bill is None:
            lexical_best_bill: Bill | None = None
            lexical_best_score = 0.0
            for bill in db.query(Bill).all():
                score = max(
                    _lexical_score(query, bill.title or ""),
                    _lexical_score(query, bill.pdf_url or ""),
                )
                if score > lexical_best_score:
                    lexical_best_score = score
                    lexical_best_bill = bill

            if lexical_best_bill is not None and lexical_best_score >= 0.4:
                print(f"🔎 ask() lexical bill match={lexical_best_score:.2f}:", lexical_best_bill.title)
                target_bill = lexical_best_bill

                if not target_bill.local_path:
                    print("⬇️ ask() downloading matched bill...")
                    path = download_pdf(target_bill.pdf_url)
                    if path:
                        target_bill.local_path = path
                        db.commit()

                has_chunks = _bill_has_chunks(db, target_bill.id)
                if target_bill.local_path and (not target_bill.processed or not has_chunks):
                    print("⚙️ ask() processing matched bill...")
                    original_chunks, compressed_chunks = process_pdf_to_chunks(target_bill.local_path)

                    if target_bill.summary is None and compressed_chunks:
                        summary = generate_summary(compressed_chunks)
                        if summary:
                            target_bill.summary = summary

                    if original_chunks:
                        for orig, comp in zip(original_chunks, compressed_chunks):
                            embedding = get_embedding(comp)
                            db.add(
                                Chunk(
                                    bill_id=target_bill.id,
                                    original_text=orig,
                                    compressed_text=comp,
                                    embedding=embedding,
                                )
                            )

                        target_bill.processed = True
                        db.commit()

                db.expire_all()
                target_bill = db.query(Bill).filter(Bill.id == target_bill.id).first()

        query_embedding = get_embedding(query)
        # If embeddings are temporarily unavailable on the host,
        # fall back to bill-level retrieval instead of hard-failing.
        if query_embedding is None:
            print("🟡 Query embedding unavailable; falling back to bill-level retrieval")

            if target_bill is not None:
                fallback_chunks = (
                    db.query(Chunk)
                    .filter(Chunk.bill_id == target_bill.id)
                    .order_by(Chunk.id.asc())
                    .limit(3)
                    .all()
                )
                context_chunks = [c.original_text for c in fallback_chunks if c.original_text]
                if context_chunks:
                    answer = answer_question(query, context_chunks)
                    return {"answer": answer, "pdf_url": target_bill.pdf_url}

                # Bill exists but has no chunks yet: trigger one direct processing attempt.
                has_chunks = _bill_has_chunks(db, target_bill.id)
                if target_bill.local_path and not has_chunks:
                    print("⚙️ ask() no chunks found; processing matched bill directly...")
                    original_chunks, compressed_chunks = process_pdf_to_chunks(target_bill.local_path)
                    if original_chunks:
                        for orig, comp in zip(original_chunks, compressed_chunks):
                            embedding = get_embedding(comp)
                            db.add(
                                Chunk(
                                    bill_id=target_bill.id,
                                    original_text=orig,
                                    compressed_text=comp,
                                    embedding=embedding,
                                )
                            )
                        target_bill.processed = True
                        db.commit()

                    fallback_chunks = (
                        db.query(Chunk)
                        .filter(Chunk.bill_id == target_bill.id)
                        .order_by(Chunk.id.asc())
                        .limit(3)
                        .all()
                    )
                    context_chunks = [c.original_text for c in fallback_chunks if c.original_text]
                    if context_chunks:
                        answer = answer_question(query, context_chunks)
                        return {"answer": answer, "pdf_url": target_bill.pdf_url}

            fetch_result = fetch_bill(query)
            pdf_url = fetch_result.get("pdf_url") if isinstance(fetch_result, dict) else None

            if not pdf_url:
                return {
                    "answer": "I could not find a closely related bill yet. Please try a more specific bill name.",
                    "pdf_url": None,
                }

            bill = db.query(Bill).filter(Bill.pdf_url == pdf_url, Bill.processed.is_(True)).first()
            if not bill:
                return {
                    "answer": "I found a related bill and started processing it. Please try again shortly.",
                    "pdf_url": pdf_url,
                }

            # Use first chunks as deterministic fallback context.
            fallback_chunks = (
                db.query(Chunk)
                .filter(Chunk.bill_id == bill.id)
                .order_by(Chunk.id.asc())
                .limit(3)
                .all()
            )
            context_chunks = [c.original_text for c in fallback_chunks if c.original_text]

            if not context_chunks:
                return {
                    "answer": "I found the bill but could not extract enough text to answer yet.",
                    "pdf_url": pdf_url,
                }

            answer = answer_question(query, context_chunks)
            return {"answer": answer, "pdf_url": pdf_url}

        # Use chunk embeddings for similarity, but only from processed bills
        if target_bill is not None:
            chunks = db.query(Chunk).filter(Chunk.bill_id == target_bill.id).all()
        else:
            chunks = db.query(Chunk).join(Bill).filter(Bill.processed.is_(True)).all()

        def cosine_similarity(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(y * y for y in b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

        scored: list[tuple[float, Chunk]] = []
        best_score: float | None = None
        for chunk in chunks:
            emb = chunk.embedding
            # "emb" can be a numpy/pgvector array; avoid boolean context
            if emb is None or len(emb) == 0:
                continue

            try:
                emb_list = list(emb)
            except Exception as exc:
                # If an embedding cannot be converted, skip that chunk
                print("❌ Error converting chunk embedding to list:", exc, type(emb))
                continue

            score = cosine_similarity(query_embedding, emb_list)
            scored.append((score, chunk))
            if best_score is None or score > best_score:
                best_score = score

        # If we have no processed chunks or the best match is weak,
        # trigger on-demand fetch & processing for a relevant bill.
        # Use a fairly strict threshold so that loosely related bills
        # don't block us from searching/downloading a better match.
        if not scored or best_score is None or best_score < 0.7:
            print("🟡 No strong match in processed data; fetching/processing bill on demand...")

            # Reuse the fetch_bill flow to locate and process the best bill.
            fetch_result = fetch_bill(query)
            pdf_url = None
            if isinstance(fetch_result, dict):
                pdf_url = fetch_result.get("pdf_url")

            if not pdf_url:
                return {
                    "answer": "I could not find any bill closely related to your question yet.",
                    "pdf_url": None,
                }

            bill = db.query(Bill).filter(Bill.pdf_url == pdf_url, Bill.processed.is_(True)).first()
            if not bill:
                return {
                    "answer": "I started processing the most relevant bill for your question. Please try again in a little while.",
                    "pdf_url": pdf_url,
                }

            # Restrict chunks to the chosen bill and rescore.
            chunks = db.query(Chunk).filter(Chunk.bill_id == bill.id).all()
            scored = []
            for chunk in chunks:
                emb = chunk.embedding
                if emb is None or len(emb) == 0:
                    continue
                try:
                    emb_list = list(emb)
                except Exception as exc:
                    print("❌ Error converting chunk embedding to list (bill-restricted):", exc, type(emb))
                    continue
                score = cosine_similarity(query_embedding, emb_list)
                scored.append((score, chunk))

            if not scored:
                # No scored chunks (all embeddings were None), fall back to first chunks by order.
                print("🟡 No embeddings available; falling back to first chunks by order...")
                fallback = (
                    db.query(Chunk)
                    .filter(Chunk.bill_id == bill.id)
                    .order_by(Chunk.id.asc())
                    .limit(3)
                    .all()
                )
                context_chunks = [c.original_text for c in fallback if c.original_text]
                if context_chunks:
                    answer = answer_question(query, context_chunks)
                    return {"answer": answer, "pdf_url": pdf_url}

                return {
                    "answer": "I processed a matching bill but could not extract enough relevant content to answer this question.",
                    "pdf_url": pdf_url,
                }

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:3]

        context_chunks = [c.original_text for _, c in top if c.original_text]
        answer = answer_question(query, context_chunks)

        # Use the best-scoring chunk's bill to expose a PDF link
        best_chunk = top[0][1]
        pdf_url = best_chunk.bill.pdf_url if best_chunk.bill else None

        return {"answer": answer, "pdf_url": pdf_url}

    finally:
        db.close()