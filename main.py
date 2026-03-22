from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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


Base.metadata.create_all(bind=engine)


LAST_BILL_ID: int | None = None
LAST_RESPONSE_TEXT: str | None = None


_TITLE_MATCH_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "for",
    "to",
    "in",
    "on",
    "with",
    "from",
    "by",
    "about",
    "what",
    "which",
    "tell",
    "me",
    "is",
    "are",
    "was",
    "were",
    "this",
    "that",
    "bill",
    "act",
    "law",
}


def _normalize_for_match(text: str | None) -> str:
    value = (text or "").lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _keyword_tokens(text: str | None) -> set[str]:
    normalized = _normalize_for_match(text)
    return {
        token
        for token in normalized.split()
        if len(token) > 2 and token not in _TITLE_MATCH_STOPWORDS
    }


def _bill_title_match_score(query: str, title: str | None) -> float:
    query_norm = _normalize_for_match(query)
    title_norm = _normalize_for_match(title)

    if not query_norm or not title_norm:
        return 0.0

    query_tokens = _keyword_tokens(query)
    title_tokens = _keyword_tokens(title)

    token_overlap_score = 0.0
    if query_tokens and title_tokens:
        overlap = query_tokens.intersection(title_tokens)
        token_overlap_score = len(overlap) / len(query_tokens)

    phrase_bonus = 0.0
    if query_norm in title_norm:
        phrase_bonus = 0.7
    elif any(token in title_norm for token in query_tokens):
        phrase_bonus = 0.25

    return token_overlap_score + phrase_bonus


def _best_lexical_bill_match(query: str, bills: list[Bill]) -> tuple[Bill | None, float]:
    best_bill: Bill | None = None
    best_score = 0.0

    for bill in bills:
        score = _bill_title_match_score(query, bill.title)
        if score > best_score:
            best_score = score
            best_bill = bill

    return best_bill, best_score


def _is_followup_query(query: str) -> bool:
    """Detect vague follow-up prompts like 'explain clearly' or 'simplify'."""

    q = (query or "").lower()
    if not q:
        return False

    keywords = [
        "explain",
        "clarify",
        "simplify",
        "simpler",
        "elaborate",
        "in detail",
        "explain clearly",
        "explain it clearly",
        "make it clear",
        "make it simple",
    ]

    return any(phrase in q for phrase in keywords)


@app.on_event("startup")
def start_app() -> None:
    print("Starting app...")

    def bootstrap_pipeline() -> None:
        """Run heavy initialization work without blocking API startup."""

        try:
            
            process_new_pdfs(initial=True)

            
            priority_items = get_pdf_links(num_bill_links=5, initial=True)
            priority_urls = {item["pdf_url"] for item in priority_items}

            if priority_urls:
                db = SessionLocal()
                try:
                    bills = db.query(Bill).filter(Bill.pdf_url.in_(priority_urls)).all()

                    for bill in bills:
                        
                        if not bill.local_path:
                            print("Initial download:", bill.title)
                            path = download_pdf(bill.pdf_url)
                            if not path:
                                
                                continue
                            bill.local_path = path

                        # process if needed
                        if not bill.processed:
                            print("Initial processing:", bill.title)

                            original_chunks, compressed_chunks = process_pdf_to_chunks(bill.local_path)

                            # Generate summary once for dashboard
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

                finally:
                    db.close()

        except Exception as exc:
            
            print("Bootstrap pipeline error:", exc)
        finally:
            
            run_scheduler()

    thread = threading.Thread(target=bootstrap_pipeline, daemon=True)
    thread.start()


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def home() -> FileResponse:
    return FileResponse("static/index.html")


@app.get("/fetch-bill/{query}")
def fetch_bill(query: str):
    db = SessionLocal()

    try:
        
        bills = db.query(Bill).all()
        lexical_bill, lexical_score = _best_lexical_bill_match(query, bills)

        if lexical_bill is not None and lexical_score >= 0.4:
            bill = lexical_bill
            print(f"Match found (title lexical {lexical_score:.2f}):", bill.title)

            if not bill.local_path:
                print("Lazy downloading (title lexical)...")
                path = download_pdf(bill.pdf_url)
                if not path:
                    return {"message": "Failed to download existing bill PDF"}
                bill.local_path = path
                db.commit()

            if not bill.processed:
                print("Lazy processing (title lexical)...")

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
                "message": "Bill ready (title lexical match)",
                "pdf_url": bill.pdf_url,
                "local_path": bill.local_path,
                "processed": bill.processed,
                "lexical_score": lexical_score,
            }

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
                print("Match found (title semantic):", bill.title)

                # Same lazy download + processing flow as keyword match
                if not bill.local_path:
                    print("Lazy downloading (title semantic)...")
                    path = download_pdf(bill.pdf_url)
                    if not path:
                        return {"message": "Failed to download existing bill PDF"}
                    bill.local_path = path
                    db.commit()

                if not bill.processed:
                    print("Lazy processing (title semantic)...")

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

        

        for bill in bills:
            if query.lower() in (bill.title or "").lower():
                print("Match found (DB):", bill.title)

                if not bill.local_path:
                    print("Lazy downloading...")
                    path = download_pdf(bill.pdf_url)
                    if not path:
                        return {"message": "Failed to download existing bill PDF"}
                    bill.local_path = path
                    db.commit()

                
                if not bill.processed:
                    print("Lazy processing...")

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
                    "message": "Bill ready",
                    "pdf_url": bill.pdf_url,
                    "local_path": bill.local_path,
                    "processed": bill.processed,
                }

        # Step 2: not in DB → search source site for a matching PDF
        print("Bill not in DB, scanning source site...")
        links = get_pdf_links(initial=False)

        for item in links:
            title = item["title"] or ""
            pdf_url = item["pdf_url"]

            if query.lower() in title.lower() or query.lower() in pdf_url.lower():
                print("Match found on source:", title)

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
                    print("On-demand downloading...")
                    path = download_pdf(bill.pdf_url)
                    if not path:
                        return {"message": "Failed to download PDF from source"}
                    bill.local_path = path
                    db.commit()

                # On-demand processing
                if not bill.processed:
                    print("On-demand processing...")

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
        global LAST_BILL_ID, LAST_RESPONSE_TEXT

        # Step -2: lightweight follow-up handling. If the user asks a vague
        # refinement like "explain clearly" and we have a previous answer,
        # reuse that answer directly instead of doing any new retrieval or
        # PDF processing.
        if _is_followup_query(query) and LAST_RESPONSE_TEXT:
            refinement_query = (
                "Explain the following in a simpler and clearer way. "
                "Do not add new facts, only clarify and rephrase the existing explanation."
            )

            refined_answer = answer_question(refinement_query, [LAST_RESPONSE_TEXT])

            # Keep the last bill context (if any) and just update the text.
            LAST_RESPONSE_TEXT = refined_answer

            last_pdf_url: str | None = None
            if LAST_BILL_ID is not None:
                try:
                    bill = db.query(Bill).get(LAST_BILL_ID)
                except Exception:
                    bill = None
                if bill is not None:
                    last_pdf_url = bill.pdf_url

            return {"answer": refined_answer, "pdf_url": last_pdf_url}

        # Step -0: if a specific bill PDF is provided, lock retrieval to that bill.
        if pdf_url:
            bill = (
                db.query(Bill)
                .filter(Bill.pdf_url == pdf_url, Bill.processed.is_(True))
                .first()
            )

            if bill is not None:
                bill_chunks = db.query(Chunk).filter(Chunk.bill_id == bill.id).all()
                if bill_chunks:
                    query_embedding = get_embedding(query)
                    if query_embedding is None:
                        return {"answer": "Could not create embedding for your query.", "pdf_url": pdf_url}

                    def cosine_similarity(a: list[float], b: list[float]) -> float:
                        dot = sum(x * y for x, y in zip(a, b))
                        norm_a = math.sqrt(sum(x * x for x in a))
                        norm_b = math.sqrt(sum(y * y for y in b))
                        if norm_a == 0 or norm_b == 0:
                            return 0.0
                        return dot / (norm_a * norm_b)

                    scored_bill_chunks: list[tuple[float, Chunk]] = []
                    for chunk in bill_chunks:
                        emb = chunk.embedding
                        if emb is None or len(emb) == 0:
                            continue
                        try:
                            score = cosine_similarity(query_embedding, list(emb))
                        except Exception as exc:
                            print("❌ Error scoring bill-locked chunk:", exc)
                            continue
                        scored_bill_chunks.append((score, chunk))

                    if scored_bill_chunks:
                        scored_bill_chunks.sort(key=lambda x: x[0], reverse=True)
                        top_chunks = scored_bill_chunks[:3]
                        context_chunks = [c.original_text for _, c in top_chunks if c.original_text]
                        answer = answer_question(query, context_chunks)

                        LAST_BILL_ID = bill.id
                        LAST_RESPONSE_TEXT = answer

                        return {"answer": answer, "pdf_url": pdf_url}

        # Step -1: if query clearly names a bill title, lock retrieval to that bill.
        processed_bills = db.query(Bill).filter(Bill.processed.is_(True)).all()
        lexical_bill, lexical_score = _best_lexical_bill_match(query, processed_bills)

        if lexical_bill is not None and lexical_score >= 0.4:
            bill_chunks = db.query(Chunk).filter(Chunk.bill_id == lexical_bill.id).all()
            if bill_chunks:
                query_embedding = get_embedding(query)
                if query_embedding is None:
                    return {"answer": "Could not create embedding for your query."}

                def cosine_similarity(a: list[float], b: list[float]) -> float:
                    dot = sum(x * y for x, y in zip(a, b))
                    norm_a = math.sqrt(sum(x * x for x in a))
                    norm_b = math.sqrt(sum(y * y for y in b))
                    if norm_a == 0 or norm_b == 0:
                        return 0.0
                    return dot / (norm_a * norm_b)

                scored_bill_chunks: list[tuple[float, Chunk]] = []
                for chunk in bill_chunks:
                    emb = chunk.embedding
                    if emb is None or len(emb) == 0:
                        continue
                    try:
                        score = cosine_similarity(query_embedding, list(emb))
                    except Exception as exc:
                        print("❌ Error scoring lexical-matched bill chunk:", exc)
                        continue
                    scored_bill_chunks.append((score, chunk))

                if scored_bill_chunks:
                    scored_bill_chunks.sort(key=lambda x: x[0], reverse=True)
                    top_chunks = scored_bill_chunks[:3]
                    context_chunks = [c.original_text for _, c in top_chunks if c.original_text]
                    answer = answer_question(query, context_chunks)

                    LAST_BILL_ID = lexical_bill.id
                    LAST_RESPONSE_TEXT = answer

                    return {"answer": answer, "pdf_url": lexical_bill.pdf_url}

        query_embedding = get_embedding(query)
        # Explicitly check for None to avoid ambiguous truth-value on arrays
        if query_embedding is None:
            return {"answer": "Could not create embedding for your query."}

        # Use chunk embeddings for similarity, but only from processed bills
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

        if best_chunk.bill is not None:
            LAST_BILL_ID = best_chunk.bill.id
        else:
            LAST_BILL_ID = None
        LAST_RESPONSE_TEXT = answer

        return {"answer": answer, "pdf_url": pdf_url}

    finally:
        db.close()