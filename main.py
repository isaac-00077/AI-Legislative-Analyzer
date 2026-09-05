from fastapi import FastAPI
from fastapi import Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import threading
import re
import traceback
from typing import List

from database import engine, SessionLocal
from models import Base, Bill, Chunk
from scraper.scheduler import run_scheduler, process_new_pdfs
from scraper.downloader import download_pdf
from scraper.fetch_links import get_pdf_links

from AI.extractor import process_pdf_to_chunks
from AI.summarizer import generate_summary
from AI.qa import answer_question

app = FastAPI()


STOP_WORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "bill", "by", "can",
    "could", "do", "for", "from", "give", "has", "have", "i", "in", "info",
    "information", "is", "it", "me", "more", "my", "of", "on", "or", "tell",
    "that", "the", "this", "to", "was", "what", "which", "with", "would", "you"
}


def _normalize_text(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()
    return re.sub(r"\s+", " ", cleaned)


def _token_set(value: str) -> set[str]:
    tokens = set()
    for t in _normalize_text(value).split(" "):
        if len(t) >= 2 and t not in STOP_WORDS:
            tokens.add(t)
    return tokens


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
            process_new_pdfs(max_bill_pages=10, source="startup")

            # 🟢 Step 2: download + process PDFs from FIRST 3 bill pages
            priority_items = get_pdf_links(max_bill_pages=3)
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
                                continue
                            bill.local_path = path

                        # process if needed
                        if not bill.processed:
                            print("⚙️ Initial processing:", bill.title)

                            original_chunks, compressed_chunks = process_pdf_to_chunks(bill.local_path)

                            if bill.summary is None and compressed_chunks:
                                summary = generate_summary(compressed_chunks)
                                if summary:
                                    bill.summary = summary

                            if original_chunks:
                                for orig, comp in zip(original_chunks, compressed_chunks):
                                    db.add(
                                        Chunk(
                                            bill_id=bill.id,
                                            original_text=orig,
                                            compressed_text=comp,
                                            embedding=None,
                                        )
                                    )

                                bill.processed = True

                    db.commit()

                finally:
                    db.close()

        except Exception as exc:
            print("❌ Bootstrap pipeline error:", exc)
        finally:
            run_scheduler()

    thread = threading.Thread(target=bootstrap_pipeline, daemon=True)
    thread.start()


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.api_route("/", methods=["GET", "HEAD"])
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
        # Step 1: try to find a matching bill already in the DB (lexical)
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
                        db.add(
                            Chunk(
                                bill_id=bill.id,
                                original_text=orig,
                                compressed_text=comp,
                                embedding=None,
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
                        db.add(
                            Chunk(
                                bill_id=bill.id,
                                original_text=orig,
                                compressed_text=comp,
                                embedding=None,
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
    """Answer a user question using text chunks from Supabase + Groq API.

    No SentenceTransformer / PyTorch needed — uses plain text retrieval.
    """
    print(f"📩 /ask called — query={query!r}, pdf_url={pdf_url!r}")
    db = SessionLocal()

    try:
        target_bill: Bill | None = None

        # 1. Lexical match query against all bill titles in DB first
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

        if lexical_best_bill is not None and lexical_best_score >= 0.25:
            print(f"🔎 ask() matched query to bill in DB (score={lexical_best_score:.2f}): {lexical_best_bill.title}")
            target_bill = lexical_best_bill
        elif pdf_url:
            # 2. Fall back to active card pdf_url if query is generic (e.g. "what is this bill about?")
            target_bill = db.query(Bill).filter(Bill.pdf_url == pdf_url).first()
            if target_bill:
                print(f"✅ ask() matched active card by pdf_url: {target_bill.title}")

        # 3. Answer from existing text chunks (no embeddings needed)
        if target_bill is not None:
            chunks = db.query(Chunk).filter(Chunk.bill_id == target_bill.id).all()
            print(f"📦 Found {len(chunks)} chunks for bill id={target_bill.id} ({target_bill.title})")

            # Lazy processing if bill has no chunks yet
            if not chunks:
                if not target_bill.local_path:
                    print("⬇️ ask() downloading matched bill...")
                    path = download_pdf(target_bill.pdf_url)
                    if path:
                        target_bill.local_path = path
                        db.commit()

                if target_bill.local_path:
                    print("⚙️ ask() processing matched bill...")
                    original_chunks, compressed_chunks = process_pdf_to_chunks(target_bill.local_path)
                    if target_bill.summary is None and compressed_chunks:
                        summary = generate_summary(compressed_chunks)
                        if summary:
                            target_bill.summary = summary

                    if original_chunks:
                        for orig, comp in zip(original_chunks, compressed_chunks):
                            db.add(
                                Chunk(
                                    bill_id=target_bill.id,
                                    original_text=orig,
                                    compressed_text=comp,
                                    embedding=None,
                                )
                            )
                        target_bill.processed = True
                        db.commit()
                    db.expire_all()
                    chunks = db.query(Chunk).filter(Chunk.bill_id == target_bill.id).all()
                    print(f"📦 After processing: {len(chunks)} chunks")

            if chunks:
                # Rank chunks by relevance to query keywords if available
                q_keywords = _token_set(query)
                if q_keywords:
                    def chunk_score(c: Chunk) -> int:
                        text_norm = _normalize_text(c.original_text or "")
                        return sum(1 for kw in q_keywords if kw in text_norm)
                    sorted_chunks = sorted(chunks, key=chunk_score, reverse=True)
                else:
                    sorted_chunks = chunks

                context_chunks = [
                    c.original_text for c in sorted_chunks[:8] if c.original_text
                ]
                print(f"📝 Sending {len(context_chunks)} context chunks to Groq for '{target_bill.title}'")

                if context_chunks:
                    answer = answer_question(query, context_chunks)
                    print("✅ Got answer from Groq, returning to frontend")
                    return {"answer": answer, "pdf_url": target_bill.pdf_url}

        # 4. No bill matched in DB — try source site on-demand
        print("🔎 No bill matched in DB, scanning source site...")
        fetch_result = fetch_bill(query)
        pdf_url_res = fetch_result.get("pdf_url") if isinstance(fetch_result, dict) else None

        if not pdf_url_res:
            return {
                "answer": "I could not find a closely related bill yet. Please try asking about a specific bill title.",
                "pdf_url": None,
            }

        bill = db.query(Bill).filter(Bill.pdf_url == pdf_url_res).first()
        if bill:
            bill_chunks = db.query(Chunk).filter(Chunk.bill_id == bill.id).limit(6).all()
            context_chunks = [c.original_text for c in bill_chunks if c.original_text]
            if context_chunks:
                answer = answer_question(query, context_chunks)
                return {"answer": answer, "pdf_url": pdf_url_res}

        return {
            "answer": "I found a related bill and started processing it. Please try your question again in a moment.",
            "pdf_url": pdf_url_res,
        }

    except Exception as exc:
        print("❌ Error in /ask endpoint:", exc)
        traceback.print_exc()
        return {"answer": "I encountered an error while processing your request. Please try again.", "pdf_url": None}

    finally:
        db.close()