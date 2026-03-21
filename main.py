from fastapi import FastAPI
import threading

from database import engine, SessionLocal
from models import Base, Bill
from scraper.scheduler import run_scheduler, process_new_pdfs
from scraper.downloader import download_pdf

app = FastAPI()

# ✅ create tables
Base.metadata.create_all(bind=engine)


@app.on_event("startup")
def start_scheduler():
    print("🚀 Starting app...")

    # ✅ initial run
    process_new_pdfs(initial=True)

    # ✅ start scheduler
    thread = threading.Thread(target=run_scheduler, daemon=True)
    thread.start()


@app.get("/")
def home():
    return {"message": "Citizen Dashboard Backend Running"}


@app.get("/fetch-bill/{query}")
def fetch_bill(query: str):
    db = SessionLocal()

    bills = db.query(Bill).all()

    for bill in bills:
        # ✅ better matching
        if query.lower() in (bill.title or "").lower():

            print("🔍 Match found:", bill.title)

            # ✅ lazy download fix
            if not bill.local_path:
                print("⬇️ Lazy downloading...")
                path = download_pdf(bill.pdf_url)

                bill.local_path = path  # ✅ update DB object
                db.commit()

            response = {
                "message": "Bill ready",
                "pdf_url": bill.pdf_url,
                "local_path": bill.local_path
            }

            db.close()
            return response

    db.close()
    return {"message": "Bill not found"}