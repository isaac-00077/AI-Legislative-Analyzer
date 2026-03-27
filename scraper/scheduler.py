from scraper.fetch_links import get_pdf_links
from models import Bill
from database import SessionLocal
from AI.embedder import get_embedding

import time


def process_new_pdfs(max_bill_pages: int = 50, source: str = "scheduler") -> None:
    """Ingest newly discovered PDF links into the Bills table.

    - `max_bill_pages` controls how many recent bill pages to inspect.
    - Existing PDFs are skipped by unique `pdf_url` lookup.
    - Heavy work (download + processing) is handled separately.
    """

    print(f"\n🔥 MODE: {source.upper()} | max_bill_pages={max_bill_pages}")


    db = SessionLocal()
    links = get_pdf_links(max_bill_pages=max_bill_pages)
 
    try:
        for item in links:
            link = item["pdf_url"]
            title = item["title"]

            bill = db.query(Bill).filter_by(pdf_url=link).first()

            if not bill:
                print("🆕 New Bill Found:", link)

                title_embedding = get_embedding(title) if title else None

                bill = Bill(
                    title=title,
                    pdf_url=link,
                    local_path=None,   # not downloaded
                    processed=False,
                    title_embedding=title_embedding,
                )

                db.add(bill)

        db.commit()

    finally:
        db.close()


def run_scheduler(interval_seconds: int = 43200) -> None:
    """Background loop that periodically ingests links for new PDFs only.

    Default interval is 12 hours (43200 seconds).
    """

    while True:
        # Sleep *before* each scheduler run so that, after the initial
        # startup seeding, we wait the full interval before re-scanning.
        print(f"\n⏱️ Scheduler sleeping for {interval_seconds} seconds (~{interval_seconds / 3600:.1f} hours) before next check...")
        time.sleep(interval_seconds)

        print("\n⏱️ Checking for new PDFs...")
        # Keep scheduler runs bounded to recent bill pages for reliability.
        process_new_pdfs(max_bill_pages=50, source="scheduler")