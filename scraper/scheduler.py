from scraper.fetch_links import get_pdf_links
from models import Bill
from database import SessionLocal
from AI.embedder import get_embedding

import time


def process_new_pdfs(initial: bool = False) -> None:
    """Ingest all known PDF links into the Bills table.

    - On initial startup, this seeds the DB with every current PDF link.
    - On scheduler runs, this only inserts links for newly appeared PDFs.
    - Heavy work (download + processing) is handled separately.
    """

    print("\n🔥 MODE:", "INITIAL" if initial else "SCHEDULER")

    db = SessionLocal()
    # Always scan the full bill listing so we don't miss new PDFs.
    links = get_pdf_links(num_bill_links=50, initial=False)

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
        process_new_pdfs(initial=False)