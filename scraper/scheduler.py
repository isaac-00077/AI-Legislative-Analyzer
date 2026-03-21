import time
from scraper.fetch_links import get_pdf_links
from scraper.downloader import download_pdf
from models import Bill
from database import SessionLocal

def process_new_pdfs(initial=False):
    print("\n🔥 MODE:", "INITIAL" if initial else "SCHEDULER")

    db = SessionLocal()

    links = get_pdf_links(initial=initial)

    for item in links:
        link = item["pdf_url"]
        title = item["title"]

        existing = db.query(Bill).filter_by(pdf_url=link).first()

        if not existing:
            print("🆕 New PDF:", link)

            local_path = None

            # ✅ only download in initial
            if initial:
                local_path = download_pdf(link)

            new_bill = Bill(
                title=title,
                pdf_url=link,
                local_path=local_path,
                processed=False
            )

            db.add(new_bill)

    db.commit()
    db.close()


def run_scheduler():
    while True:
        print("\n⏱️ Checking for new PDFs...")
        process_new_pdfs(initial=False)
        time.sleep(600)  # change to 10 for testing