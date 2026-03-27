import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from typing import Optional

BASE_URL = "https://prsindia.org"

HEADERS = {"User-Agent": "Mozilla/5.0"}


def get_pdf_links(max_bill_pages: Optional[int] = 50):
    """Scrape PRS bill pages and return discovered PDF links.

    Args:
        max_bill_pages: Maximum number of bill detail pages to scan from
            the billtrack listing order. Use None to scan all pages.
    """
    main_url = BASE_URL + "/billtrack"

    try:
        res = requests.get(main_url, headers=HEADERS, timeout=20)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")

        bill_links = []
    
        for a in soup.find_all("a", href=True):
            href = a["href"]

            if (
                "/billtrack/" in href and
                "category" not in href and
                "field" not in href and
                href != "/billtrack"
            ):
                full_link = urljoin(BASE_URL, href)

                if full_link not in bill_links:
                    bill_links.append(full_link)

                if max_bill_pages is not None and len(bill_links) >= max_bill_pages:
                    break

        print(f"\nFound {len(bill_links)} bill links\n")

        results = []
        total = len(bill_links)

        for i, bill_url in enumerate(bill_links, start=1):
            print(f"[{i}/{total}] Visiting: {bill_url}")

            try:
                bill_res = requests.get(bill_url, headers=HEADERS, timeout=20)
                bill_res.raise_for_status()
                bill_soup = BeautifulSoup(bill_res.text, "html.parser")

                for a in bill_soup.find_all("a", href=True):
                    href = a["href"]

                    if ".pdf" in href.lower():
                        pdf = urljoin(BASE_URL, href)

                        results.append({
                            "pdf_url": pdf,
                            "title": pdf.split("/")[-1]
                        })

            except Exception as e:
                print("❌ Error:", bill_url, e)

        return results

    except Exception as e:
        print("❌ Error scraping:", e)
        return []