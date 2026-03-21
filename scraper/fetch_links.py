import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://prsindia.org"

HEADERS = {"User-Agent": "Mozilla/5.0"}

def get_pdf_links(num_bill_links=5, initial=True):
    main_url = BASE_URL + "/billtrack"

    try:
        res = requests.get(main_url, headers=HEADERS)
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

                if initial and len(bill_links) >= num_bill_links:
                    break

        print(f"\n📄 Found {len(bill_links)} bill links\n")

        results = []
        total = len(bill_links)

        for i, bill_url in enumerate(bill_links, start=1):
            print(f"[{i}/{total}] 🔗 Visiting: {bill_url}")

            try:
                bill_res = requests.get(bill_url, headers=HEADERS)
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