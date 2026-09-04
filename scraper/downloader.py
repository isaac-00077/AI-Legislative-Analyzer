import requests
from pathlib import Path
from urllib.parse import unquote
import urllib3

# Suppress SSL warnings when falling back to verify=False for gov portals
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

SAVE_DIR = Path("data/pdfs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/pdf,application/xhtml+xml,text/html;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

def download_pdf(url: str, timeout: int = 15):
    try:
        filename = unquote(url.split("/")[-1]).replace("/", "_")
        if not filename.endswith(".pdf"):
            filename += ".pdf"
        path = SAVE_DIR / filename

        if path.exists() and path.stat().st_size > 0:
            print(" Already exists:", filename)
            return str(path)

        print("⬇️ Downloading:", url)

        # Attempt 1: Standard HTTPS request
        try:
            res = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        except requests.exceptions.SSLError:
            # Fallback 1: Try without SSL verification for gov portals with self-signed/expired certs
            res = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True, verify=False)
        except requests.exceptions.RequestException:
            # Retry attempt
            res = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True, verify=False)

        if res.status_code != 200:
            print(f"❌ Failed (HTTP {res.status_code}):", url)
            return None

        with open(path, "wb") as f:
            f.write(res.content)

        return str(path)

    except Exception as e:
        print("Download error:", url, e)
        return None