import requests
from pathlib import Path
from urllib.parse import unquote

SAVE_DIR = Path("data/pdfs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

def download_pdf(url):
    try:
        # ✅ safer filename (decode %20 etc)
        filename = unquote(url.split("/")[-1]).replace("/", "_")
        path = SAVE_DIR / filename

        if path.exists():
            print("⚡ Already exists:", filename)
            return str(path)

        print("⬇️ Downloading:", url)

        res = requests.get(url, headers=HEADERS, timeout=10)

        if res.status_code != 200:
            print("❌ Failed:", url)
            return None

        with open(path, "wb") as f:
            f.write(res.content)

        return str(path)

    except Exception as e:
        print("Download error:", e)
        return None