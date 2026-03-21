import os
import fitz
from chunker import chunk_text

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""

        for page in doc:
            text += page.get_text()

        return text

    except Exception as e:
        print("❌ Error extracting text:", e)
        return ""


# 🔥 FIXED PATH HANDLING
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

path = os.path.join(
    BASE_DIR,
    "data",
    "pdfs",
    "the_cable_television_networks_(regulation).pdf"
)

text = extract_text_from_pdf(path)
chunks = chunk_text(text)

print("✅ TOTAL CHUNKS:", len(chunks))

for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- CHUNK {i+1} ---\n")
    print(chunk[:300])