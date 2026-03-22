import json
from pathlib import Path
import fitz

from AI.chunker import chunk_text
from AI.compressor import compress_chunks_batch


def extract_text_from_pdf(pdf_path):
    pdf_path = Path(pdf_path)
    pages_text = []

    with fitz.open(pdf_path) as doc:
        for page in doc:
            text = page.get_text("text", sort=True).strip()
            if text:
                pages_text.append(text)

    return "\n\n".join(pages_text)


def process_pdf_to_chunks(pdf_path):
    text = extract_text_from_pdf(pdf_path)

    if not text:
        return None, None

    # 🔹 original chunks
    original_chunks = chunk_text(text, chunk_size=180, overlap=40)

    # 🔹 compressed chunks
    compressed_chunks = compress_chunks_batch(original_chunks)

    return original_chunks, compressed_chunks