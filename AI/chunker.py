def clean_text(text):
    return text.replace("\n", " ").replace("  ", " ").strip()


import re

def chunk_text(text, chunk_size=150, overlap=30):
    # Split into sentences
    sentences = re.split(r'(?<=[.!?]) +', text)

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        words = sentence.split()
        length = len(words)

        if current_length + length > chunk_size:
            chunks.append(" ".join(current_chunk))

            # overlap
            current_chunk = current_chunk[-overlap:]
            current_length = len(current_chunk)

        current_chunk.extend(words)
        current_length += length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks