import re


def clean_text(text):
    if not text:
        return ""

    # Join words split by line-break hyphenation and normalize whitespace.
    text = re.sub(r"(?<=\w)-\s+(?=\w)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def chunk_text(text, chunk_size=150, overlap=30):
    text = clean_text(text)
    if not text:
        return []

    chunk_size = max(1, int(chunk_size))
    overlap = max(0, int(overlap))
    if overlap >= chunk_size:
        overlap = chunk_size - 1

    words = text.split()
    if not words:
        return []

    sentence_end_indices = {
        index for index, word in enumerate(words) if re.search(r"[.!?][\"')\]]?$", word)
    }

    chunks = []
    start = 0
    max_sentence_lookahead = max(5, min(30, chunk_size // 4))

    while start < len(words):
        base_end = min(start + chunk_size, len(words))
        end = base_end

        # Prefer ending on a sentence boundary if it is close enough.
        if base_end < len(words):
            upper = min(len(words) - 1, base_end + max_sentence_lookahead)
            for idx in range(base_end - 1, upper + 1):
                if idx in sentence_end_indices:
                    end = idx + 1
                    break

        chunk_words = words[start:end]
        if chunk_words:
            chunks.append(" ".join(chunk_words).strip())

        if end >= len(words):
            break

        # Enforce exact word overlap between adjacent chunks.
        start = max(0, end - overlap)

    return chunks