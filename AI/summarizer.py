from __future__ import annotations

import os
from typing import Sequence

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_summary(compressed_chunks: Sequence[str], max_chunks: int = 5) -> str | None:
    """Generate a short public-facing summary from compressed chunks.

    Uses the first few compressed chunks to build a 5–7 bullet-point summary
    suitable for the public dashboard.
    """

    if not compressed_chunks:
        return None

    head = list(compressed_chunks[:max_chunks])
    context = "\n\n".join(head)

    prompt = f"""
You are helping to summarize an Indian legislative bill for the public.

CONTEXT (compressed chunks):
{context}

TASK:
- Write a clear, neutral summary in 5–7 bullet points.
- Target an informed but non-expert citizen.
- Focus on: purpose of the bill, who is affected, key provisions, and major changes.
- Do NOT mention that this is a summary of chunks.
- Do NOT reference this prompt or the word 'chunk'.
- Avoid legal jargon where possible; keep it simple and precise.
""".strip()

    try:
        response = _client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=400,
            temperature=0.3,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        text = response.choices[0].message.content or ""
        return text.strip() or None
    except Exception as exc:  # pragma: no cover
        print("Summary generation error:", exc)
        return None
