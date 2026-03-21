from __future__ import annotations

import os
from typing import Sequence

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def answer_question(query: str, context_chunks: Sequence[str]) -> str:
    """Answer a user question using ONLY the given context chunks.

    Falls back to an explicit "not enough information" style response when
    the context is empty.
    """

    if not context_chunks:
        return "I do not have enough information in the documents to answer that question."

    context = "\n\n".join(context_chunks)

    prompt = f"""
You are an AI assistant answering questions about Indian legislative bills.

CONTEXT:
{context}

USER QUESTION:
{query}

INSTRUCTIONS:
- Answer ONLY using the information given in CONTEXT.
- If the context is insufficient, clearly say you do not have enough information.
- Do NOT hallucinate or invent details.
- Be clear, concise, and neutral.
- Use short paragraphs or bullet points when helpful.
""".strip()

    try:
        response = _client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=512,
            temperature=0.2,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        text = response.choices[0].message.content or ""
        return text.strip()
    except Exception as exc:  # pragma: no cover
        print("Q&A generation error:", exc)
        return "I encountered an error while generating the answer. Please try again later."
