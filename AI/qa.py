from __future__ import annotations

import os
from typing import Sequence

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def answer_question(query: str, context_chunks: Sequence[str]) -> str:
    """Answer a user question using ONLY the given context chunks."""

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
- STRICT OUTPUT FORMAT RULES:
    - Return plain text only (no markdown, no headings).
    - If using bullets, each bullet MUST be on its own new line.
    - Use only "- " as the bullet prefix.
    - Do NOT use "+", "*", "•", numbered bullets, or other symbols.
    - Do NOT include markdown like "**" or "###".
    - Do NOT wrap points in quotes unless absolutely necessary.
    - Do NOT merge multiple points into one line.
    - Keep each bullet to one sentence maximum.
    - Keep consistent spacing between lines.
    - Output must be clean plain text suitable for direct HTML rendering.
""".strip()

    models_to_try = ["llama-3.3-70b-versatile", "llama3-8b-8192", "mixtral-8x7b-32768"]
    for model_name in models_to_try:
        try:
            response = _client.chat.completions.create(
                model=model_name,
                max_tokens=512,
                temperature=0.2,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            text = response.choices[0].message.content or ""
            if text.strip():
                return text.strip()
        except Exception as exc:
            print(f"Q&A generation error with model {model_name}:", exc)
            continue

    return "I encountered an error while generating the answer. Please try again later."
