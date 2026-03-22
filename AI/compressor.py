from dotenv import load_dotenv
import os
from groq import Groq
import re

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def compress_chunks_batch(chunks, model="llama-3.1-8b-instant", batch_size=5):
    compressed = []
    total = len(chunks)

    for i in range(0, total, batch_size):
        batch = chunks[i:i+batch_size]

        print(f"  Compressing batch {i//batch_size + 1} ({len(batch)} chunks)...")

        # 🔹 Step 1: Number chunks
        numbered_text = ""
        for idx, chunk in enumerate(batch, start=1):
            numbered_text += f"[CHUNK {idx}]\n{chunk}\n\n"

        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": f"""
You will receive multiple chunks labeled as [CHUNK 1], [CHUNK 2], etc.

TASK:
Compress each chunk into a concise, structured, and query-friendly form.

IMPORTANT GOAL:
Each chunk must independently answer user queries.

RULES:
- Preserve ALL key legal facts (acts, sections, dates, purpose)
- Remove repetition, filler words, and unnecessary legal phrasing
- DO NOT merge chunks
- DO NOT skip chunks
- Maintain SAME numbering
- Keep output concise (target ~40–60% shorter than input)

STRICT ACCURACY RULES:
- DO NOT hallucinate or invent facts
- Preserve exact dates, numbers, and legal references
- Use ONLY information present in the input

QUALITY RULES:
- Avoid repeating the same information across chunks
- Each bullet must be specific and meaningful
- Prefer precise legal wording over vague summaries
- Avoid repeating the same facts across chunks unless necessary
- Ensure no broken words or split numbers (e.g., "1 926" → "1926")

STYLE (STRICT):
- Use ONLY bullet points (no paragraphs)
- Each chunk should have 3–6 bullet points max
- Each bullet must be short (1 line preferred)

INCLUDE (if present):
- Bill name
- Effective dates
- Sections / clauses
- Acts affected
- Purpose / reason

AVOID:
- Long sentences
- Repetition
- Decorative or narrative text

OUTPUT FORMAT (STRICT):

[CHUNK 1]
- point
- point
- point

[CHUNK 2]
- point
- point

[CHUNK 3]
- point
- point

Text:
{numbered_text}
"""
                    }
                ],
            )

            output = response.choices[0].message.content.strip()

            # 🔹 Extract chunks
            pattern = r"\[CHUNK (\d+)\]\s*(.*?)(?=\n\[CHUNK|\Z)"
            matches = re.findall(pattern, output, re.DOTALL)

            # 🔹 Convert to dict
            parsed = {}
            for num, text in matches:
                parsed[int(num)] = text.strip()

            # 🔥 PARTIAL RECOVERY LOGIC (MAIN FIX)
            for idx in range(1, len(batch) + 1):
                if idx in parsed and parsed[idx]:
                    compressed.append(parsed[idx])
                else:
                    print(f"⚠️ Missing CHUNK {idx}, using original")
                    compressed.append(batch[idx - 1])

        except Exception as e:
            print(f"Compression error: {e}")
            compressed.extend(batch)

    return compressed