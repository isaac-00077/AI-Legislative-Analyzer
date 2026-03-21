from sentence_transformers import SentenceTransformer

# 🔥 Load model ONCE (important)
model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text: str):
    try:
        if not text or not text.strip():
            return None

        embedding = model.encode(text)

        # convert numpy → list (pgvector accepts list)
        return embedding.tolist()

    except Exception as e:
        print("❌ Embedding error:", e)
        return None