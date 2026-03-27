import threading
import traceback


# Lazily loaded global model instance so that importing this module
# does not block app startup. Both the heavy library import and the
# model construction happen only on first use.
_model = None
_model_lock = threading.Lock()


def _get_model():
    global _model

    # Fast path without acquiring the lock when the model is ready.
    if _model is not None:
        return _model

    with _model_lock:
        if _model is None:
            # Local import so that uvicorn startup does not pay the
            # cost of loading sentence_transformers and its deps.
            try:
                from sentence_transformers import SentenceTransformer

                print("🧠 Loading embedding model: all-MiniLM-L6-v2")
                _model = SentenceTransformer("all-MiniLM-L6-v2")
                print("✅ Embedding model loaded")
            except Exception as exc:
                print("❌ Failed to load embedding model:", exc)
                traceback.print_exc()
                raise
        return _model


def get_embedding(text: str):
    try:
        if not text or not text.strip():
            return None

        model = _get_model()
        embedding = model.encode(text)

        # convert numpy → list (pgvector accepts list)
        return embedding.tolist()

    except Exception as e:
        print("❌ Embedding error:", e)
        traceback.print_exc()
        return None