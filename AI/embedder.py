"""Embedding helper.

On Render Free Tier (512 MB RAM) the SentenceTransformer model cannot
load without OOM-killing the process.  Until the deployment target has
enough memory, this module is a deliberate **no-op** that always returns
``None``.  Every call-site already handles ``None`` by falling through
to plain-text chunk retrieval, so the Q&A pipeline works correctly
without embeddings.
"""


def get_embedding(text: str):
    """Return ``None`` – embedding is disabled on low-memory hosts."""
    return None