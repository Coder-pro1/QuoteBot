"""
Shared Sentence Encoder Singleton.
Loads the embedding model ONCE across all modules. Both VectorDBManager
and QuoteDBManager import this instead of each instantiating their own
SentenceTransformer — cuts startup time and RAM in half.
"""
from sentence_transformers import SentenceTransformer

_encoder: SentenceTransformer | None = None
_MODEL_NAME = "all-MiniLM-L6-v2"


def get_encoder() -> SentenceTransformer:
    """Returns the global singleton encoder, loading it on first call."""
    global _encoder
    if _encoder is None:
        print(f"[Encoder] Loading shared model ({_MODEL_NAME})...")
        _encoder = SentenceTransformer(_MODEL_NAME)
    return _encoder
