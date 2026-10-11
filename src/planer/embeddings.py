"""384-dimensional sentence embeddings with all-MiniLM-L6-v2 (FR-2)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .config import EMBED_DIM, settings


@lru_cache(maxsize=1)
def get_model():
    from sentence_transformers import SentenceTransformer  # heavy import, load lazily

    model = SentenceTransformer(settings.embed_model, device="cpu")
    dim = model.get_sentence_embedding_dimension()
    if dim != EMBED_DIM:
        raise ValueError(f"{settings.embed_model} produces {dim}-d vectors, expected {EMBED_DIM}")
    return model


def embed(texts: list[str], batch_size: int = 128, show_progress: bool = False) -> np.ndarray:
    """Unit-normalized float32 embeddings, shape (len(texts), 384). Empty strings are allowed."""
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)
    vecs = get_model().encode(
        [t or " " for t in texts],
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
    )
    return vecs.astype(np.float32)
