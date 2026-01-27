# rag/embeddings.py
from _future_ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

from shared.constants import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    # multilingual MiniLM works for English + Arabic
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_texts(texts: Iterable[str]) -> np.ndarray:
    """
    Returns float32 normalized embeddings for a list of texts.
    """
    arr = _model().encode(list(texts), normalize_embeddings=True, batch_size=64)
    return np.asarray(arr, dtype="float32")


def embed_query(query: str) -> np.ndarray:
    """
    Returns shape (1, d) float32 normalized embedding for a query.
    """
    return embed_texts([query])