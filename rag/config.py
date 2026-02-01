# rag/config.py
from __future__ import annotations

from pathlib import Path

from shared.constants import DOCUMENTS_DIR, RAG_TOP_K

# Where your raw docs live (you said you'll add mock docs later)
DOCS_DIR = Path(DOCUMENTS_DIR)

# Where Chroma persists embeddings
CHROMA_DIR = Path("build/chroma")

# Chroma collection name
CHROMA_COLLECTION = "vcai_rag"

# Where FAISS export lives (built from Chroma)
FAISS_INDEX_PATH = Path("build/faiss_from_chroma.index")
FAISS_META_PATH = Path("build/faiss_from_chroma_meta.pkl")

DEFAULT_TOP_K = RAG_TOP_K