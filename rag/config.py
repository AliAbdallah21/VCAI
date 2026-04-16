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

# Where FAISS export lives (built from Chroma — legacy)
FAISS_INDEX_PATH = Path("build/faiss_from_chroma.index")
FAISS_META_PATH = Path("build/faiss_from_chroma_meta.pkl")

# Direct FAISS index (built by rag/index_build.py — preferred)
FAISS_DIR = Path("data/faiss_index")
FAISS_DIRECT_INDEX_PATH = FAISS_DIR / "index.faiss"
FAISS_DIRECT_DOCS_PATH = FAISS_DIR / "documents.json"

DEFAULT_TOP_K = RAG_TOP_K