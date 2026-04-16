# rag/vector_store.py
from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any

import chromadb
import faiss
import numpy as np

from rag.config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    FAISS_INDEX_PATH,
    FAISS_META_PATH,
    FAISS_DIRECT_INDEX_PATH,
    FAISS_DIRECT_DOCS_PATH,
)
from rag.document_loader import load_chunks, RawChunk
from rag.embeddings import embed_texts, embed_query
from shared.exceptions import RAGIndexNotFoundError, RAGRetrievalError

# Optional: silence telemetry warnings
os.environ["CHROMADB_DISABLE_TELEMETRY"] = "1"

_log = logging.getLogger(__name__)

# Module-level cache: (faiss.Index, list[dict]) — populated on first search
_INDEX_CACHE: tuple | None = None


def _chroma_client() -> chromadb.PersistentClient:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def _clean_meta(d: dict) -> dict:
    """
    Chroma metadata cannot contain None or complex types.
    """
    out: dict[str, Any] = {}
    for k, v in d.items():
        if v is None:
            out[k] = 0 if k == "page" else ""
        elif isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def get_collection():
    client = _chroma_client()
    return client.get_or_create_collection(CHROMA_COLLECTION)


def build_chroma_index() -> dict:
    """
    Loads docs from data/documents, chunks them, embeds them,
    and stores everything in Chroma (persistent).
    """
    try:
        col = get_collection()
        chunks = load_chunks()

        if not chunks:
            # Don’t crash: just keep collection empty
            return {"inserted": 0, "message": "No documents found in data/documents."}

        ids = [c.id for c in chunks]
        docs = [c.content for c in chunks]
        metas = [
            _clean_meta(
                {
                    "source": c.source,  # required for citations in output
                    **(c.metadata or {}),
                }
            )
            for c in chunks
        ]

        X = embed_texts(docs)  # normalized float32
        col.add(ids=ids, documents=docs, metadatas=metas, embeddings=X)

        return {"inserted": len(ids), "message": f"Inserted {len(ids)} chunks into Chroma."}
    except Exception as e:
        raise RAGRetrievalError("Failed to build Chroma index", query=None, original_error=e)


def export_faiss_from_chroma(batch_limit: int = 2000) -> dict:
    """
    Exports all vectors/docs/metas from Chroma and writes:
      - build/faiss_from_chroma.index
      - build/faiss_from_chroma_meta.pkl
    """
    try:
        col = get_collection()
        total = col.count()

        all_ids: list[str] = []
        all_vecs: list[list[float]] = []
        all_texts: list[str] = []
        all_metas: list[dict] = []

        offset = 0
        while offset < total:
            batch = col.get(
                include=["embeddings", "documents", "metadatas"],
                limit=min(batch_limit, total - offset),
                offset=offset,
            )
            # ids are always returned
            all_ids.extend(batch["ids"])
            all_vecs.extend(batch["embeddings"])
            all_texts.extend(batch["documents"])
            all_metas.extend(batch["metadatas"])
            offset += len(batch["ids"])

        if not all_ids:
            raise RAGIndexNotFoundError("FAISS export files not found")

        vecs = np.asarray(all_vecs, dtype="float32")
        d = int(vecs.shape[1])

        # Using IP because we normalized embeddings => cosine similarity
        index = faiss.IndexFlatIP(d)
        index.add(vecs)

        FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(FAISS_INDEX_PATH))

        with open(FAISS_META_PATH, "wb") as f:
            pickle.dump({"ids": all_ids, "texts": all_texts, "metas": all_metas}, f)

        return {"exported": len(all_ids), "index": str(FAISS_INDEX_PATH), "meta": str(FAISS_META_PATH)}
    except RAGIndexNotFoundError:
        raise
    except Exception as e:
        raise RAGRetrievalError("Failed to export FAISS from Chroma", query=None, original_error=e)


def _load_faiss() -> tuple[faiss.Index, dict]:
    """Load legacy Chroma-exported FAISS index (pickle-based)."""
    if not FAISS_INDEX_PATH.exists() or not FAISS_META_PATH.exists():
        raise RAGIndexNotFoundError("FAISS export files not found")

    index = faiss.read_index(str(FAISS_INDEX_PATH))
    meta = pickle.load(open(FAISS_META_PATH, "rb"))
    return index, meta


# ---------------------------------------------------------------------------
# Direct FAISS index (built by rag/index_build.py — preferred path)
# ---------------------------------------------------------------------------

def _load_direct_faiss() -> tuple[faiss.Index, list[dict]]:
    """Load the direct FAISS index and its JSON document store."""
    if not FAISS_DIRECT_INDEX_PATH.exists() or not FAISS_DIRECT_DOCS_PATH.exists():
        raise RAGIndexNotFoundError("Direct FAISS index not found — run rag/index_build.py")

    index = faiss.read_index(str(FAISS_DIRECT_INDEX_PATH))
    with open(FAISS_DIRECT_DOCS_PATH, "r", encoding="utf-8") as fh:
        docs = json.load(fh)
    return index, docs


def _get_index() -> tuple[faiss.Index, list[dict]]:
    """
    Return the cached direct FAISS index, loading or auto-building it as needed.

    Startup behaviour:
      - index.faiss exists  → load and log "RAG: Loaded X documents from index"
      - index.faiss missing → build via index_build.build_index(), then load
                              and log "RAG: Built new index from X documents"
    """
    global _INDEX_CACHE
    if _INDEX_CACHE is not None:
        return _INDEX_CACHE

    if FAISS_DIRECT_INDEX_PATH.exists() and FAISS_DIRECT_DOCS_PATH.exists():
        index, docs = _load_direct_faiss()
        msg = f"RAG: Loaded {len(docs)} documents from index"
        _log.info(msg)
        print(f"[RAG] Loaded {len(docs)} documents from index")
    else:
        _log.info("RAG: index.faiss not found — building now from data/documents/...")
        print("[RAG] index.faiss not found — building now from data/documents/...")
        from rag.index_build import build_index
        result = build_index()
        if result["chunks"] == 0:
            raise RAGIndexNotFoundError(
                "Cannot build RAG index: no documents found in data/documents/"
            )
        index, docs = _load_direct_faiss()
        msg = f"RAG: Built new index from {len(docs)} documents"
        _log.info(msg)
        print(f"[RAG] Built new index from {len(docs)} documents")

    _INDEX_CACHE = (index, docs)
    return _INDEX_CACHE


def faiss_search(query: str, top_k: int = 3) -> list[dict]:
    """
    Search the direct FAISS index (auto-built on first call if missing).

    Returns list of hits:
      { id, score, content, metadata }
    """
    try:
        index, docs = _get_index()
        qv = embed_query(query)        # (1, d) float32 normalized
        scores, idxs = index.search(qv, top_k)

        hits: list[dict] = []
        for s, i in zip(scores[0], idxs[0]):
            if int(i) == -1:
                continue
            doc = docs[int(i)]
            hits.append(
                {
                    "id": doc["id"],
                    "score": float(s),
                    "content": doc["content"],
                    "metadata": doc.get("metadata", {}),
                }
            )
        return hits
    except RAGIndexNotFoundError:
        raise
    except Exception as e:
        raise RAGRetrievalError("FAISS search failed", query=query, original_error=e)