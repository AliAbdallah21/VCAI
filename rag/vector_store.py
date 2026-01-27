# rag/vector_store.py
from _future_ import annotations

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
)
from rag.document_loader import load_chunks, RawChunk
from rag.embeddings import embed_texts, embed_query
from shared.exceptions import RAGIndexNotFoundError, RAGRetrievalError

# Optional: silence telemetry warnings
os.environ["CHROMADB_DISABLE_TELEMETRY"] = "1"


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
    if not FAISS_INDEX_PATH.exists() or not FAISS_META_PATH.exists():
        raise RAGIndexNotFoundError("FAISS export files not found")

    index = faiss.read_index(str(FAISS_INDEX_PATH))
    meta = pickle.load(open(FAISS_META_PATH, "rb"))
    return index, meta


def faiss_search(query: str, top_k: int = 3) -> list[dict]:
    """
    Returns list of hits:
      { id, score, content, metadata }
    """
    try:
        index, meta = _load_faiss()
        qv = embed_query(query)  # (1, d) float32 normalized
        scores, idxs = index.search(qv, top_k)

        ids = meta["ids"]
        texts = meta["texts"]
        metas = meta["metas"]

        hits: list[dict] = []
        for s, i in zip(scores[0], idxs[0]):
            if int(i) == -1:
                continue
            m = metas[int(i)] or {}
            hits.append(
                {
                    "id": ids[int(i)],
                    "score": float(s),
                    "content": texts[int(i)],
                    "metadata": dict(m),
                }
            )
        return hits
    except RAGIndexNotFoundError:
        raise
    except Exception as e:
        raise RAGRetrievalError("FAISS search failed", query=query, original_error=e)