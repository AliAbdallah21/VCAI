# rag/agent.py
from _future_ import annotations

from shared.exceptions import RAGIndexNotFoundError, RAGRetrievalError
from shared.types import RAGContext

from rag.vector_store import faiss_search


def retrieve_context(query: str, top_k: int = 3) -> RAGContext:
    """
    INPUT:
      query: str
      top_k: int (1-10)

    OUTPUT (must match interface guide):
      RAGContext: {
        "query": str,
        "documents": [
          {"content": str, "source": str, "score": float, "metadata": dict}
        ],
        "total_found": int
      }
    """
    if not query or not query.strip():
        return {"query": query, "documents": [], "total_found": 0}

    if top_k < 1:
        top_k = 1
    if top_k > 10:
        top_k = 10

    hits = faiss_search(query, top_k=top_k)

    docs = []
    for h in hits:
        md = h.get("metadata") or {}
        source = md.get("source") or md.get("path") or "unknown"
        docs.append(
            {
                "content": h["content"],
                "source": str(source),
                "score": float(h["score"]),
                "metadata": dict(md),
            }
        )

    return {
        "query": query,
        "documents": docs,
        "total_found": len(docs),
    }