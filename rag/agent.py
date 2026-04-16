# rag/agent.py
from __future__ import annotations

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


def retrieve_structured(query: str) -> dict | None:
    """
    FAISS semantic search → return full structured property record.

    Returns the best-matching property dict if score > 0.6, else None.
    Looks for hits that carry 'property_id' in their metadata (set by index_build.py).
    """
    if not query or not query.strip():
        return None

    from rag.structured_store import get_property_by_id

    hits = faiss_search(query, top_k=5)
    for hit in hits:
        if hit["score"] < 0.6:
            continue
        md = hit.get("metadata", {})
        property_id = md.get("property_id")
        if property_id:
            prop = get_property_by_id(property_id)
            if prop:
                return prop

    return None


def fact_check_transcript(transcript: list[dict]) -> dict:
    """
    Extract salesperson claims from a conversation transcript and verify them
    against the structured knowledge base.

    Each transcript entry: {speaker, text, turn_number}

    Returns the full fact-check result dict:
    {
        "claims_checked":      int,
        "accurate_count":      int,
        "inaccurate_count":    int,
        "accuracy_rate":       float,
        "errors":              list[dict],
        "property_mentions":   list[dict],
        "unverifiable_claims": list[dict],
    }
    """
    from rag.claim_extractor import extract_salesperson_claims
    from rag.fact_checker import check_facts
    from rag.structured_store import load_properties, load_policies

    claims = extract_salesperson_claims(transcript)

    if not claims:
        return {
            "claims_checked":      0,
            "accurate_count":      0,
            "inaccurate_count":    0,
            "accuracy_rate":       1.0,
            "errors":              [],
            "property_mentions":   [],
            "unverifiable_claims": [],
        }

    return check_facts(claims, load_properties(), load_policies())