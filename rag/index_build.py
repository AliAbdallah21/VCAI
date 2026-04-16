# rag/index_build.py
"""
Build FAISS index directly from data/documents/ — no Chroma required.

Saves:
  data/faiss_index/index.faiss     — FAISS IndexFlatIP (cosine via L2-normalized vectors)
  data/faiss_index/documents.json  — chunk text + metadata, indexed by position

Usage:
  python rag/index_build.py
  # or from within Python:
  from rag.index_build import build_index
  result = build_index()
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path

import faiss
import numpy as np

# Allow `python rag/index_build.py` from project root
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.config import FAISS_DIR, FAISS_DIRECT_INDEX_PATH, FAISS_DIRECT_DOCS_PATH
from rag.document_loader import load_chunks, RawChunk
from rag.embeddings import embed_texts

_log = logging.getLogger(__name__)

_PROPERTIES_PATH = Path("data/documents/properties.json")
_POLICIES_PATH   = Path("data/documents/policies.json")


def _chunk_id(content: str, source: str, suffix: str) -> str:
    """Generate a short stable ID for a structured chunk."""
    raw = f"{source}:{suffix}:{content[:80]}"
    return hashlib.md5(raw.encode()).hexdigest()[:16] + f"_{suffix}"


def _build_property_chunks(path: Path) -> list[RawChunk]:
    """
    One rich searchable chunk per property with 'property_id' in metadata.

    Content combines: name_ar | name_en | location_ar | developer |
                      keywords_ar | keywords_en | price info | features
    This is used by fact_checker.py to resolve which property a claim refers to.
    """
    if not path.exists():
        return []

    chunks: list[RawChunk] = []
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    for prop in data.get("properties", []):
        pid = prop.get("id", "")
        pmin = prop.get("price_min", 0)
        pmax = prop.get("price_max", 0)

        parts = [
            prop.get("name_ar", ""),
            prop.get("name_en", ""),
            prop.get("location_ar", ""),
            prop.get("location_en", ""),
            prop.get("developer", ""),
            prop.get("city", ""),
            " ".join(prop.get("keywords_ar", [])),
            " ".join(prop.get("keywords_en", [])),
            f"سعر من {pmin:,} إلى {pmax:,} جنيه",
            f"price from {pmin:,} to {pmax:,} EGP",
            " ".join(prop.get("features", [])),
            prop.get("notes", ""),
        ]
        content = " | ".join(p for p in parts if p)

        chunks.append(
            RawChunk(
                id=_chunk_id(content, "properties.json", pid),
                content=content,
                source="properties.json",
                metadata={
                    "source":      "properties.json",
                    "property_id": pid,
                    "type":        "property_structured",
                    "name_ar":     prop.get("name_ar", ""),
                    "city":        prop.get("city", ""),
                },
            )
        )

    return chunks


def _build_policy_chunks(path: Path) -> list[RawChunk]:
    """
    One rich searchable chunk per policy with 'policy_id' in metadata.
    """
    if not path.exists():
        return []

    chunks: list[RawChunk] = []
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    for pol in data.get("policies", []):
        pol_id = pol.get("id", "")

        rule_texts_ar = [r.get("description_ar", "") for r in pol.get("rules", [])]
        rule_texts_en = [r.get("description_en", "") for r in pol.get("rules", [])]

        parts = [
            pol.get("title_ar", ""),
            pol.get("title_en", ""),
            pol.get("category", ""),
            " ".join(pol.get("keywords_ar", [])),
            " ".join(pol.get("keywords_en", [])),
            " ".join(r for r in rule_texts_ar if r),
            " ".join(r for r in rule_texts_en if r),
        ]
        content = " | ".join(p for p in parts if p)

        chunks.append(
            RawChunk(
                id=_chunk_id(content, "policies.json", pol_id),
                content=content,
                source="policies.json",
                metadata={
                    "source":    "policies.json",
                    "policy_id": pol_id,
                    "type":      "policy_structured",
                    "title_ar":  pol.get("title_ar", ""),
                    "category":  pol.get("category", ""),
                },
            )
        )

    return chunks


def build_index() -> dict:
    """
    Load all documents from data/documents/, embed them, and persist a FAISS index.

    Structured JSON files (properties.json, policies.json) are indexed twice:
      - Generic chunks via document_loader (rich raw content, no structured IDs)
      - Structured chunks (rich searchable text + property_id/policy_id in metadata)
    The structured chunks enable fact_checker.py to resolve property/policy IDs
    directly from FAISS search results.

    Returns:
        {"chunks": int, "path": str | None}
    """
    FAISS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Generic chunks from all files under data/documents/
    generic_chunks = load_chunks()

    # 2. Structured property / policy chunks with IDs baked into metadata
    structured_chunks: list[RawChunk] = []
    structured_chunks.extend(_build_property_chunks(_PROPERTIES_PATH))
    structured_chunks.extend(_build_policy_chunks(_POLICIES_PATH))

    if structured_chunks:
        _log.info(
            "[RAG] Built %d structured chunks (%d properties, %d policies)",
            len(structured_chunks),
            len(_build_property_chunks(_PROPERTIES_PATH)),  # informational only
            len(_build_policy_chunks(_POLICIES_PATH)),
        )

    all_chunks = generic_chunks + structured_chunks

    if not all_chunks:
        _log.warning("[RAG] No documents found in data/documents/ — index not built.")
        return {"chunks": 0, "path": None}

    # 3. Embed all chunk texts (returns L2-normalised float32 array)
    texts = [c.content for c in all_chunks]
    _log.info("[RAG] Embedding %d chunks with sentence-transformers...", len(texts))
    vectors = embed_texts(texts)          # shape: (N, d), dtype float32, normalised

    # 4. Build FAISS index (IndexFlatIP = dot-product on normalised ⟹ cosine similarity)
    d = int(vectors.shape[1])
    index = faiss.IndexFlatIP(d)
    index.add(vectors)

    # 5. Persist index file
    faiss.write_index(index, str(FAISS_DIRECT_INDEX_PATH))
    _log.info("[RAG] Saved FAISS index → %s", FAISS_DIRECT_INDEX_PATH)

    # 6. Persist document metadata as JSON (position-aligned with vectors)
    docs = [
        {
            "id":      c.id,
            "content": c.content,
            "source":  c.source,
            "metadata": {"source": c.source, **(c.metadata or {})},
        }
        for c in all_chunks
    ]
    with open(FAISS_DIRECT_DOCS_PATH, "w", encoding="utf-8") as fh:
        json.dump(docs, fh, ensure_ascii=False, indent=2)
    _log.info("[RAG] Saved document metadata → %s", FAISS_DIRECT_DOCS_PATH)

    structured_count = len(structured_chunks)
    generic_count    = len(generic_chunks)
    _log.info(
        "[RAG] Index complete: %d total (%d generic + %d structured)",
        len(all_chunks), generic_count, structured_count,
    )

    return {"chunks": len(all_chunks), "path": str(FAISS_DIRECT_INDEX_PATH)}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    result = build_index()
    if result["chunks"] > 0:
        print(f"\nIndex built: {result['chunks']} chunks")
        print(f"   Index : {FAISS_DIRECT_INDEX_PATH}")
        print(f"   Docs  : {FAISS_DIRECT_DOCS_PATH}")
    else:
        print("\nNo documents found -- check data/documents/")
        sys.exit(1)
