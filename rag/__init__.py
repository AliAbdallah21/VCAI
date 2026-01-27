# rag/_init_.py
"""
RAG (Retrieval Augmented Generation) Module.

Provides document retrieval for:
1. AI Customer - Ask realistic, company-specific questions
2. Evaluation - Check if salesperson gave correct information

Usage:
    from rag import retrieve_context
    
    # Get relevant documents for a query
    context = retrieve_context("سعر شقة في التجمع")
    print(context["documents"])
"""

from rag.agent import retrieve_context
from rag.vector_store import (
    build_chroma_index,
    export_faiss_from_chroma,
    faiss_search,
    get_collection
)
from rag.document_loader import load_chunks, RawChunk
from rag.embeddings import embed_texts, embed_query

_all_ = [
    # Main interface
    "retrieve_context",
    
    # Index building
    "build_chroma_index",
    "export_faiss_from_chroma",
    
    # Search
    "faiss_search",
    "get_collection",
    
    # Document loading
    "load_chunks",
    "RawChunk",
    
    # Embeddings
    "embed_texts",
    "embed_query",
]