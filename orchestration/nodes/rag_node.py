# orchestration/nodes/rag_node.py
"""
RAG (Retrieval-Augmented Generation) node for orchestration.
Retrieves relevant documents based on conversation context.
"""

import time

from orchestration.state import ConversationState
from orchestration.config import OrchestrationConfig
from shared.exceptions import RAGRetrievalError


def rag_node(
    state: ConversationState,
    config: OrchestrationConfig = None
) -> ConversationState:
    """
    Retrieve relevant documents for context.
    
    This node:
    1. Builds query from transcription
    2. Retrieves relevant documents
    3. Updates state with RAG context
    
    Args:
        state: Current conversation state
        config: Orchestration configuration
    
    Returns:
        ConversationState: Updated state with RAG context
    """
    start_time = time.time()
    
    if config and config.verbose:
        print("\n[RAG NODE] Retrieving context...")
    
    # Skip if RAG is disabled
    if config and not config.enable_rag:
        if config.verbose:
            print("[RAG NODE] Skipped (disabled)")
        return state
    
    try:
        transcription = state.get("transcription")
        
        if not transcription:
            raise RAGRetrievalError("No transcription available", query=None)
        
        # Get RAG function
        if config and config.use_mocks:
            from orchestration.mocks import retrieve_context
        else:
            from rag.agent import retrieve_context
        
        # Build query (could be enhanced with history context)
        query = _build_rag_query(transcription, state.get("history", []))
        
        # Retrieve documents
        top_k = config.rag_top_k if config else 3
        rag_context = retrieve_context(query=query, top_k=top_k)
        
        # Update state
        state["rag_context"] = rag_context
        
        # Log timing
        elapsed = time.time() - start_time
        state["node_timings"]["rag"] = elapsed
        
        if config and config.verbose:
            print(f"[RAG NODE] Query: '{query[:50]}...'")
            print(f"[RAG NODE] Found {len(rag_context['documents'])} documents")
            print(f"[RAG NODE] Time: {elapsed:.3f}s")
        
    except Exception as e:
        # Don't fail the whole pipeline for RAG errors
        state["rag_context"] = {
            "query": state.get("transcription", ""),
            "documents": [],
            "total_found": 0
        }
        if config and config.verbose:
            print(f"[RAG NODE] Error (empty context): {str(e)}")
    
    return state


def _build_rag_query(transcription: str, history: list) -> str:
    """
    Build an optimized query for RAG retrieval.
    
    Can incorporate context from conversation history
    to improve relevance.
    
    Args:
        transcription: Current transcription
        history: Conversation history
    
    Returns:
        str: Optimized query
    """
    # For now, just use the transcription
    # Could be enhanced to:
    # - Extract key topics from history
    # - Focus on specific entities mentioned
    # - Remove filler words
    
    query = transcription
    
    # Remove common filler words (simple approach)
    filler_words = ["أنا", "عايز", "محتاج", "يعني", "كده"]
    for word in filler_words:
        query = query.replace(word, "")
    
    # Clean up extra spaces
    query = " ".join(query.split())
    
    return query if query.strip() else transcription