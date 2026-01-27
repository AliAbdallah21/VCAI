# orchestration/nodes/rag_node.py
"""
RAG (Retrieval-Augmented Generation) node for orchestration.
Retrieves relevant documents based on conversation context.

Used for:
1. AI Customer - Ask realistic, company-specific questions
2. Evaluation - Check if salesperson gave correct information
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
    1. Builds query from transcription + history context
    2. Retrieves relevant documents from vector store
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
        state["rag_context"] = {"query": "", "documents": [], "total_found": 0}
        if config.verbose:
            print("[RAG NODE] Skipped (disabled)")
        return state
    
    try:
        transcription = state.get("transcription", "")
        
        if not transcription or not transcription.strip():
            # No transcription - return empty context
            state["rag_context"] = {"query": "", "documents": [], "total_found": 0}
            if config and config.verbose:
                print("[RAG NODE] Skipped (no transcription)")
            return state
        
        # Get RAG function based on config
        if config and config.use_mocks:
            from orchestration.mocks import retrieve_context
        else:
            # Use REAL RAG!
            from rag.agent import retrieve_context
        
        # Build optimized query
        history = state.get("history", [])
        query = _build_rag_query(transcription, history)
        
        # Retrieve documents
        top_k = getattr(config, 'rag_top_k', 3) if config else 3
        rag_context = retrieve_context(query=query, top_k=top_k)
        
        # Update state
        state["rag_context"] = rag_context
        
        # Log timing
        elapsed = time.time() - start_time
        state["node_timings"]["rag"] = elapsed
        
        if config and config.verbose:
            query_preview = query[:50] + "..." if len(query) > 50 else query
            print(f"[RAG NODE] Query: '{query_preview}'")
            print(f"[RAG NODE] Found {rag_context.get('total_found', 0)} documents")
            
            # Show top result preview
            docs = rag_context.get("documents", [])
            if docs:
                top_doc = docs[0]
                score = top_doc.get("score", 0)
                source = top_doc.get("source", "unknown")
                print(f"[RAG NODE] Top result: {source} (score: {score:.3f})")
            
            print(f"[RAG NODE] Time: {elapsed:.3f}s")
        
    except ImportError as e:
        # RAG module not available - use empty context
        state["rag_context"] = {
            "query": state.get("transcription", ""),
            "documents": [],
            "total_found": 0
        }
        if config and config.verbose:
            print(f"[RAG NODE] Import error (using empty context): {str(e)}")
    
    except Exception as e:
        # Don't fail the whole pipeline for RAG errors
        state["rag_context"] = {
            "query": state.get("transcription", ""),
            "documents": [],
            "total_found": 0
        }
        if config and config.verbose:
            print(f"[RAG NODE] Error (using empty context): {str(e)}")
    
    return state


def _build_rag_query(transcription: str, history: list) -> str:
    """
    Build an optimized query for RAG retrieval.
    
    Incorporates context from conversation history to improve relevance.
    
    Args:
        transcription: Current transcription (salesperson's message)
        history: Conversation history
    
    Returns:
        str: Optimized query for vector search
    """
    # Start with the transcription
    query = transcription
    
    # Remove common Arabic filler words that don't help retrieval
    filler_words = [
        "أنا", "عايز", "محتاج", "يعني", "كده", "بس", "طيب",
        "أه", "لا", "ايوه", "معلش", "خلاص", "تمام", "حاضر",
        "ممكن", "لو سمحت", "من فضلك"
    ]
    
    for word in filler_words:
        # Remove with word boundaries to avoid partial matches
        query = query.replace(f" {word} ", " ")
        query = query.replace(f" {word}", "")
        query = query.replace(f"{word} ", "")
    
    # Clean up extra spaces
    query = " ".join(query.split())
    
    # If query is too short after cleaning, use original
    if len(query.strip()) < 5:
        query = transcription
    
    # Optionally add context from recent history
    # This helps when salesperson refers to something discussed earlier
    if history and len(history) >= 2:
        # Get last customer message for context
        recent_context = _extract_recent_topics(history[-4:])  # Last 2 turns
        if recent_context:
            query = f"{query} {recent_context}"
    
    return query.strip()


def _extract_recent_topics(recent_history: list) -> str:
    """
    Extract key topics from recent conversation history.
    
    Args:
        recent_history: Last few messages
    
    Returns:
        str: Key topics to add to query
    """
    # Keywords that indicate important topics
    topic_keywords = [
        # Property types
        "شقة", "فيلا", "دوبلكس", "ستوديو", "بنتهاوس",
        # Locations
        "التجمع", "مدينتي", "زايد", "أكتوبر", "العاصمة", "المعادي", "الزمالك",
        # Features
        "متر", "غرف", "سعر", "مليون", "تقسيط", "مقدم", "تشطيب",
        # Compounds
        "بالم", "ماونتن", "هيلز", "دريم"
    ]
    
    found_topics = []
    
    for msg in recent_history:
        text = msg.get("content", "") if isinstance(msg, dict) else str(msg)
        for keyword in topic_keywords:
            if keyword in text and keyword not in found_topics:
                found_topics.append(keyword)
    
    # Return top 3 topics to avoid query being too long
    return " ".join(found_topics[:3])


def format_rag_context_for_llm(rag_context: dict, max_docs: int = 3) -> str:
    """
    Format RAG context as a string for the LLM prompt.
    
    Args:
        rag_context: RAG context from retrieve_context()
        max_docs: Maximum documents to include
    
    Returns:
        str: Formatted context for LLM
    """
    if not rag_context or not rag_context.get("documents"):
        return ""
    
    docs = rag_context["documents"][:max_docs]
    
    if not docs:
        return ""
    
    formatted_parts = ["معلومات متاحة:"]
    
    for i, doc in enumerate(docs, 1):
        content = doc.get("content", "")
        source = doc.get("source", "unknown")
        
        # Truncate long content
        if len(content) > 300:
            content = content[:300] + "..."
        
        formatted_parts.append(f"\n[{i}] ({source}):\n{content}")
    
    return "\n".join(formatted_parts)