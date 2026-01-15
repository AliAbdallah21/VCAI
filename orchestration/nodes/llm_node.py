# orchestration/nodes/llm_node.py
"""
LLM (Large Language Model) node for orchestration.
Generates VC response based on context.
"""

import time

from orchestration.state import ConversationState
from orchestration.config import OrchestrationConfig
from shared.exceptions import LLMGenerationError


def llm_node(
    state: ConversationState,
    config: OrchestrationConfig = None
) -> ConversationState:
    """
    Generate VC response using LLM.
    
    This node:
    1. Gathers all context (emotion, memory, RAG)
    2. Generates response using LLM
    3. Updates state with response
    
    Args:
        state: Current conversation state
        config: Orchestration configuration
    
    Returns:
        ConversationState: Updated state with LLM response
    """
    start_time = time.time()
    
    if config and config.verbose:
        print("\n[LLM NODE] Generating response...")
    
    try:
        transcription = state.get("transcription")
        
        if not transcription:
            raise LLMGenerationError("No transcription available")
        
        # Get LLM function
        if config and config.use_mocks:
            from orchestration.mocks import generate_response
        else:
            from llm.agent import generate_response
        
        # Gather context
        emotion = state.get("emotion") or {
            "primary_emotion": "neutral",
            "confidence": 0.5,
            "voice_emotion": "neutral",
            "text_emotion": "neutral",
            "intensity": "low",
            "scores": {}
        }
        
        emotional_context = state.get("emotional_context") or {
            "current": emotion,
            "trend": "stable",
            "recommendation": "be_professional",
            "risk_level": "low"
        }
        
        persona = state.get("persona") or {
            "id": "default",
            "name": "عميل افتراضي",
            "personality_prompt": "أنت عميل مصري بتدور على شقة"
        }
        
        memory = state.get("memory") or {
            "session_id": state.get("session_id", ""),
            "checkpoints": [],
            "recent_messages": [],
            "total_turns": 0
        }
        
        rag_context = state.get("rag_context") or {
            "query": transcription,
            "documents": [],
            "total_found": 0
        }
        
        # Generate response
        response = generate_response(
            customer_text=transcription,
            emotion=emotion,
            emotional_context=emotional_context,
            persona=persona,
            memory=memory,
            rag_context=rag_context
        )
        
        # Update state
        state["llm_response"] = response
        state["phase"] = "responding"
        
        # Log timing
        elapsed = time.time() - start_time
        state["node_timings"]["llm"] = elapsed
        
        if config and config.verbose:
            print(f"[LLM NODE] Response: '{response[:50]}...'")
            print(f"[LLM NODE] Time: {elapsed:.3f}s")
        
    except Exception as e:
        state["error"] = f"LLM Error: {str(e)}"
        state["llm_response"] = "عذراً، حصل مشكلة. ممكن تعيد الكلام؟"
        if config and config.verbose:
            print(f"[LLM NODE] Error (fallback response): {str(e)}")
    
    return state