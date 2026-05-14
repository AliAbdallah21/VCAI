# orchestration/nodes/llm_node.py
"""
LLM node for orchestration.
Supports both full and streaming response generation.
"""

import time

from orchestration.state import ConversationState
from orchestration.config import OrchestrationConfig
from shared.exceptions import LLMGenerationError


def _gather_context(state):
    """Gather all context needed for LLM generation."""
    emotion = state.get("emotion") or {
        "primary_emotion": "neutral", "confidence": 0.5,
        "voice_emotion": "neutral", "text_emotion": "neutral",
        "intensity": "low", "scores": {}
    }
    emotional_context = state.get("emotional_context") or {
        "current": emotion, "trend": "stable",
        "recommendation": "be_professional", "risk_level": "low"
    }
    persona = state.get("persona") or {
        "id": "default", "name": "عميل افتراضي",
        "personality_prompt": "أنت عميل مصري بتدور على شقة"
    }
    memory = state.get("memory") or {
        "session_id": state.get("session_id", ""),
        "checkpoints": [], "recent_messages": [], "total_turns": 0
    }
    rag_context = state.get("rag_context") or {
        "query": state.get("transcription", ""), "documents": [], "total_found": 0
    }
    return emotion, emotional_context, persona, memory, rag_context


def llm_node(
    state: ConversationState,
    config: OrchestrationConfig = None
) -> ConversationState:
    """
    Generate VC response using LLM (non-streaming, backward compatible).
    """
    start_time = time.time()

    if config and config.verbose:
        print("\n[LLM NODE] Generating response...")

    try:
        transcription = state.get("transcription")
        if not transcription:
            raise LLMGenerationError("No transcription available")

        if config and config.use_mocks:
            from orchestration.mocks import generate_response
        else:
            from llm.agent import generate_response

        emotion, emotional_context, persona, memory, rag_context = _gather_context(state)

        response = generate_response(
            customer_text=transcription, emotion=emotion,
            emotional_context=emotional_context, persona=persona,
            memory=memory, rag_context=rag_context
        )

        state["llm_response"] = response
        state["phase"] = "responding"

        elapsed = time.time() - start_time
        state["node_timings"]["llm"] = elapsed

        if config and config.verbose:
            print(f"[LLM NODE] Response: '{response[:50]}...'")
            print(f"[LLM NODE] Time: {elapsed:.3f}s")

    except Exception as e:
        state["error"] = f"LLM Error: {str(e)}"
        state["llm_response"] = "عذراً، حصل مشكلة. ممكن تعيد الكلام؟"
        if config and config.verbose:
            print(f"[LLM NODE] Error: {str(e)}")

    return state


def llm_node_streaming(
    state: ConversationState,
    config: OrchestrationConfig = None
):
    """
    Generator version: yields (sentence, state) tuples as sentences are generated.
    Final state has complete response in state["llm_response"].
    """
    start_time = time.time()

    if config and config.verbose:
        print("\n[LLM NODE] Generating response (streaming)...")

    try:
        transcription = state.get("transcription")
        if not transcription:
            raise LLMGenerationError("No transcription available")

        if config and config.use_mocks:
            from orchestration.mocks import generate_response
            emotion, emotional_context, persona, memory, rag_context = _gather_context(state)
            response = generate_response(
                customer_text=transcription, emotion=emotion,
                emotional_context=emotional_context, persona=persona,
                memory=memory, rag_context=rag_context
            )
            state["llm_response"] = response
            yield response, state
            return

        from llm.agent import generate_response_streaming

        emotion, emotional_context, persona, memory, rag_context = _gather_context(state)

        full_response = ""
        sentence_count = 0

        for sentence in generate_response_streaming(
            customer_text=transcription, emotion=emotion,
            emotional_context=emotional_context, persona=persona,
            memory=memory, rag_context=rag_context
        ):
            sentence_count += 1
            full_response += (" " + sentence if full_response else sentence)

            if config and config.verbose:
                print(f"[LLM NODE] Sentence {sentence_count}: '{sentence[:40]}...'")

            yield sentence, state

        state["llm_response"] = full_response.strip()
        state["phase"] = "responding"

        elapsed = time.time() - start_time
        state["node_timings"]["llm"] = elapsed

        if config and config.verbose:
            print(f"[LLM NODE] Full: '{full_response[:50]}...' ({sentence_count} sentences)")
            print(f"[LLM NODE] Time: {elapsed:.3f}s")

    except Exception as e:
        state["error"] = f"LLM Error: {str(e)}"
        state["llm_response"] = "عذراً، حصل مشكلة. ممكن تعيد الكلام؟"
        yield state["llm_response"], state
        if config and config.verbose:
            print(f"[LLM NODE] Streaming error: {str(e)}")