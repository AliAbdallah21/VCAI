# orchestration/mocks/__init__.py
"""
Mock functions for VCAI components.

These mocks allow development and testing of the orchestration agent
without waiting for other team members to complete their components.

USAGE:
    # Import individual mocks
    from orchestration.mocks.mock_tts import text_to_speech
    from orchestration.mocks.mock_persona import get_persona, list_personas
    from orchestration.mocks.mock_emotion import detect_emotion, analyze_emotional_context
    from orchestration.mocks.mock_rag import retrieve_context
    from orchestration.mocks.mock_memory import store_message, get_recent_messages
    from orchestration.mocks.mock_llm import generate_response

    # Or import all at once
    from orchestration.mocks import (
        text_to_speech,
        get_persona,
        list_personas,
        detect_emotion,
        analyze_emotional_context,
        retrieve_context,
        store_message,
        get_recent_messages,
        store_checkpoint,
        get_checkpoints,
        get_session_memory,
        generate_response,
        summarize_conversation
    )

SWITCHING TO REAL IMPLEMENTATIONS:
    When teammates complete their components, change imports:
    
    # Before (mock):
    from orchestration.mocks.mock_emotion import detect_emotion
    
    # After (real):
    from emotion.agent import detect_emotion
"""

# TTS (Person B)
from orchestration.mocks.mock_tts import (
    text_to_speech,
    get_available_voices
)

# Persona (Person B)
from orchestration.mocks.mock_persona import (
    get_persona,
    list_personas,
    get_personas_by_difficulty
)

# Emotion (Person C)
from orchestration.mocks.mock_emotion import (
    detect_emotion,
    analyze_emotional_context
)

# RAG (Person D)
from orchestration.mocks.mock_rag import (
    retrieve_context,
    add_document,
    get_document_count
)

# Memory (Person D)
from orchestration.mocks.mock_memory import (
    store_message,
    get_recent_messages,
    get_all_messages,
    store_checkpoint,
    get_checkpoints,
    get_session_memory,
    clear_session,
    clear_all as clear_memory
)

# LLM (Person D)
from orchestration.mocks.mock_llm import (
    generate_response,
    summarize_conversation,
    extract_key_points
)


__all__ = [
    # TTS
    "text_to_speech",
    "get_available_voices",
    
    # Persona
    "get_persona",
    "list_personas",
    "get_personas_by_difficulty",
    
    # Emotion
    "detect_emotion",
    "analyze_emotional_context",
    
    # RAG
    "retrieve_context",
    "add_document",
    "get_document_count",
    
    # Memory
    "store_message",
    "get_recent_messages",
    "get_all_messages",
    "store_checkpoint",
    "get_checkpoints",
    "get_session_memory",
    "clear_session",
    "clear_memory",
    
    # LLM
    "generate_response",
    "summarize_conversation",
    "extract_key_points",
]