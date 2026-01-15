# orchestration/__init__.py
"""
VCAI Orchestration Agent.

Main entry point for conversation flow management.
Coordinates STT → Emotion → RAG → LLM → TTS pipeline.

Usage:
    from orchestration import OrchestrationAgent
    
    # Create agent
    agent = OrchestrationAgent(use_mocks=True)
    
    # Start session
    agent.start_session(
        session_id="session_001",
        user_id="user_001", 
        persona_id="difficult_customer"
    )
    
    # Process audio turn
    result = agent.process_turn(audio_input)
    
    # Get response
    text = agent.get_response_text()
    audio = agent.get_response_audio()
    
    # End session
    agent.end_session()
"""

from orchestration.agent import (
    OrchestrationAgent,
    create_agent,
    quick_test
)

from orchestration.state import (
    ConversationState,
    create_initial_state,
    reset_turn_state,
    get_state_summary
)

from orchestration.config import (
    OrchestrationConfig,
    get_config,
    DEFAULT_CONFIG
)

from orchestration.graphs import (
    create_conversation_graph,
    create_simple_graph
)

__all__ = [
    # Main agent
    "OrchestrationAgent",
    "create_agent",
    "quick_test",
    
    # State
    "ConversationState",
    "create_initial_state",
    "reset_turn_state",
    "get_state_summary",
    
    # Config
    "OrchestrationConfig",
    "get_config",
    "DEFAULT_CONFIG",
    
    # Graphs
    "create_conversation_graph",
    "create_simple_graph"
]