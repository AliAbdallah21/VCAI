# orchestration/state.py
"""
Conversation state definition for LangGraph orchestration.
This state flows through all nodes in the graph.
"""

from typing import TypedDict, Optional, Annotated
from datetime import datetime
import numpy as np
from operator import add

from shared.types import (
    Persona,
    Scenario,
    EmotionResult,
    EmotionalContext,
    SessionMemory,
    RAGContext,
    Message,
    TurnPhase
)


class ConversationState(TypedDict):
    """
    Main state object that flows through the LangGraph pipeline.
    Each node reads from and writes to this state.
    """
    
    # ══════════════════════════════════════════════════════════════════════════
    # SESSION INFO (Set at start, rarely changes)
    # ══════════════════════════════════════════════════════════════════════════
    session_id: str                         # Unique session identifier
    user_id: str                            # User/trainee ID
    persona: Optional[Persona]              # VC persona configuration
    scenario: Optional[Scenario]            # Training scenario
    
    # ══════════════════════════════════════════════════════════════════════════
    # TURN TRACKING
    # ══════════════════════════════════════════════════════════════════════════
    turn_count: int                         # Current turn number
    phase: TurnPhase                        # "listening", "processing", "responding", "idle"
    
    # ══════════════════════════════════════════════════════════════════════════
    # AUDIO (Input/Output for current turn)
    # ══════════════════════════════════════════════════════════════════════════
    audio_input: Optional[np.ndarray]       # Salesperson's voice (16kHz, float32)
    audio_output: Optional[np.ndarray]      # VC's voice (22kHz, float32)
    
    # ══════════════════════════════════════════════════════════════════════════
    # TEXT (Current turn)
    # ══════════════════════════════════════════════════════════════════════════
    transcription: Optional[str]            # STT output - what salesperson said
    llm_response: Optional[str]             # LLM output - what VC will say
    
    # ══════════════════════════════════════════════════════════════════════════
    # CONTEXT (Analysis for current turn)
    # ══════════════════════════════════════════════════════════════════════════
    emotion: Optional[EmotionResult]        # Detected emotion
    emotional_context: Optional[EmotionalContext]  # Emotional analysis
    memory: Optional[SessionMemory]         # Conversation memory
    rag_context: Optional[RAGContext]       # Retrieved documents
    
    # ══════════════════════════════════════════════════════════════════════════
    # HISTORY (Accumulated across turns)
    # ══════════════════════════════════════════════════════════════════════════
    history: list[Message]                  # All messages in session
    
    # ══════════════════════════════════════════════════════════════════════════
    # CONTROL FLAGS
    # ══════════════════════════════════════════════════════════════════════════
    is_active: bool                         # Is session active?
    should_end: bool                        # Should session end?
    error: Optional[str]                    # Error message if any
    
    # ══════════════════════════════════════════════════════════════════════════
    # TIMING (For performance monitoring)
    # ══════════════════════════════════════════════════════════════════════════
    turn_start_time: Optional[datetime]     # When current turn started
    node_timings: dict                      # Time taken by each node


def create_initial_state(
    session_id: str,
    user_id: str,
    persona: Persona = None,
    scenario: Scenario = None
) -> ConversationState:
    """
    Create a fresh conversation state for a new session.
    
    Args:
        session_id: Unique session identifier
        user_id: User/trainee ID
        persona: Optional persona (can be set later)
        scenario: Optional scenario (can be set later)
    
    Returns:
        ConversationState: Initialized state
    """
    return ConversationState(
        # Session info
        session_id=session_id,
        user_id=user_id,
        persona=persona,
        scenario=scenario,
        
        # Turn tracking
        turn_count=0,
        phase="idle",
        
        # Audio
        audio_input=None,
        audio_output=None,
        
        # Text
        transcription=None,
        llm_response=None,
        
        # Context
        emotion=None,
        emotional_context=None,
        memory=None,
        rag_context=None,
        
        # History
        history=[],
        
        # Control
        is_active=True,
        should_end=False,
        error=None,
        
        # Timing
        turn_start_time=None,
        node_timings={}
    )


def reset_turn_state(state: ConversationState) -> ConversationState:
    """
    Reset state fields that are specific to a single turn.
    Called at the start of each new turn.
    
    Args:
        state: Current state
    
    Returns:
        ConversationState: State with turn-specific fields reset
    """
    state["audio_input"] = None
    state["audio_output"] = None
    state["transcription"] = None
    state["llm_response"] = None
    state["emotion"] = None
    state["emotional_context"] = None
    state["rag_context"] = None
    state["error"] = None
    state["turn_start_time"] = datetime.now()
    state["node_timings"] = {}
    state["phase"] = "listening"
    
    return state


def get_state_summary(state: ConversationState) -> dict:
    """
    Get a summary of the current state for logging/debugging.
    
    Args:
        state: Current state
    
    Returns:
        dict: Summary of key state values
    """
    return {
        "session_id": state["session_id"],
        "turn_count": state["turn_count"],
        "phase": state["phase"],
        "is_active": state["is_active"],
        "has_transcription": state["transcription"] is not None,
        "has_response": state["llm_response"] is not None,
        "emotion": state["emotion"]["primary_emotion"] if state["emotion"] else None,
        "history_length": len(state["history"]),
        "error": state["error"]
    }