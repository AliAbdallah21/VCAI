# evaluation/state.py
"""
Evaluation State - TypedDict for LangGraph evaluation pipeline.

This defines the state that flows through the evaluation graph:
    gather_inputs → analyzer_node → synthesizer_node → save_report

Each node reads from and writes to this state.
"""

from typing import TypedDict, Optional, Any
from datetime import datetime

from evaluation.schemas import (
    AnalysisReport,
    FinalReport,
    QuickStats,
    EvaluationMode,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Input Data (gathered before evaluation starts)
# ═══════════════════════════════════════════════════════════════════════════════

class TranscriptMessage(TypedDict):
    """A single message in the transcript."""
    
    turn_number: int
    speaker: str  # "salesperson" or "customer"
    text: str
    timestamp: Optional[str]  # ISO format
    
    # Emotion data (if available)
    detected_emotion: Optional[str]
    emotion_confidence: Optional[float]


class EmotionLogEntry(TypedDict):
    """An entry in the emotion log."""
    
    turn_number: int
    emotion: str
    confidence: float
    mood_score: Optional[int]  # -100 to +100
    risk_level: Optional[str]  # low, medium, high
    trend: Optional[str]  # improving, stable, worsening


class RAGContext(TypedDict):
    """RAG context that was used during the conversation."""
    
    query: str
    documents: list[str]  # Retrieved document chunks
    sources: list[str]  # Document sources/titles


class SessionInfo(TypedDict):
    """Basic session information."""
    
    session_id: str
    user_id: str
    persona_id: str
    persona_name: str
    persona_difficulty: str  # easy, medium, hard
    
    started_at: str  # ISO format
    ended_at: str  # ISO format
    duration_seconds: int


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation State
# ═══════════════════════════════════════════════════════════════════════════════

class EvaluationState(TypedDict):
    """
    Complete state for the evaluation LangGraph pipeline.
    
    Flow:
        1. gather_inputs: Populates session_info, transcript, emotion_log, rag_context
        2. compute_quick_stats: Populates quick_stats (no LLM needed)
        3. analyzer_node: Reads inputs, writes analysis_report
        4. synthesizer_node: Reads analysis_report + mode, writes final_report
        5. save_report: Saves final_report to database
    """
    
    # ─── Request Parameters ───────────────────────────────────────────────────
    
    # What we're evaluating
    session_id: str
    
    # Evaluation mode
    mode: EvaluationMode  # "training" or "testing"

    llm: Optional[Any]
    
    # ─── Input Data (gathered in step 1) ──────────────────────────────────────
    
    # Session metadata
    session_info: Optional[SessionInfo]
    
    # Full conversation transcript
    transcript: list[TranscriptMessage]
    
    # Emotion tracking from conversation
    emotion_log: list[EmotionLogEntry]
    
    # RAG context that was retrieved during conversation
    rag_context: list[RAGContext]
    
    # Checkpoints that were tracked during conversation (if any)
    # This comes from the memory system if checkpoint tracking was enabled
    existing_checkpoints: list[dict]
    
    # ─── Quick Stats (computed without LLM) ───────────────────────────────────
    
    quick_stats: Optional[QuickStats]
    
    # ─── Analysis Output (from analyzer_node) ─────────────────────────────────
    
    analysis_report: Optional[AnalysisReport]
    
    # Raw LLM response (for debugging)
    analysis_raw_response: Optional[str]
    
    # ─── Final Report (from synthesizer_node) ─────────────────────────────────
    
    final_report: Optional[FinalReport]
    
    # Raw LLM response (for debugging)
    synthesis_raw_response: Optional[str]
    
    # ─── Pipeline Metadata ────────────────────────────────────────────────────
    
    # Timing
    started_at: Optional[str]  # ISO format
    completed_at: Optional[str]  # ISO format
    
    # Node timing (for performance tracking)
    node_timings: dict[str, float]  # node_name → seconds
    
    # Errors (if any)
    errors: list[str]
    
    # Current status
    status: str  # "pending", "gathering", "analyzing", "synthesizing", "saving", "completed", "failed"
    
    # Progress (0-100)
    progress: int


# ═══════════════════════════════════════════════════════════════════════════════
# State Initialization
# ═══════════════════════════════════════════════════════════════════════════════

def create_initial_state(
    session_id: str,
    mode: str = "training"
) -> EvaluationState:
    """
    Create initial evaluation state.
    
    Args:
        session_id: Session to evaluate
        mode: "training" or "testing"
    
    Returns:
        Initial EvaluationState with empty fields
    """
    return EvaluationState(
        # Request
        session_id=session_id,
        mode=EvaluationMode(mode),
        llm=None,
        
        # Inputs (to be filled by gather_inputs)
        session_info=None,
        transcript=[],
        emotion_log=[],
        rag_context=[],
        existing_checkpoints=[],
        
        # Quick stats (to be filled by compute_quick_stats)
        quick_stats=None,
        
        # Analysis (to be filled by analyzer_node)
        analysis_report=None,
        analysis_raw_response=None,
        
        # Final report (to be filled by synthesizer_node)
        final_report=None,
        synthesis_raw_response=None,
        
        # Metadata
        started_at=datetime.utcnow().isoformat(),
        completed_at=None,
        node_timings={},
        errors=[],
        status="pending",
        progress=0,
    )


def update_state_status(
    state: EvaluationState,
    status: str,
    progress: int
) -> EvaluationState:
    """
    Update state status and progress.
    Returns a new state dict (immutable update).
    """
    return {
        **state,
        "status": status,
        "progress": progress,
    }


def add_error(
    state: EvaluationState,
    error: str
) -> EvaluationState:
    """
    Add an error to the state.
    Returns a new state dict (immutable update).
    """
    return {
        **state,
        "errors": state["errors"] + [error],
    }


def record_node_timing(
    state: EvaluationState,
    node_name: str,
    duration_seconds: float
) -> EvaluationState:
    """
    Record timing for a node.
    Returns a new state dict (immutable update).
    """
    return {
        **state,
        "node_timings": {
            **state["node_timings"],
            node_name: duration_seconds,
        },
    }


def mark_completed(state: EvaluationState) -> EvaluationState:
    """
    Mark evaluation as completed.
    Returns a new state dict (immutable update).
    """
    return {
        **state,
        "status": "completed",
        "progress": 100,
        "completed_at": datetime.utcnow().isoformat(),
    }


def mark_failed(state: EvaluationState, error: str) -> EvaluationState:
    """
    Mark evaluation as failed.
    Returns a new state dict (immutable update).
    """
    return {
        **state,
        "status": "failed",
        "errors": state["errors"] + [error],
        "completed_at": datetime.utcnow().isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# State Validation
# ═══════════════════════════════════════════════════════════════════════════════

def validate_inputs_ready(state: EvaluationState) -> tuple[bool, list[str]]:
    """
    Check if all inputs are ready for analysis.
    
    Returns:
        (is_valid, list of missing items)
    """
    missing = []
    
    if not state.get("session_info"):
        missing.append("session_info")
    
    if not state.get("transcript") or len(state["transcript"]) == 0:
        missing.append("transcript (empty)")
    
    # emotion_log and rag_context can be empty (optional)
    
    return len(missing) == 0, missing


def validate_analysis_ready(state: EvaluationState) -> tuple[bool, list[str]]:
    """
    Check if analysis is ready for synthesis.
    
    Returns:
        (is_valid, list of missing items)
    """
    missing = []
    
    if not state.get("analysis_report"):
        missing.append("analysis_report")
    
    if not state.get("quick_stats"):
        missing.append("quick_stats")
    
    return len(missing) == 0, missing


# ═══════════════════════════════════════════════════════════════════════════════
# Progress Constants
# ═══════════════════════════════════════════════════════════════════════════════

class EvaluationProgress:
    """Progress milestones for UI updates."""
    
    STARTED = 0
    GATHERING_INPUTS = 10
    INPUTS_READY = 20
    COMPUTING_QUICK_STATS = 25
    QUICK_STATS_READY = 30
    ANALYZING = 40
    ANALYSIS_COMPLETE = 70
    SYNTHESIZING = 80
    SYNTHESIS_COMPLETE = 95
    SAVING = 98
    COMPLETED = 100