# shared/types.py
"""
Shared type definitions for VCAI project.
All team members must use these types for consistency.
"""

from typing import TypedDict, Optional, Literal
from datetime import datetime
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# AUDIO TYPES
# ══════════════════════════════════════════════════════════════════════════════

class AudioConfig(TypedDict):
    """Standard audio configuration"""
    sample_rate: int          # 16000 Hz for STT, 22050 Hz for TTS
    channels: int             # 1 (mono)
    dtype: str                # "float32"


class AudioData(TypedDict):
    """Audio data with metadata"""
    data: np.ndarray          # Audio samples
    sample_rate: int          # Sample rate
    duration: float           # Duration in seconds


# ══════════════════════════════════════════════════════════════════════════════
# USER TYPES
# ══════════════════════════════════════════════════════════════════════════════

UserRole = Literal["trainee", "admin", "supervisor"]

class User(TypedDict):
    """User data structure"""
    id: str
    name: str
    email: str
    role: UserRole
    created_at: datetime
    updated_at: Optional[datetime]


class UserCreate(TypedDict):
    """Data required to create a user"""
    name: str
    email: str
    password: str
    role: UserRole


class UserLogin(TypedDict):
    """Login credentials"""
    email: str
    password: str


class AuthToken(TypedDict):
    """Authentication token response"""
    access_token: str
    token_type: str
    expires_in: int
    user: User


# ══════════════════════════════════════════════════════════════════════════════
# PERSONA TYPES
# ══════════════════════════════════════════════════════════════════════════════

PersonaDifficulty = Literal["easy", "medium", "hard"]
PersonaEmotion = Literal["neutral", "friendly", "frustrated", "hesitant", "angry", "interested"]

class Persona(TypedDict):
    """Virtual Customer persona definition"""
    id: str
    name: str                           # "عميل صعب"
    name_en: str                        # "Difficult Customer"
    description: str                    # "عميل متشكك وبيفاصل كتير"
    personality_prompt: str             # System prompt for LLM
    voice_id: str                       # Voice ID for TTS
    default_emotion: PersonaEmotion     # Default emotional state
    difficulty: PersonaDifficulty       # Difficulty level
    traits: list[str]                   # ["متشكك", "بيفاصل", "صبور"]
    avatar_url: Optional[str]           # Avatar image URL


class PersonaSummary(TypedDict):
    """Simplified persona for listing"""
    id: str
    name: str
    name_en: str
    difficulty: PersonaDifficulty
    avatar_url: Optional[str]


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO TYPES
# ══════════════════════════════════════════════════════════════════════════════

class ScenarioObjective(TypedDict):
    """Single objective in a scenario"""
    id: str
    description: str                    # "Handle price objection"
    description_ar: str                 # "التعامل مع اعتراض السعر"
    points: int                         # Points for completing this objective
    required: bool                      # Is this objective required to pass?


class Scenario(TypedDict):
    """Training scenario definition"""
    id: str
    name: str                           # "اعتراض على السعر"
    name_en: str                        # "Price Objection"
    description: str                    # Scenario description
    briefing: str                       # Pre-session briefing for trainee
    context: str                        # Context for LLM
    persona_id: str                     # Which persona to use
    objectives: list[ScenarioObjective] # Learning objectives
    max_duration: int                   # Max duration in seconds
    difficulty: PersonaDifficulty


class ScenarioSummary(TypedDict):
    """Simplified scenario for listing"""
    id: str
    name: str
    name_en: str
    persona_id: str
    difficulty: PersonaDifficulty


# ══════════════════════════════════════════════════════════════════════════════
# EMOTION TYPES
# ══════════════════════════════════════════════════════════════════════════════

EmotionLabel = Literal["happy", "sad", "angry", "fearful", "surprised", "disgusted", "neutral"]
EmotionIntensity = Literal["low", "medium", "high"]
EmotionTrend = Literal["improving", "worsening", "stable"]
RiskLevel = Literal["low", "medium", "high"]

class EmotionScores(TypedDict):
    """Detailed emotion scores"""
    happy: float                        # 0.0 to 1.0
    sad: float
    angry: float
    fearful: float
    surprised: float
    disgusted: float
    neutral: float


class EmotionResult(TypedDict):
    """Output from emotion detection"""
    primary_emotion: EmotionLabel       # Main detected emotion
    confidence: float                   # 0.0 to 1.0
    voice_emotion: EmotionLabel         # Emotion from audio analysis
    text_emotion: EmotionLabel          # Emotion from text analysis
    intensity: EmotionIntensity         # Emotion intensity
    scores: EmotionScores               # Detailed scores


class EmotionalContext(TypedDict):
    """Emotional context with history analysis"""
    current: EmotionResult              # Current emotion
    trend: EmotionTrend                 # How emotions are changing
    recommendation: str                 # e.g., "be_gentle", "show_empathy"
    risk_level: RiskLevel               # Risk of losing customer


# ══════════════════════════════════════════════════════════════════════════════
# CONVERSATION TYPES
# ══════════════════════════════════════════════════════════════════════════════

Speaker = Literal["salesperson", "vc"]

class Message(TypedDict):
    """Single message in conversation"""
    id: str
    turn: int                           # Turn number (1, 2, 3, ...)
    speaker: Speaker                    # Who spoke
    text: str                           # What was said
    emotion: Optional[EmotionResult]    # Emotion (for salesperson only)
    audio_path: Optional[str]           # Path to audio file (optional)
    timestamp: datetime


class ConversationHistory(TypedDict):
    """List of messages with metadata"""
    session_id: str
    messages: list[Message]
    turn_count: int


# ══════════════════════════════════════════════════════════════════════════════
# MEMORY TYPES
# ══════════════════════════════════════════════════════════════════════════════

class MemoryCheckpoint(TypedDict):
    """Summary checkpoint of conversation"""
    id: str
    session_id: str
    turn_range: tuple[int, int]         # (start_turn, end_turn)
    summary: str                        # Summary of this section
    key_points: list[str]               # Important points mentioned
    customer_preferences: dict          # What customer wants
    objections_raised: list[str]        # Customer concerns
    created_at: datetime


class SessionMemory(TypedDict):
    """Full session memory"""
    session_id: str
    checkpoints: list[MemoryCheckpoint] # Historical summaries
    recent_messages: list[Message]      # Recent full messages
    total_turns: int


# ══════════════════════════════════════════════════════════════════════════════
# RAG TYPES
# ══════════════════════════════════════════════════════════════════════════════

class RAGDocument(TypedDict):
    """Retrieved document from RAG"""
    content: str                        # Document content
    source: str                         # Source file name
    score: float                        # Relevance score (0.0 to 1.0)
    metadata: dict                      # Additional metadata


class RAGContext(TypedDict):
    """RAG retrieval result"""
    query: str                          # Original query
    documents: list[RAGDocument]        # Retrieved documents
    total_found: int


# ══════════════════════════════════════════════════════════════════════════════
# SESSION TYPES
# ══════════════════════════════════════════════════════════════════════════════

SessionStatus = Literal["pending", "active", "completed", "abandoned"]

class Session(TypedDict):
    """Training session"""
    id: str
    user_id: str
    persona_id: str
    scenario_id: str
    status: SessionStatus
    started_at: Optional[datetime]
    ended_at: Optional[datetime]
    created_at: datetime


class SessionCreate(TypedDict):
    """Data to create a session"""
    user_id: str
    persona_id: str
    scenario_id: str


class SessionDetail(TypedDict):
    """Full session with related data"""
    session: Session
    persona: Persona
    scenario: Scenario
    messages: list[Message]
    evaluation: Optional['Evaluation']


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION TYPES
# ══════════════════════════════════════════════════════════════════════════════

class ObjectiveScore(TypedDict):
    """Score for a single objective"""
    objective_id: str
    achieved: bool
    score: int
    feedback: str


class Evaluation(TypedDict):
    """Session evaluation result"""
    id: str
    session_id: str
    total_score: float                  # 0.0 to 100.0
    grade: str                          # "A", "B", "C", "D", "F"
    objective_scores: list[ObjectiveScore]
    strengths: list[str]                # What trainee did well
    improvements: list[str]             # What needs improvement
    feedback: str                       # Overall feedback
    created_at: datetime


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATION STATE
# ══════════════════════════════════════════════════════════════════════════════

TurnPhase = Literal["listening", "processing", "responding", "idle"]

class ConversationState(TypedDict):
    """State object for LangGraph orchestration"""
    # Session info
    session_id: str
    user_id: str
    persona: Persona
    scenario: Scenario
    
    # Current turn
    turn_count: int
    phase: TurnPhase
    
    # Audio
    audio_input: Optional[np.ndarray]   # Salesperson's voice
    audio_output: Optional[np.ndarray]  # VC's voice
    
    # Text
    transcription: Optional[str]        # STT output
    llm_response: Optional[str]         # LLM output
    
    # Context
    emotion: Optional[EmotionResult]
    emotional_context: Optional[EmotionalContext]
    memory: Optional[SessionMemory]
    rag_context: Optional[RAGContext]
    
    # History
    history: list[Message]
    
    # Control
    is_active: bool
    error: Optional[str]


# ══════════════════════════════════════════════════════════════════════════════
# WEBSOCKET MESSAGE TYPES
# ══════════════════════════════════════════════════════════════════════════════

WebSocketMessageType = Literal[
    "audio_chunk",          # Client sends audio
    "transcription",        # Server sends STT result
    "emotion",              # Server sends emotion
    "vc_response_text",     # Server sends VC text
    "vc_response_audio",    # Server sends VC audio
    "session_start",        # Session started
    "session_end",          # Session ended
    "error",                # Error occurred
    "ping",                 # Keep alive
    "pong"                  # Keep alive response
]

class WebSocketMessage(TypedDict):
    """WebSocket message structure"""
    type: WebSocketMessageType
    payload: dict
    timestamp: datetime