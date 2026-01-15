# backend/schemas/__init__.py
"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime
from uuid import UUID


# ══════════════════════════════════════════════════════════════════════════════
# USER SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class UserCreate(BaseModel):
    """Schema for user registration."""
    email: EmailStr
    password: str = Field(..., min_length=6)
    full_name: str = Field(..., min_length=2)
    company: Optional[str] = None
    experience_level: str = "beginner"


class UserLogin(BaseModel):
    """Schema for user login."""
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """Schema for user response."""
    id: UUID
    email: str
    full_name: str
    company: Optional[str]
    role: str
    experience_level: str
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    """Schema for user update."""
    full_name: Optional[str] = None
    company: Optional[str] = None
    experience_level: Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
# AUTH SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class Token(BaseModel):
    """Schema for JWT token response."""
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class TokenData(BaseModel):
    """Schema for decoded token data."""
    user_id: Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
# PERSONA SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class PersonaResponse(BaseModel):
    """Schema for persona response."""
    id: str
    name_ar: str
    name_en: str
    description_ar: Optional[str]
    description_en: Optional[str]
    difficulty: str
    patience_level: int
    emotion_sensitivity: int
    traits: List[str]
    voice_id: Optional[str]
    avatar_url: Optional[str]
    
    class Config:
        from_attributes = True


class PersonaListResponse(BaseModel):
    """Schema for list of personas."""
    personas: List[PersonaResponse]
    total: int


# ══════════════════════════════════════════════════════════════════════════════
# SESSION SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class SessionCreate(BaseModel):
    """Schema for starting a new session."""
    persona_id: str
    difficulty: str = "medium"


class SessionResponse(BaseModel):
    """Schema for session response."""
    id: UUID
    user_id: UUID
    persona_id: str
    status: str
    difficulty: str
    started_at: datetime
    ended_at: Optional[datetime]
    duration_seconds: Optional[int]
    turn_count: int
    overall_score: Optional[int]
    communication_score: Optional[int]
    product_knowledge_score: Optional[int]
    objection_handling_score: Optional[int]
    rapport_score: Optional[int]
    closing_score: Optional[int]
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    
    class Config:
        from_attributes = True


class SessionSummary(BaseModel):
    """Schema for session list item."""
    id: UUID
    persona_id: str
    persona_name: Optional[str] = None
    status: str
    difficulty: str
    started_at: datetime
    duration_seconds: Optional[int]
    overall_score: Optional[int]


class SessionListResponse(BaseModel):
    """Schema for list of sessions."""
    sessions: List[SessionSummary]
    total: int


# ══════════════════════════════════════════════════════════════════════════════
# MESSAGE SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class MessageCreate(BaseModel):
    """Schema for creating a message (internal use)."""
    turn_number: int
    speaker: str  # salesperson, customer
    text: str
    detected_emotion: Optional[str] = None
    emotion_confidence: Optional[float] = None
    response_quality: Optional[str] = None
    quality_reason: Optional[str] = None
    suggestion: Optional[str] = None


class MessageResponse(BaseModel):
    """Schema for message response."""
    id: UUID
    turn_number: int
    speaker: str
    text: str
    detected_emotion: Optional[str]
    emotion_confidence: Optional[float]
    response_quality: Optional[str]
    quality_reason: Optional[str]
    suggestion: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True


# ══════════════════════════════════════════════════════════════════════════════
# EMOTION SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class EmotionState(BaseModel):
    """Schema for current emotion state (sent to frontend)."""
    customer_emotion: str
    customer_mood_score: int  # -100 to +100
    risk_level: str  # low, medium, high
    emotion_trend: str  # improving, stable, worsening
    tip: Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class TurnEvaluation(BaseModel):
    """Schema for real-time turn evaluation."""
    quality: str  # good, neutral, bad
    reason: str
    suggestion: Optional[str] = None


class SessionEvaluation(BaseModel):
    """Schema for final session evaluation."""
    overall_score: int
    communication_score: int
    product_knowledge_score: int
    objection_handling_score: int
    rapport_score: int
    closing_score: int
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]


# ══════════════════════════════════════════════════════════════════════════════
# USER STATS SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class UserStatsResponse(BaseModel):
    """Schema for user statistics."""
    total_sessions: int
    completed_sessions: int
    total_training_minutes: int
    avg_overall_score: Optional[float]
    avg_communication_score: Optional[float]
    avg_product_knowledge_score: Optional[float]
    avg_objection_handling_score: Optional[float]
    avg_rapport_score: Optional[float]
    avg_closing_score: Optional[float]
    best_score: Optional[int]
    current_streak: int
    longest_streak: int
    
    class Config:
        from_attributes = True


# ══════════════════════════════════════════════════════════════════════════════
# WEBSOCKET SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class WSMessage(BaseModel):
    """Schema for WebSocket messages."""
    type: str  # audio, text, control, emotion, evaluation
    data: dict


class WSAudioMessage(BaseModel):
    """Schema for audio data in WebSocket."""
    audio_base64: str  # Base64 encoded audio
    sample_rate: int = 16000


class WSResponseMessage(BaseModel):
    """Schema for response from server."""
    type: str  # transcription, response, audio, emotion, tip, error
    data: dict