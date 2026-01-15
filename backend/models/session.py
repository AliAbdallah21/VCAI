# backend/models/session.py
"""
Session and Message SQLAlchemy models.
"""

from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
import uuid

from backend.database import Base


class Session(Base):
    """Training session model."""
    
    __tablename__ = "sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    persona_id = Column(String(100), ForeignKey("personas.id"), nullable=False)
    
    # Session info
    status = Column(String(50), default="active")  # active, completed, abandoned
    difficulty = Column(String(50), nullable=False)
    
    # Timing
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    ended_at = Column(DateTime(timezone=True))
    duration_seconds = Column(Integer)
    
    # Scores (filled after session ends)
    overall_score = Column(Integer)
    communication_score = Column(Integer)
    product_knowledge_score = Column(Integer)
    objection_handling_score = Column(Integer)
    rapport_score = Column(Integer)
    closing_score = Column(Integer)
    
    # Feedback
    strengths = Column(JSONB, default=[])
    weaknesses = Column(JSONB, default=[])
    recommendations = Column(JSONB, default=[])
    
    # Metadata
    turn_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    persona = relationship("Persona", back_populates="sessions")
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")
    emotion_logs = relationship("EmotionLog", back_populates="session", cascade="all, delete-orphan")
    checkpoints = relationship("Checkpoint", back_populates="session", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Session {self.id} ({self.status})>"


class Message(Base):
    """Conversation message model."""
    
    __tablename__ = "messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    
    # Message info
    turn_number = Column(Integer, nullable=False)
    speaker = Column(String(50), nullable=False)  # salesperson, customer
    text = Column(Text, nullable=False)
    
    # Audio
    audio_path = Column(String(500))
    audio_duration_seconds = Column(Float)
    
    # Emotion analysis
    detected_emotion = Column(String(50))
    emotion_confidence = Column(Float)
    emotion_scores = Column(JSONB)
    
    # Real-time evaluation
    response_quality = Column(String(50))  # good, neutral, bad
    quality_reason = Column(Text)
    suggestion = Column(Text)
    
    # Timing
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    processing_time_ms = Column(Integer)
    
    # Relationships
    session = relationship("Session", back_populates="messages")
    
    def __repr__(self):
        return f"<Message {self.turn_number} ({self.speaker})>"


class EmotionLog(Base):
    """Emotion tracking log model."""
    
    __tablename__ = "emotion_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    message_id = Column(UUID(as_uuid=True), ForeignKey("messages.id", ondelete="CASCADE"))
    
    # Customer emotion state
    customer_emotion = Column(String(50), nullable=False)
    customer_mood_score = Column(Integer)  # -100 to +100
    risk_level = Column(String(50))  # low, medium, high
    
    # Trend
    emotion_trend = Column(String(50))  # improving, stable, worsening
    
    # Tip
    tip_shown = Column(Text)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    session = relationship("Session", back_populates="emotion_logs")
    
    def __repr__(self):
        return f"<EmotionLog {self.customer_emotion} ({self.risk_level})>"


class Checkpoint(Base):
    """Memory checkpoint model."""
    
    __tablename__ = "checkpoints"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    
    turn_start = Column(Integer, nullable=False)
    turn_end = Column(Integer, nullable=False)
    summary = Column(Text, nullable=False)
    key_points = Column(JSONB, default=[])
    customer_preferences = Column(JSONB, default={})
    objections_raised = Column(JSONB, default=[])
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    session = relationship("Session", back_populates="checkpoints")
    
    def __repr__(self):
        return f"<Checkpoint turns {self.turn_start}-{self.turn_end}>"


class UserStats(Base):
    """User statistics model."""
    
    __tablename__ = "user_stats"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)
    
    # Counts
    total_sessions = Column(Integer, default=0)
    completed_sessions = Column(Integer, default=0)
    total_training_minutes = Column(Integer, default=0)
    
    # Average scores
    avg_overall_score = Column(Float)
    avg_communication_score = Column(Float)
    avg_product_knowledge_score = Column(Float)
    avg_objection_handling_score = Column(Float)
    avg_rapport_score = Column(Float)
    avg_closing_score = Column(Float)
    
    # Best/Worst
    best_session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"))
    best_score = Column(Integer)
    
    # Streaks
    current_streak = Column(Integer, default=0)
    longest_streak = Column(Integer, default=0)
    last_session_date = Column(DateTime)
    
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="stats")
    
    def __repr__(self):
        return f"<UserStats for user {self.user_id}>"