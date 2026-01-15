# backend/services/session_service.py
"""
Session service - handles training session management.
"""

from datetime import datetime
from typing import Optional, List
from uuid import UUID

from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from backend.models import Session as TrainingSession, Message, EmotionLog, Persona, User
from backend.schemas import SessionCreate, SessionResponse, MessageCreate, EmotionState


def get_persona(db: Session, persona_id: str) -> Persona:
    """Get a persona by ID."""
    persona = db.query(Persona).filter(Persona.id == persona_id).first()
    if not persona:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Persona '{persona_id}' not found"
        )
    return persona


def get_all_personas(db: Session, active_only: bool = True) -> List[Persona]:
    """Get all personas."""
    query = db.query(Persona)
    if active_only:
        query = query.filter(Persona.is_active == True)
    return query.order_by(Persona.difficulty).all()


def get_personas_by_difficulty(db: Session, difficulty: str) -> List[Persona]:
    """Get personas by difficulty level."""
    return db.query(Persona).filter(
        Persona.difficulty == difficulty,
        Persona.is_active == True
    ).all()


def create_session(
    db: Session,
    user: User,
    session_data: SessionCreate
) -> TrainingSession:
    """Create a new training session."""
    # Verify persona exists
    persona = get_persona(db, session_data.persona_id)
    
    # Create session
    session = TrainingSession(
        user_id=user.id,
        persona_id=session_data.persona_id,
        difficulty=session_data.difficulty,
        status="active"
    )
    
    db.add(session)
    db.commit()
    db.refresh(session)
    
    return session


def get_session(db: Session, session_id: UUID) -> TrainingSession:
    """Get a session by ID."""
    session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    return session


def get_user_sessions(
    db: Session,
    user_id: UUID,
    limit: int = 20,
    offset: int = 0
) -> tuple[List[TrainingSession], int]:
    """Get sessions for a user."""
    query = db.query(TrainingSession).filter(TrainingSession.user_id == user_id)
    total = query.count()
    sessions = query.order_by(TrainingSession.started_at.desc()).offset(offset).limit(limit).all()
    return sessions, total


def add_message(
    db: Session,
    session_id: UUID,
    message_data: MessageCreate
) -> Message:
    """Add a message to a session."""
    message = Message(
        session_id=session_id,
        turn_number=message_data.turn_number,
        speaker=message_data.speaker,
        text=message_data.text,
        detected_emotion=message_data.detected_emotion,
        emotion_confidence=message_data.emotion_confidence,
        response_quality=message_data.response_quality,
        quality_reason=message_data.quality_reason,
        suggestion=message_data.suggestion
    )
    
    db.add(message)
    
    # Update turn count
    session = get_session(db, session_id)
    session.turn_count = message_data.turn_number
    
    db.commit()
    db.refresh(message)
    
    return message


def add_emotion_log(
    db: Session,
    session_id: UUID,
    message_id: UUID,
    emotion_state: EmotionState
) -> EmotionLog:
    """Log an emotion state."""
    log = EmotionLog(
        session_id=session_id,
        message_id=message_id,
        customer_emotion=emotion_state.customer_emotion,
        customer_mood_score=emotion_state.customer_mood_score,
        risk_level=emotion_state.risk_level,
        emotion_trend=emotion_state.emotion_trend,
        tip_shown=emotion_state.tip
    )
    
    db.add(log)
    db.commit()
    db.refresh(log)
    
    return log


def end_session(
    db: Session,
    session_id: UUID,
    evaluation: dict = None
) -> TrainingSession:
    """End a training session."""
    session = get_session(db, session_id)
    
    session.status = "completed"
    session.ended_at = datetime.utcnow()
    
    # Calculate duration
    if session.started_at:
        duration = (session.ended_at - session.started_at).total_seconds()
        session.duration_seconds = int(duration)
    
    # Add evaluation if provided
    if evaluation:
        session.overall_score = evaluation.get("overall_score")
        session.communication_score = evaluation.get("communication_score")
        session.product_knowledge_score = evaluation.get("product_knowledge_score")
        session.objection_handling_score = evaluation.get("objection_handling_score")
        session.rapport_score = evaluation.get("rapport_score")
        session.closing_score = evaluation.get("closing_score")
        session.strengths = evaluation.get("strengths", [])
        session.weaknesses = evaluation.get("weaknesses", [])
        session.recommendations = evaluation.get("recommendations", [])
    
    db.commit()
    db.refresh(session)
    
    return session


def get_session_messages(db: Session, session_id: UUID) -> List[Message]:
    """Get all messages for a session."""
    return db.query(Message).filter(
        Message.session_id == session_id
    ).order_by(Message.turn_number).all()