# backend/services/session_service.py
"""
Session service - handles training session management.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional, List
from uuid import UUID

from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from backend.models import Session as TrainingSession, Message, EmotionLog, Persona, User
from backend.models.evaluation import EvaluationReport
from backend.schemas import SessionCreate, SessionResponse, MessageCreate, EmotionState, session


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


def get_personas_by_gender(db: Session, gender: str) -> List[Persona]:
    """Get personas by gender (male | female)."""
    return db.query(Persona).filter(
        Persona.gender == gender,
        Persona.is_active == True
    ).order_by(Persona.difficulty).all()


def create_session(
    db: Session,
    user: User,
    session_data: SessionCreate
) -> TrainingSession:
    """
    Create a new training session.

    Tenant-scoped (Phase 4): before creating, the company's billing status,
    monthly session quota, and free-plan persona gating are enforced. The usage
    counter is incremented inside the same transaction as the session insert so
    the increment happens exactly once per created session (and rolls back with
    it on failure). Users without a company (legacy single-tenant accounts) skip
    quota/gating and behave as before.
    """
    from backend.services.usage_service import (
        assert_can_create_session,
        record_session_usage,
    )

    # Verify persona exists
    persona = get_persona(db, session_data.persona_id)

    company_id = user.company_id

    # Quota + billing + persona gating (only for tenant-scoped users).
    if company_id is not None:
        assert_can_create_session(db, company_id)
        assert_persona_allowed(db, company_id, session_data.persona_id)

    # Resolve the buyer scenario. resolve_scenario() never raises — bad/None
    # input falls back to a fully random scenario — so the session always
    # gets a coherent scenario.
    from shared.scenarios import resolve_scenario
    scenario = resolve_scenario(session_data.scenario)

    # Create session
    session = TrainingSession(
        user_id=user.id,
        persona_id=session_data.persona_id,
        difficulty=session_data.difficulty,
        status="active",
        scenario=scenario,
        training_focus=session_data.training_focus or None,
    )

    db.add(session)

    # Record usage in the SAME transaction so it commits atomically with the
    # session insert (exactly-once per created session).
    if company_id is not None:
        record_session_usage(db, company_id)

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


_STALE_THRESHOLD = timedelta(hours=2)


def _auto_end_stale(db: Session, session: TrainingSession) -> None:
    """End a session that has been stuck as 'active' with no recent activity."""
    now = datetime.now(timezone.utc)
    # Use timestamp of last message as the real end time, fall back to started_at
    last_msg = (
        db.query(Message)
        .filter(Message.session_id == session.id)
        .order_by(Message.created_at.desc())
        .first()
    )
    end_time = last_msg.created_at if (last_msg and last_msg.created_at) else session.started_at
    if end_time and end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=timezone.utc)
    session.status = "ended"
    session.ended_at = end_time
    session.duration_seconds = max(0, int((end_time - session.started_at.replace(tzinfo=timezone.utc) if session.started_at.tzinfo is None else end_time - session.started_at).total_seconds()))
    db.commit()


def get_user_sessions(
    db: Session,
    user_id: UUID,
    limit: int = 20,
    offset: int = 0
) -> tuple[List[TrainingSession], int]:
    """Get sessions for a user. Auto-ends sessions stuck as active for over 2 hours."""
    now = datetime.now(timezone.utc)
    stale_cutoff = now - _STALE_THRESHOLD
    stale = (
        db.query(TrainingSession)
        .filter(
            TrainingSession.user_id == user_id,
            TrainingSession.status == "active",
            TrainingSession.started_at < stale_cutoff,
        )
        .all()
    )
    for s in stale:
        _auto_end_stale(db, s)

    query = db.query(TrainingSession).filter(TrainingSession.user_id == user_id)
    total = query.count()
    sessions = query.order_by(TrainingSession.started_at.desc()).offset(offset).limit(limit).all()

    # Sync overall_score from EvaluationReport for sessions that are missing it
    needs_commit = False
    for s in sessions:
        if s.overall_score is None:
            report = (
                db.query(EvaluationReport)
                .filter(EvaluationReport.session_id == s.id, EvaluationReport.status == "completed")
                .first()
            )
            if report and report.overall_score is not None:
                s.overall_score = report.overall_score
                needs_commit = True
    if needs_commit:
        db.commit()

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
    session.ended_at = datetime.now(timezone.utc)

    # Duration = actual conversation span (first msg → last msg), not idle wait time
    first_msg = (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.created_at.asc())
        .first()
    )
    last_msg = (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.created_at.desc())
        .first()
    )
    if first_msg and last_msg and first_msg.created_at and last_msg.created_at:
        t0 = first_msg.created_at if first_msg.created_at.tzinfo else first_msg.created_at.replace(tzinfo=timezone.utc)
        t1 = last_msg.created_at if last_msg.created_at.tzinfo else last_msg.created_at.replace(tzinfo=timezone.utc)
        session.duration_seconds = max(0, int((t1 - t0).total_seconds()))
    elif session.started_at:
        session.duration_seconds = max(0, int((session.ended_at - session.started_at).total_seconds()))
    
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
    ).order_by(Message.created_at).all()


# ─────────────────────────────────────────────────────────────────────────────
# Tenant plan resolution + persona gating (Phase 4)
# ─────────────────────────────────────────────────────────────────────────────

def _company_plan_name(db: Session, company_id) -> Optional[str]:
    """Return the plan name for a company via its subscription, or None."""
    from backend.models import Subscription

    if company_id is None:
        return None
    sub = db.query(Subscription).filter(Subscription.company_id == company_id).first()
    return sub.plan_name if sub else None


def assert_persona_allowed(db: Session, company_id, persona_id: str) -> None:
    """
    Free plans may only start sessions with the two Easy personas
    (backend.plans.FREE_PERSONA_IDS). Paid plans allow all active personas.

    Raises 403 "Upgrade to access this persona" when a free-plan company
    requests a gated persona. Companies with no resolvable plan are treated as
    free (the most restrictive default).
    """
    from backend.plans import FREE_PERSONA_IDS

    plan_name = _company_plan_name(db, company_id)
    if plan_name in (None, "free"):
        if persona_id not in FREE_PERSONA_IDS:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Upgrade to access this persona",
            )


def get_all_personas_for_company(
    db: Session, company_id, active_only: bool = True
) -> List[Persona]:
    """
    Return all active personas, each annotated with a transient `locked` bool.
    For free plans, personas outside FREE_PERSONA_IDS are marked locked so the
    UI can grey them out (better upsell) rather than hide them. Paid plans get
    `locked = False` on every persona.

    The `locked` attribute is set on the ORM instances in-memory (not a DB
    column); PersonaResponse reads it via from_attributes.
    """
    from backend.plans import FREE_PERSONA_IDS

    personas = get_all_personas(db, active_only=active_only)
    plan_name = _company_plan_name(db, company_id)
    is_free = plan_name in (None, "free")
    for p in personas:
        p.locked = is_free and p.id not in FREE_PERSONA_IDS
    return personas