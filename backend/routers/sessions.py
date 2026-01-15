# backend/routers/sessions.py
"""
Training sessions API endpoints.
"""

from typing import Optional
from uuid import UUID
from fastapi import APIRouter, Depends, Query, HTTPException, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.schemas import (
    SessionCreate, 
    SessionResponse, 
    SessionListResponse, 
    SessionSummary,
    MessageResponse
)
from backend.services import (
    get_current_user,
    create_session,
    get_session,
    get_user_sessions,
    end_session,
    get_session_messages,
    get_persona
)
from backend.models import User

router = APIRouter(prefix="/sessions", tags=["Training Sessions"])


@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
def start_session(
    session_data: SessionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Start a new training session.
    
    The session will be active until ended via the /sessions/{id}/end endpoint
    or via WebSocket disconnect.
    """
    session = create_session(db, current_user, session_data)
    return SessionResponse.model_validate(session)


@router.get("", response_model=SessionListResponse)
def list_sessions(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get user's training session history.
    """
    sessions, total = get_user_sessions(db, current_user.id, limit, offset)
    
    # Convert to summaries with persona names
    summaries = []
    for session in sessions:
        persona = get_persona(db, session.persona_id)
        summaries.append(SessionSummary(
            id=session.id,
            persona_id=session.persona_id,
            persona_name=persona.name_ar if persona else None,
            status=session.status,
            difficulty=session.difficulty,
            started_at=session.started_at,
            duration_seconds=session.duration_seconds,
            overall_score=session.overall_score
        ))
    
    return SessionListResponse(sessions=summaries, total=total)


@router.get("/{session_id}", response_model=SessionResponse)
def get_session_detail(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get details of a specific session.
    """
    session = get_session(db, session_id)
    
    # Verify ownership
    if session.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this session"
        )
    
    return SessionResponse.model_validate(session)


@router.get("/{session_id}/messages", response_model=list[MessageResponse])
def get_messages(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get all messages from a session.
    """
    session = get_session(db, session_id)
    
    # Verify ownership
    if session.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this session"
        )
    
    messages = get_session_messages(db, session_id)
    return [MessageResponse.model_validate(m) for m in messages]


@router.post("/{session_id}/end", response_model=SessionResponse)
def end_training_session(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    End a training session.
    
    This will trigger the final evaluation.
    Note: In production, evaluation would be done here or via WebSocket.
    """
    session = get_session(db, session_id)
    
    # Verify ownership
    if session.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this session"
        )
    
    # Verify session is active
    if session.status != "active":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Session is already {session.status}"
        )
    
    # End session (evaluation would be added here)
    # For now, just end without evaluation
    session = end_session(db, session_id)
    
    return SessionResponse.model_validate(session)