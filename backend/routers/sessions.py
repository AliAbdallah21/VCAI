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
            overall_score=session.overall_score,
            turn_count=session.turn_count or 0
        ))
    
    return SessionListResponse(sessions=summaries, total=total)


@router.get("/scenario-presets")
def get_scenario_presets(
    current_user: User = Depends(get_current_user),
):
    """
    Return the curated buyer-scenario presets for the session-setup picker.

    Declared BEFORE /{session_id} so the literal path matches first
    (a parametrized UUID route would otherwise swallow 'scenario-presets').
    """
    from shared.scenarios import list_presets
    return {"presets": list_presets()}


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


@router.get("/{session_id}/messages/{message_id}/audio")
def get_message_audio(
    session_id: UUID,
    message_id: UUID,
    token: str = Query(..., description="JWT token (HTML5 <audio> can't send the Authorization header)"),
    db: Session = Depends(get_db),
):
    """
    Stream the saved audio for one message of a session.

    HTML5 <audio> tags don't include the Authorization header, so this
    endpoint authenticates via the `?token=` query parameter instead of
    the usual `Depends(get_current_user)`. The session's user_id is
    verified to match the token owner.
    """
    from fastapi.responses import FileResponse
    from backend.services.auth_service import decode_token
    from backend.services.audio_storage import resolve_audio_path
    from backend.models import Message

    # ── Auth: validate token from query, find user ──
    token_data = decode_token(token)
    if token_data is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    user = db.query(User).filter(User.id == token_data.user_id).first()
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    # ── Verify session ownership ──
    session = get_session(db, session_id)
    if session.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="You don't have access to this session")

    # ── Look up the message and its audio path ──
    msg = (
        db.query(Message)
        .filter(Message.id == message_id, Message.session_id == session_id)
        .first()
    )
    if msg is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Message not found")
    if not msg.audio_path:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No audio saved for this message")

    abs_path = resolve_audio_path(msg.audio_path)
    if abs_path is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Audio file not available on disk")

    suffix = abs_path.suffix.lower()
    media_type = {
        ".wav":  "audio/wav",
        ".webm": "audio/webm",
        ".mp3":  "audio/mpeg",
        ".ogg":  "audio/ogg",
    }.get(suffix, "application/octet-stream")

    return FileResponse(str(abs_path), media_type=media_type, filename=abs_path.name)


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
    
    session = end_session(db, session_id)
    return SessionResponse.model_validate(session)


@router.post("/{session_id}/reactivate", response_model=SessionResponse)
def reactivate_session(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Reactivate an ended/completed session so the user can resume the conversation."""
    session = get_session(db, session_id)
    if session.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    session.status = "active"
    session.ended_at = None
    db.commit()
    db.refresh(session)
    return SessionResponse.model_validate(session)