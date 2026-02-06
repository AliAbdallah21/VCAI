# backend/routers/evaluation.py
"""
Evaluation API endpoints.
"""

from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, Dict, Any

from backend.database import get_db
from backend.services.auth_service import get_current_user
from backend.models import User, Session as TrainingSession
from backend.services.evaluation_service import (
    get_evaluation_report_by_session,
    trigger_evaluation,
    compute_quick_stats
)


router = APIRouter(prefix="/evaluation", tags=["Evaluation"])


# ═══════════════════════════════════════════════════════════════════════════════
# Response Models
# ═══════════════════════════════════════════════════════════════════════════════

class EvaluationTriggerResponse(BaseModel):
    """Response when evaluation is triggered."""
    session_id: str
    status: str  # "started", "already_exists", "error"
    message: str
    report_id: Optional[str] = None


class EvaluationStatusResponse(BaseModel):
    """Response for evaluation status check."""
    session_id: str
    status: str  # "pending", "processing", "completed", "failed", "not_found"
    progress: Optional[int] = None
    report_id: Optional[str] = None
    error: Optional[str] = None


class QuickStatsResponse(BaseModel):
    """Response for immediate stats after call ends."""
    session_id: str
    stats: Dict[str, Any]
    evaluation_available: bool = True


class EvaluationReportResponse(BaseModel):
    """Response containing the full evaluation report."""
    session_id: str
    report_id: str
    status: str
    mode: str
    overall_score: Optional[int] = None
    passed: Optional[bool] = None
    report: Optional[Dict[str, Any]] = None
    quick_stats: Optional[Dict[str, Any]] = None


# ═══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════════

def verify_session_ownership(
    db: Session,
    session_id: UUID,
    user: User
) -> TrainingSession:
    """Verify user owns the session."""
    training_session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
    
    if not training_session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    if training_session.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this session"
        )
    
    return training_session


# ═══════════════════════════════════════════════════════════════════════════════
# API Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/{session_id}/trigger", response_model=EvaluationTriggerResponse)
def trigger_session_evaluation(
    session_id: UUID,
    mode: str = "training",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Trigger evaluation for a completed session.
    
    Creates an evaluation report with 'pending' status.
    The evaluation pipeline will process it asynchronously.
    """
    # Verify ownership
    training_session = verify_session_ownership(db, session_id, current_user)
    
    # Check session is completed
    if training_session.status not in ("completed", "ended"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot evaluate session with status '{training_session.status}'. Session must be completed first."
        )
    
    # Check for existing evaluation
    existing = get_evaluation_report_by_session(db, session_id)
    if existing:
        return EvaluationTriggerResponse(
            session_id=str(session_id),
            status="already_exists",
            message=f"Evaluation already exists (status: {existing.status})",
            report_id=existing.report_id
        )
    
    # Trigger evaluation
    report = trigger_evaluation(db, session_id, mode)
    
    return EvaluationTriggerResponse(
        session_id=str(session_id),
        status="started",
        message="Evaluation started successfully",
        report_id=report.report_id
    )


@router.get("/{session_id}/status", response_model=EvaluationStatusResponse)
def get_evaluation_status(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get the current status of an evaluation.
    
    Returns progress percentage if evaluation is processing.
    """
    # Verify ownership
    verify_session_ownership(db, session_id, current_user)
    
    # Get evaluation report
    report = get_evaluation_report_by_session(db, session_id)
    
    if not report:
        return EvaluationStatusResponse(
            session_id=str(session_id),
            status="not_found",
            progress=None
        )
    
    return EvaluationStatusResponse(
        session_id=str(session_id),
        status=report.status,
        progress=report.progress,
        report_id=report.report_id,
        error=report.error_message
    )


@router.get("/{session_id}/report", response_model=EvaluationReportResponse)
def get_evaluation_report(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get the full evaluation report for a session.
    
    Returns 404 if evaluation hasn't been completed yet.
    """
    # Verify ownership
    verify_session_ownership(db, session_id, current_user)
    
    # Get evaluation report
    report = get_evaluation_report_by_session(db, session_id)
    
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No evaluation found for this session. Trigger an evaluation first."
        )
    
    return EvaluationReportResponse(
        session_id=str(session_id),
        report_id=report.report_id,
        status=report.status,
        mode=report.mode,
        overall_score=report.overall_score,
        passed=report.passed,
        report=report.report_json,
        quick_stats=report.quick_stats_json
    )


@router.get("/{session_id}/quick-stats", response_model=QuickStatsResponse)
def get_quick_stats(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get immediate statistics for a session.
    
    These stats are computed from session data and don't require
    the full LLM evaluation pipeline.
    """
    # Verify ownership
    verify_session_ownership(db, session_id, current_user)
    
    # Check for cached stats in report
    report = get_evaluation_report_by_session(db, session_id)
    if report and report.quick_stats_json:
        return QuickStatsResponse(
            session_id=str(session_id),
            stats=report.quick_stats_json,
            evaluation_available=(report.status != "completed")
        )
    
    # Compute stats
    stats = compute_quick_stats(db, session_id)
    
    return QuickStatsResponse(
        session_id=str(session_id),
        stats=stats,
        evaluation_available=True
    )
