# backend/services/evaluation_service.py
"""
Evaluation service - handles evaluation report management.
"""

from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from uuid import UUID, uuid4

from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from backend.models.evaluation import EvaluationReport
from backend.models import Session as TrainingSession, Message, EmotionLog


def generate_report_id() -> str:
    """Generate a unique report ID."""
    return f"eval_{uuid4().hex[:12]}"


def get_evaluation_report_by_session(db: Session, session_id: UUID) -> Optional[EvaluationReport]:
    """Get evaluation report by session ID."""
    return db.query(EvaluationReport).filter(
        EvaluationReport.session_id == session_id
    ).first()


def get_evaluation_report_by_id(db: Session, report_id: str) -> Optional[EvaluationReport]:
    """Get evaluation report by report ID."""
    return db.query(EvaluationReport).filter(
        EvaluationReport.report_id == report_id
    ).first()


def create_evaluation_report(
    db: Session,
    session_id: UUID,
    mode: str = "training"
) -> EvaluationReport:
    """
    Create a new evaluation report with 'pending' status.
    
    Raises HTTPException if report already exists for session.
    """
    # Check for existing report
    existing = get_evaluation_report_by_session(db, session_id)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Evaluation already exists for session (status: {existing.status})"
        )
    
    # Verify session exists
    training_session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
    if not training_session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    # Create report
    report = EvaluationReport(
        session_id=session_id,
        report_id=generate_report_id(),
        status="pending",
        mode=mode,
        progress=0
    )
    
    db.add(report)
    db.commit()
    db.refresh(report)
    
    return report


def update_evaluation_status(
    db: Session,
    session_id: UUID,
    status_value: str,
    progress: int = None,
    error_message: str = None
) -> EvaluationReport:
    """Update evaluation status and progress."""
    report = get_evaluation_report_by_session(db, session_id)
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation report not found"
        )
    
    report.status = status_value
    if progress is not None:
        report.progress = progress
    if error_message:
        report.error_message = error_message
    
    if status_value == "processing" and not report.started_at:
        report.started_at = datetime.now(timezone.utc)
    elif status_value in ("completed", "failed"):
        report.completed_at = datetime.now(timezone.utc)
        report.progress = 100
    
    db.commit()
    db.refresh(report)
    
    return report


def save_evaluation_result(
    db: Session,
    session_id: UUID,
    report_json: Dict[str, Any]
) -> EvaluationReport:
    """
    Save completed evaluation result.
    
    Extracts key fields from report_json for query optimization.
    """
    report = get_evaluation_report_by_session(db, session_id)
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation report not found"
        )
    
    # Extract scores from report
    score_breakdown = report_json.get("score_breakdown", {})
    
    report.report_json = report_json
    report.overall_score = score_breakdown.get("overall_score")
    report.pass_threshold = score_breakdown.get("pass_threshold", 75)
    report.passed = report_json.get("passed", False)
    report.status = "completed"
    report.progress = 100
    report.completed_at = datetime.now(timezone.utc)
    
    db.commit()
    db.refresh(report)
    
    return report


def compute_quick_stats(db: Session, session_id: UUID) -> Dict[str, Any]:
    """
    Compute quick stats from session data.
    
    These stats don't require LLM evaluation - just data aggregation.
    """
    # Get session
    training_session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
    if not training_session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    # Get messages
    messages = db.query(Message).filter(Message.session_id == session_id).all()
    
    # Get emotion logs
    emotion_logs = db.query(EmotionLog).filter(EmotionLog.session_id == session_id).all()
    
    # Compute stats
    salesperson_turns = [m for m in messages if m.speaker == "salesperson"]
    customer_turns = [m for m in messages if m.speaker == "customer"]
    
    # Emotion journey summary
    emotions = [log.customer_emotion for log in emotion_logs if log.customer_emotion]
    emotion_journey = []
    if emotions:
        # Get unique transitions
        prev = None
        for emo in emotions:
            if emo != prev:
                emotion_journey.append(emo)
                prev = emo
    
    quick_stats = {
        "duration_seconds": training_session.duration_seconds or 0,
        "total_turns": len(messages),
        "salesperson_turns": len(salesperson_turns),
        "customer_turns": len(customer_turns),
        "emotion_journey": emotion_journey[:5],  # First 5 transitions
        "checkpoints_achieved": [],  # Would need checkpoint analysis
        "call_outcome": "completed"  # Would need analysis to determine
    }
    
    # Save quick stats to report if it exists
    report = get_evaluation_report_by_session(db, session_id)
    if report:
        report.quick_stats_json = quick_stats
        db.commit()
    
    return quick_stats


def trigger_evaluation(
    db: Session,
    session_id: UUID,
    mode: str = "training"
) -> EvaluationReport:
    """
    Trigger evaluation for a session.
    
    Creates a report with 'pending' status. Actual LangGraph pipeline
    execution would be triggered here in production.
    
    Returns the created/existing report.
    """
    # Check for existing report
    existing = get_evaluation_report_by_session(db, session_id)
    if existing:
        return existing
    
    # Create new report
    report = create_evaluation_report(db, session_id, mode)
    
    # Compute and store quick stats
    try:
        compute_quick_stats(db, session_id)
    except Exception:
        pass  # Quick stats are optional
    
    # TODO: Trigger LangGraph evaluation pipeline here
    # For now, just return the pending report
    
    return report
