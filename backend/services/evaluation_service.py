# backend/services/evaluation_service.py
"""
Evaluation service - handles evaluation report management.
"""

from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from uuid import UUID, uuid4
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException, status

from backend.models.evaluation import EvaluationReport
from backend.models import Session as TrainingSession, Message, EmotionLog


# Thread pool for background evaluation
_executor = ThreadPoolExecutor(max_workers=2)


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
    Create a new evaluation report for a session.
    Handles race conditions gracefully.
    """
    
    # Check if session exists
    session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    # Check if report already exists - RETURN IT instead of deleting
    existing_report = db.query(EvaluationReport).filter(
        EvaluationReport.session_id == session_id
    ).first()
    
    if existing_report:
        print(f"[EVAL] Report already exists for session {session_id} (status: {existing_report.status})")
        return existing_report
    
    # Try to create new report with race condition handling
    try:
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
        print(f"[EVAL] Created new report for session {session_id}")
        return report
        
    except IntegrityError:
        # Race condition - another request created it first
        db.rollback()
        existing = db.query(EvaluationReport).filter(
            EvaluationReport.session_id == session_id
        ).first()
        if existing:
            print(f"[EVAL] Race condition handled - returning existing report for {session_id}")
            return existing
        raise  # Re-raise if it's a different error


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


def _run_evaluation_background(session_id: UUID, mode: str):
    """
    Run evaluation in background thread.
    Calls Menna's EvaluationManager which runs Ismail's LangGraph pipeline.
    """
    from backend.database import get_db_context
    
    print(f"[EVAL] Starting evaluation for session {session_id} (mode={mode})")
    
    try:
        # Import your team's evaluation system
        from evaluation import EvaluationManager
        
        # Update status to processing
        with get_db_context() as db:
            report = db.query(EvaluationReport).filter(
                EvaluationReport.session_id == session_id
            ).first()
            if report:
                report.status = "processing"
                report.progress = 10
                report.started_at = datetime.now(timezone.utc)
                db.commit()
        
        # Run Menna's EvaluationManager
        manager = EvaluationManager(verbose=True)
        final_report = manager.evaluate(
            session_id=str(session_id),
            mode=mode
        )
        
        # Save result to database
        with get_db_context() as db:
            report = db.query(EvaluationReport).filter(
                EvaluationReport.session_id == session_id
            ).first()
            
            if report:
                # Convert Pydantic model to dict
                report_dict = final_report.model_dump(mode="json") if hasattr(final_report, 'model_dump') else final_report.dict()
                
                report.report_json = report_dict
                report.overall_score = report_dict.get("scores", {}).get("overall_score")
                report.pass_threshold = report_dict.get("scores", {}).get("pass_threshold", 75)
                report.passed = report_dict.get("passed")
                report.status = "completed"
                report.progress = 100
                report.completed_at = datetime.now(timezone.utc)
                db.commit()
        
        print(f"[EVAL] ✅ Evaluation completed for session {session_id}")
        print(f"[EVAL] Score: {final_report.scores.overall_score}/100")
        
    except Exception as e:
        print(f"[EVAL] ❌ Error evaluating session {session_id}: {e}")
        import traceback
        traceback.print_exc()
        
        # Mark as failed
        try:
            with get_db_context() as db:
                report = db.query(EvaluationReport).filter(
                    EvaluationReport.session_id == session_id
                ).first()
                if report:
                    report.status = "failed"
                    report.error_message = str(e)
                    report.completed_at = datetime.now(timezone.utc)
                    db.commit()
        except Exception:
            pass


def trigger_evaluation(
    db: Session,
    session_id: UUID,
    mode: str = "training"
) -> EvaluationReport:
    """
    Trigger evaluation for a session.
    
    Creates a report with 'pending' status and starts evaluation in background.
    Handles duplicate requests gracefully.
    """
    # Check for existing report FIRST
    existing = get_evaluation_report_by_session(db, session_id)
    
    if existing:
        # If already processing or completed or pending, just return it
        if existing.status in ("processing", "completed", "pending"):
            print(f"[EVAL] Report already exists for {session_id} (status: {existing.status})")
            return existing
        
        # If failed, allow retry
        if existing.status == "failed":
            print(f"[EVAL] Retrying failed evaluation for {session_id}")
            existing.status = "pending"
            existing.progress = 0
            existing.error_message = None
            existing.started_at = None
            existing.completed_at = None
            db.commit()
            db.refresh(existing)
            
            # Start evaluation in background
            _executor.submit(_run_evaluation_background, session_id, mode)
            return existing
    
    # Create new report (handles race conditions internally)
    report = create_evaluation_report(db, session_id, mode)
    
    # Only start background evaluation if we created a new pending report
    if report.status == "pending" and report.progress == 0:
        # Compute and store quick stats
        try:
            compute_quick_stats(db, session_id)
        except Exception as e:
            print(f"[EVAL] Quick stats error: {e}")
        
        # Start evaluation in background thread
        _executor.submit(_run_evaluation_background, session_id, mode)
    
    return report