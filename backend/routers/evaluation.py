# backend/routers/evaluation.py
"""
Evaluation API endpoints.

URL structure (all under /api prefix set in main.py):
  POST /api/sessions/{session_id}/evaluate   — start background evaluation
  GET  /api/sessions/{session_id}/report     — poll for results
  GET  /api/sessions/{session_id}/eval-status — lightweight progress check
  GET  /api/sessions/{session_id}/quick-stats — immediate stats, no LLM needed
"""

from uuid import UUID
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.services.auth_service import get_current_user
from backend.models import User, Session as TrainingSession
from backend.services.evaluation_service import (
    get_evaluation_report_by_session,
    create_evaluation_report,
    compute_quick_stats,
    run_evaluation_background,
)


router = APIRouter(prefix="/sessions", tags=["Evaluation"])


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _require_session(db: Session, session_id: UUID, user: User) -> TrainingSession:
    """Return the session or raise 404/403."""
    session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    if session.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    return session


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/sessions/{session_id}/evaluate
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/{session_id}/evaluate")
def start_evaluation(
    session_id: UUID,
    background_tasks: BackgroundTasks,
    mode: str = "training",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Trigger evaluation for a completed session.

    Immediately returns {"status": "started", "session_id": ...}.
    The full LLM pipeline (EvaluationManager → LangGraph) runs in the background.
    Poll GET /report to check progress.

    mode: "training" (encouraging feedback) | "testing" (pass/fail)
    """
    training_session = _require_session(db, session_id, current_user)

    # Only completed/ended sessions can be evaluated
    if training_session.status not in ("completed", "ended"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Session status is '{training_session.status}'. "
                "End the session before requesting evaluation."
            ),
        )

    # Get or create the report row (handles race conditions internally)
    report = create_evaluation_report(db, session_id, mode)

    # Decide whether to dispatch background work
    if report.status == "failed":
        # Reset so we can retry
        from datetime import datetime, timezone
        report.status = "pending"
        report.progress = 0
        report.error_message = None
        report.started_at = None
        report.completed_at = None
        db.commit()

    if report.status in ("pending",):
        # FastAPI will run this after the response is sent
        background_tasks.add_task(run_evaluation_background, session_id, mode)

    # Always return immediately — frontend polls GET /report
    return {"status": "started", "session_id": str(session_id)}


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/sessions/{session_id}/report
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/{session_id}/report")
def get_report(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Poll for evaluation results.

    Possible responses:
      {"status": "not_started"}
      {"status": "pending",    "progress": 0}
      {"status": "processing", "progress": 40}
      {"status": "completed",  "report": {...}, "overall_score": 82, "passed": true}
      {"status": "failed",     "error": "..."}
    """
    _require_session(db, session_id, current_user)

    report = get_evaluation_report_by_session(db, session_id)

    if not report:
        return {"status": "not_started"}

    if report.status == "completed":
        return {
            "status": "completed",
            "report": report.report_json,
            "overall_score": report.overall_score,
            "passed": report.passed,
        }

    if report.status == "failed":
        return {
            "status": "failed",
            "error": report.error_message or "Evaluation failed — check server logs.",
        }

    # pending or processing
    return {
        "status": report.status,
        "progress": report.progress or 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/sessions/{session_id}/eval-status  (lightweight progress check)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/{session_id}/eval-status")
def get_eval_status(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Lightweight status check — returns status + progress without the full report JSON.
    Useful for progress bars while evaluation is running.
    """
    _require_session(db, session_id, current_user)

    report = get_evaluation_report_by_session(db, session_id)
    if not report:
        return {"status": "not_started", "progress": None}

    return {
        "status": report.status,
        "progress": report.progress,
        "report_id": report.report_id,
        "error": report.error_message if report.status == "failed" else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/sessions/{session_id}/quick-stats
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/{session_id}/quick-stats")
def get_quick_stats(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Return immediate statistics computed from session data (no LLM required).
    If a completed report exists, returns its cached quick_stats instead.
    """
    _require_session(db, session_id, current_user)

    # Return cached quick stats from existing report if available
    report = get_evaluation_report_by_session(db, session_id)
    if report and report.quick_stats_json:
        return {
            "session_id": str(session_id),
            "stats": report.quick_stats_json,
            "from_cache": True,
        }

    # Compute fresh stats
    stats = compute_quick_stats(db, session_id)
    return {
        "session_id": str(session_id),
        "stats": stats,
        "from_cache": False,
    }
