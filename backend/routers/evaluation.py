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
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.rate_limit import limiter
from backend.services.auth_service import get_current_user
from backend.models import User, Session as TrainingSession
from backend.services.evaluation_service import (
    get_evaluation_report_by_session,
    create_evaluation_report,
    compute_quick_stats,
    run_evaluation_background,
    get_eval_stage,
)


router = APIRouter(prefix="/sessions", tags=["Evaluation"])


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _require_session(db: Session, session_id: UUID, user: User) -> TrainingSession:
    """
    Return the session after applying tenant-scope rules, or raise 404.

    Delegates to the central scope helper (Phase 4): salesperson -> own session
    only; manager -> any session in their company; superadmin -> unrestricted.
    """
    from backend.services.scope_service import get_session_or_403
    return get_session_or_403(db, session_id, user)


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/sessions/{session_id}/evaluate
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/{session_id}/evaluate")
@limiter.limit("10/minute")  # tighter than the global 60/min — each call burns 2 Gemini requests
def start_evaluation(
    request: Request,
    session_id: UUID,
    background_tasks: BackgroundTasks,
    mode: str = "training",
    force: bool = False,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Trigger evaluation for a completed session.
    Pass force=true to re-run evaluation even if a report already exists.
    """
    from datetime import datetime, timezone
    from backend.services.session_service import _auto_end_stale
    training_session = _require_session(db, session_id, current_user)

    # Auto-end active sessions so they can be evaluated
    if training_session.status == "active":
        _auto_end_stale(db, training_session)
        db.refresh(training_session)

    if training_session.status not in ("completed", "ended"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Session status is '{training_session.status}'. End the session first.",
        )

    # Force re-evaluation: reset existing report
    if force:
        existing = get_evaluation_report_by_session(db, session_id)
        if existing:
            existing.status = "pending"
            existing.progress = 0
            existing.report_json = None
            existing.overall_score = None
            existing.passed = None
            existing.error_message = None
            existing.started_at = None
            existing.completed_at = None
            db.commit()

    report = create_evaluation_report(db, session_id, mode)

    if report.status == "failed":
        report.status = "pending"
        report.progress = 0
        report.error_message = None
        report.started_at = None
        report.completed_at = None
        db.commit()

    # Atomic claim: only the request that successfully transitions the row
    # from pending → processing (with started_at flipping from NULL to NOW)
    # gets to fire the background task. The other concurrent request loses
    # the race and returns quietly. This prevents the double-eval pattern
    # we saw where two browser tabs both fired Gemini and burned quota.
    if report.status == "pending":
        from datetime import datetime, timezone
        # Use UPDATE...WHERE...started_at IS NULL as the atomic gate.
        from sqlalchemy import update
        from backend.models.evaluation import EvaluationReport as _ER
        now = datetime.now(timezone.utc)
        result = db.execute(
            update(_ER)
            .where(_ER.id == report.id)
            .where(_ER.started_at.is_(None))
            .values(status="processing", started_at=now, progress=1)
        )
        db.commit()
        if result.rowcount == 1:
            # We won the race — fire the background task.
            background_tasks.add_task(run_evaluation_background, session_id, mode)
            print(f"[EVAL] Claimed evaluation lock for {session_id} — firing background task.")
        else:
            # Another request already claimed it; do nothing.
            print(f"[EVAL] Lost evaluation race for {session_id} — another request is processing.")

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

    # pending or processing — include live stage message from tracker
    live = get_eval_stage(session_id)
    return {
        "status": report.status,
        "progress": live.get("progress", report.progress or 0),
        "stage":    live.get("stage", "Initializing..."),
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
