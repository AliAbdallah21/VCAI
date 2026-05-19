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

# ── In-memory stage tracker ───────────────────────────────────────────────────
# Maps str(session_id) → (progress: int, stage_message: str)
# Written by the background worker, read by the /report endpoint in real-time.
_eval_stages: dict[str, tuple[int, str]] = {}


def _set_stage(session_id, progress: int, msg: str) -> None:
    """Update the in-memory stage + DB progress atomically."""
    key = str(session_id)
    _eval_stages[key] = (progress, msg)
    print(f"[EVAL {progress:>3}%] {msg}")
    try:
        from backend.database import get_db_context
        with get_db_context() as db:
            r = db.query(EvaluationReport).filter(
                EvaluationReport.session_id == session_id
            ).first()
            if r:
                r.progress = progress
                db.commit()
    except Exception:
        pass  # Progress update is best-effort


def get_eval_stage(session_id) -> dict:
    """Return the live stage info for a session (or empty dict if none)."""
    data = _eval_stages.get(str(session_id))
    if data:
        return {"progress": data[0], "stage": data[1]}
    return {}


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


def update_evaluation_status(
    db: Session,
    session_id: UUID,
    status_value: str,
    *,
    progress: Optional[int] = None,
    error_message: Optional[str] = None,
) -> EvaluationReport:
    """Update an evaluation report's status (and related timestamps).

    Transitions:
      - "processing" stamps started_at (first time only).
      - "completed"/"failed" stamp completed_at; "completed" also forces
        progress to 100.
      - progress / error_message are applied when provided.

    Raises 404 if no report exists for the session.
    """
    report = get_evaluation_report_by_session(db, session_id)
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation report not found"
        )

    report.status = status_value

    if status_value == "processing" and report.started_at is None:
        report.started_at = datetime.now(timezone.utc)

    if status_value in ("completed", "failed"):
        report.completed_at = datetime.now(timezone.utc)
    if status_value == "completed":
        report.progress = 100

    if progress is not None:
        report.progress = progress
    if error_message is not None:
        report.error_message = error_message

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
    
    # Prefer first→last message span over stored value (avoids idle pre-session time)
    msgs_with_ts = sorted([m for m in messages if m.created_at], key=lambda m: m.created_at)
    if len(msgs_with_ts) >= 2:
        from datetime import timezone as _tz
        t0 = msgs_with_ts[0].created_at
        t1 = msgs_with_ts[-1].created_at
        if t0.tzinfo is None:
            t0 = t0.replace(tzinfo=_tz.utc)
        if t1.tzinfo is None:
            t1 = t1.replace(tzinfo=_tz.utc)
        duration_seconds = max(0, int((t1 - t0).total_seconds()))
    else:
        duration_seconds = training_session.duration_seconds or 0

    quick_stats = {
        "duration_seconds": duration_seconds,
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


def run_evaluation_background(session_id: UUID, mode: str):
    """
    Run evaluation in a background worker.
    Calls each pipeline node explicitly so we can emit live stage updates
    and validate inputs before making any expensive LLM calls.
    """
    from backend.database import get_db_context

    sid = str(session_id)

    def stage(progress: int, msg: str):
        _set_stage(session_id, progress, msg)

    def fail(msg: str):
        stage(0, f"Failed: {msg}")
        try:
            with get_db_context() as db:
                r = db.query(EvaluationReport).filter(
                    EvaluationReport.session_id == session_id
                ).first()
                if r:
                    r.status = "failed"
                    r.error_message = msg
                    r.completed_at = datetime.now(timezone.utc)
                    db.commit()
        except Exception:
            pass

    print(f"[EVAL] ▶ Starting evaluation for session {sid} (mode={mode})")

    # ── Mark as processing ────────────────────────────────────────────────────
    try:
        with get_db_context() as db:
            r = db.query(EvaluationReport).filter(
                EvaluationReport.session_id == session_id
            ).first()
            if r:
                r.status = "processing"
                r.progress = 10
                r.started_at = datetime.now(timezone.utc)
                db.commit()
    except Exception as e:
        print(f"[EVAL] ⚠ Could not mark processing: {e}")

    try:
        # ── STEP 1: Fetch conversation from DB ────────────────────────────────
        stage(12, "Connecting to database and loading conversation...")
        from evaluation.manager import gather_evaluation_inputs_db_only
        db_inputs = gather_evaluation_inputs_db_only(sid)

        n_turns = len(db_inputs["transcript"])
        n_salesperson = sum(1 for t in db_inputs["transcript"] if t["speaker"] == "salesperson")
        n_customer    = sum(1 for t in db_inputs["transcript"] if t["speaker"] == "customer")

        print(f"[EVAL] DB loaded: {n_turns} turns ({n_salesperson} salesperson, {n_customer} customer)")

        # ── Critical guard: empty transcript ─────────────────────────────────
        if n_turns == 0:
            fail(
                "No conversation turns found in the database. "
                "The session may not have saved messages correctly. "
                "Check the WebSocket handler and memory_save node."
            )
            return

        if n_salesperson == 0:
            fail(
                f"Found {n_turns} turns but none from the salesperson. "
                "Check that the STT/transcription pipeline is saving messages "
                "with speaker='salesperson'."
            )
            return

        # ── STEP 2: Fact-check via RAG ────────────────────────────────────────
        stage(22, f"Loaded {n_turns} turns ({n_salesperson} salesperson) — fact-checking property claims...")
        from evaluation.manager import _gather_fact_check_context
        try:
            fact_check = _gather_fact_check_context(db_inputs["transcript"])
            n_claims = fact_check.get("claims_checked", 0)
            n_errors = len(fact_check.get("errors", []))
            stage(34, f"Fact-check done: {n_claims} claims, {n_errors} errors — building evaluation state...")
        except Exception as fc_err:
            print(f"[EVAL] ⚠ Fact-check failed ({fc_err}) — continuing without it")
            fact_check = {}
            stage(34, "Fact-check skipped — building evaluation state...")

        # ── STEP 3: Build LangGraph state ─────────────────────────────────────
        from evaluation.state import create_initial_state
        from evaluation.manager import get_evaluation_llm
        state = create_initial_state(sid, mode)
        state["llm"]                  = get_evaluation_llm()
        state["transcript"]           = db_inputs["transcript"]
        state["emotion_log"]          = db_inputs["emotion_log"]
        state["rag_context"]          = []
        state["structured_fact_check"]= fact_check
        state["session_info"]         = db_inputs["session_info"]

        # ── STEP 4: Quick stats (no LLM) ──────────────────────────────────────
        stage(36, "Computing session statistics...")
        from evaluation.utils.report_formatter import compute_quick_stats_node
        state = compute_quick_stats_node(state)

        # ── STEP 5: Analyzer — LLM call #1 ────────────────────────────────────
        stage(40, f"Sending {n_turns} conversation turns to Claude for deep analysis...")
        from evaluation.pipeline.analyzer import analyzer_node
        state = analyzer_node(state)

        if state.get("errors"):
            fail("Analyzer failed: " + "; ".join(state["errors"]))
            return

        if not state.get("analysis_report"):
            fail("Analyzer returned no analysis_report — check prompts and LLM response parsing.")
            return

        print(f"[EVAL] ✓ Analyzer complete")
        stage(72, "Analysis complete — sending to Claude for final report synthesis...")

        # ── STEP 6: Synthesizer — LLM call #2 ────────────────────────────────
        stage(76, "Generating scored performance report...")
        from evaluation.pipeline.synthesizer import synthesizer_node
        state = synthesizer_node(state)

        if state.get("errors"):
            fail("Synthesizer failed: " + "; ".join(state["errors"]))
            return

        final_report = state.get("final_report")
        if not final_report:
            fail("Synthesizer returned no final_report — check prompts and LLM response parsing.")
            return

        print(f"[EVAL] ✓ Synthesizer complete")

        # ── STEP 7: Save to DB ────────────────────────────────────────────────
        stage(95, "Saving report to database...")
        report_dict = (
            final_report.model_dump(mode="json")
            if hasattr(final_report, "model_dump")
            else final_report.dict()
        )

        with get_db_context() as db:
            r = db.query(EvaluationReport).filter(
                EvaluationReport.session_id == session_id
            ).first()
            if r:
                overall = report_dict.get("scores", {}).get("overall_score")
                r.report_json     = report_dict
                r.overall_score   = overall
                r.pass_threshold  = report_dict.get("scores", {}).get("pass_threshold", 75)
                r.passed          = report_dict.get("passed")
                r.status          = "completed"
                r.progress        = 100
                r.completed_at    = datetime.now(timezone.utc)
                # Write scores back to session for learning service queries
                s = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
                if s and overall is not None:
                    s.overall_score = overall
                    # Map eval skill keys → session skill columns
                    skill_rows = {
                        sk.get("skill_key"): sk.get("score")
                        for sk in report_dict.get("scores", {}).get("skills", [])
                        if sk.get("skill_key") and sk.get("score") is not None
                    }
                    _EVAL_TO_SESSION = {
                        "communication_clarity": "communication_score",
                        "product_knowledge":     "product_knowledge_score",
                        "objection_handling":    "objection_handling_score",
                        "rapport_building":      "rapport_score",
                        "closing_skills":        "closing_score",
                    }
                    for eval_key, col in _EVAL_TO_SESSION.items():
                        if eval_key in skill_rows:
                            setattr(s, col, skill_rows[eval_key])
                db.commit()

        score = report_dict.get("scores", {}).get("overall_score", "?")
        print(f"[EVAL] ✅ Evaluation complete for {sid} — score: {score}/100")

        # ── STEP 8: Save report JSON to disk ─────────────────────────────────
        try:
            import json as _json
            import pathlib as _pl
            reports_dir = _pl.Path("C:/VCAI/reports")
            reports_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = reports_dir / f"report_{sid[:8]}_{ts}.json"
            with open(filename, "w", encoding="utf-8") as f:
                _json.dump(report_dict, f, ensure_ascii=False, indent=2, default=str)
            print(f"[EVAL] 📄 Report saved to {filename}")
        except Exception as disk_err:
            print(f"[EVAL] ⚠ Could not save report to disk: {disk_err}")

        stage(100, "Complete!")
        _eval_stages.pop(sid, None)  # Clean up tracker

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[EVAL] ❌ Unhandled error for {sid}: {e}")
        fail(str(e))


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
            _executor.submit(run_evaluation_background, session_id, mode)
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
        
        # Start evaluation in background thread (legacy path; new router uses BackgroundTasks)
        _executor.submit(run_evaluation_background, session_id, mode)

    return report