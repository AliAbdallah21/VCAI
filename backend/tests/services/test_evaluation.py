# backend/tests/services/test_evaluation.py
"""
Unit tests for evaluation service.

Tests all CRUD operations and business logic for evaluation reports.
"""

import pytest
from uuid import uuid4
from datetime import datetime, timezone

from backend.models import EvaluationReport
from backend.services.evaluation_service import (
    generate_report_id,
    get_evaluation_report_by_session,
    get_evaluation_report_by_id,
    create_evaluation_report,
    update_evaluation_status,
    save_evaluation_result,
    compute_quick_stats,
    trigger_evaluation
)


# ═══════════════════════════════════════════════════════════════════════════════
# Test: generate_report_id
# ═══════════════════════════════════════════════════════════════════════════════

class TestGenerateReportId:
    """Tests for generate_report_id function."""
    
    def test_generates_unique_id(self):
        """Should generate a unique report ID each time."""
        id1 = generate_report_id()
        id2 = generate_report_id()
        
        assert id1 != id2
    
    def test_id_format(self):
        """Should generate ID with 'eval_' prefix."""
        report_id = generate_report_id()
        
        assert report_id.startswith("eval_")
        assert len(report_id) == 17  # "eval_" (5) + 12 hex chars


# ═══════════════════════════════════════════════════════════════════════════════
# Test: get_evaluation_report_by_session
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetEvaluationReportBySession:
    """Tests for get_evaluation_report_by_session function."""
    
    def test_returns_none_when_no_report(self, db, test_session):
        """Should return None when no evaluation exists."""
        result = get_evaluation_report_by_session(db, test_session.id)
        
        assert result is None
    
    def test_returns_report_when_exists(self, db, test_session):
        """Should return the report when it exists."""
        # Create a report
        report = EvaluationReport(
            session_id=test_session.id,
            report_id=generate_report_id(),
            status="pending"
        )
        db.add(report)
        db.commit()
        
        result = get_evaluation_report_by_session(db, test_session.id)
        
        assert result is not None
        assert result.session_id == test_session.id
        assert result.status == "pending"


# ═══════════════════════════════════════════════════════════════════════════════
# Test: get_evaluation_report_by_id
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetEvaluationReportById:
    """Tests for get_evaluation_report_by_id function."""
    
    def test_returns_none_for_invalid_id(self, db):
        """Should return None for non-existent report ID."""
        result = get_evaluation_report_by_id(db, "eval_nonexistent")
        
        assert result is None
    
    def test_returns_report_by_id(self, db, test_session):
        """Should return report by its report_id."""
        report_id = generate_report_id()
        report = EvaluationReport(
            session_id=test_session.id,
            report_id=report_id,
            status="completed"
        )
        db.add(report)
        db.commit()
        
        result = get_evaluation_report_by_id(db, report_id)
        
        assert result is not None
        assert result.report_id == report_id


# ═══════════════════════════════════════════════════════════════════════════════
# Test: create_evaluation_report
# ═══════════════════════════════════════════════════════════════════════════════

class TestCreateEvaluationReport:
    """Tests for create_evaluation_report function."""
    
    def test_creates_report_with_pending_status(self, db, test_session):
        """Should create a report with 'pending' status."""
        report = create_evaluation_report(db, test_session.id)
        
        assert report.status == "pending"
        assert report.progress == 0
        assert report.mode == "training"
        assert report.session_id == test_session.id
    
    def test_creates_report_with_custom_mode(self, db, test_session):
        """Should create a report with custom mode."""
        report = create_evaluation_report(db, test_session.id, mode="testing")
        
        assert report.mode == "testing"
    
    def test_raises_409_when_report_exists(self, db, test_session):
        """Should raise 409 Conflict when report already exists."""
        from fastapi import HTTPException
        
        # Create first report
        create_evaluation_report(db, test_session.id)
        
        # Try to create another
        with pytest.raises(HTTPException) as exc_info:
            create_evaluation_report(db, test_session.id)
        
        assert exc_info.value.status_code == 409
        assert "already exists" in str(exc_info.value.detail)
    
    def test_raises_404_for_invalid_session(self, db):
        """Should raise 404 when session doesn't exist."""
        from fastapi import HTTPException
        
        with pytest.raises(HTTPException) as exc_info:
            create_evaluation_report(db, uuid4())
        
        assert exc_info.value.status_code == 404
        assert "Session not found" in str(exc_info.value.detail)


# ═══════════════════════════════════════════════════════════════════════════════
# Test: update_evaluation_status
# ═══════════════════════════════════════════════════════════════════════════════

class TestUpdateEvaluationStatus:
    """Tests for update_evaluation_status function."""
    
    def test_updates_status(self, db, test_session):
        """Should update the evaluation status."""
        create_evaluation_report(db, test_session.id)
        
        result = update_evaluation_status(db, test_session.id, "processing")
        
        assert result.status == "processing"
    
    def test_updates_progress(self, db, test_session):
        """Should update progress when provided."""
        create_evaluation_report(db, test_session.id)
        
        result = update_evaluation_status(
            db, test_session.id, "processing", progress=50
        )
        
        assert result.progress == 50
    
    def test_sets_started_at_when_processing(self, db, test_session):
        """Should set started_at when status becomes 'processing'."""
        report = create_evaluation_report(db, test_session.id)
        assert report.started_at is None
        
        result = update_evaluation_status(db, test_session.id, "processing")
        
        assert result.started_at is not None
    
    def test_sets_completed_at_when_done(self, db, test_session):
        """Should set completed_at when status is 'completed' or 'failed'."""
        create_evaluation_report(db, test_session.id)
        
        result = update_evaluation_status(db, test_session.id, "completed")
        
        assert result.completed_at is not None
        assert result.progress == 100
    
    def test_stores_error_message(self, db, test_session):
        """Should store error message when provided."""
        create_evaluation_report(db, test_session.id)
        
        result = update_evaluation_status(
            db, test_session.id, "failed", error_message="LLM timeout"
        )
        
        assert result.error_message == "LLM timeout"
    
    def test_raises_404_when_report_not_found(self, db, test_session):
        """Should raise 404 when no report exists."""
        from fastapi import HTTPException
        
        with pytest.raises(HTTPException) as exc_info:
            update_evaluation_status(db, test_session.id, "processing")
        
        assert exc_info.value.status_code == 404


# ═══════════════════════════════════════════════════════════════════════════════
# Test: save_evaluation_result
# ═══════════════════════════════════════════════════════════════════════════════

class TestSaveEvaluationResult:
    """Tests for save_evaluation_result function."""
    
    def test_saves_report_json(self, db, test_session):
        """Should save the full report JSON."""
        create_evaluation_report(db, test_session.id)
        
        report_json = {
            "score_breakdown": {
                "overall_score": 85,
                "pass_threshold": 75
            },
            "passed": True,
            "summary": "Good performance"
        }
        
        result = save_evaluation_result(db, test_session.id, report_json)
        
        assert result.report_json == report_json
        assert result.overall_score == 85
        assert result.pass_threshold == 75
        assert result.passed == True
        assert result.status == "completed"
        assert result.progress == 100
    
    def test_raises_404_when_report_not_found(self, db, test_session):
        """Should raise 404 when no report exists."""
        from fastapi import HTTPException
        
        with pytest.raises(HTTPException) as exc_info:
            save_evaluation_result(db, test_session.id, {})
        
        assert exc_info.value.status_code == 404


# ═══════════════════════════════════════════════════════════════════════════════
# Test: compute_quick_stats
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeQuickStats:
    """Tests for compute_quick_stats function."""
    
    def test_computes_basic_stats(self, db, test_session, test_messages):
        """Should compute basic session statistics."""
        stats = compute_quick_stats(db, test_session.id)
        
        assert stats["duration_seconds"] == 300
        assert stats["total_turns"] == 4
        assert stats["salesperson_turns"] == 2
        assert stats["customer_turns"] == 2
    
    def test_computes_emotion_journey(self, db, test_session, test_messages, test_emotion_logs):
        """Should compute emotion journey from logs."""
        stats = compute_quick_stats(db, test_session.id)
        
        assert "emotion_journey" in stats
        assert len(stats["emotion_journey"]) > 0
    
    def test_raises_404_for_invalid_session(self, db):
        """Should raise 404 when session doesn't exist."""
        from fastapi import HTTPException
        
        with pytest.raises(HTTPException) as exc_info:
            compute_quick_stats(db, uuid4())
        
        assert exc_info.value.status_code == 404
    
    def test_saves_stats_to_report(self, db, test_session, test_messages):
        """Should save quick stats to report if it exists."""
        report = create_evaluation_report(db, test_session.id)
        
        stats = compute_quick_stats(db, test_session.id)
        
        # Refresh report to get updated quick_stats_json
        db.refresh(report)
        assert report.quick_stats_json == stats


# ═══════════════════════════════════════════════════════════════════════════════
# Test: trigger_evaluation
# ═══════════════════════════════════════════════════════════════════════════════

class TestTriggerEvaluation:
    """Tests for trigger_evaluation function."""
    
    def test_creates_new_report(self, db, test_session, test_messages):
        """Should create a new evaluation report."""
        report = trigger_evaluation(db, test_session.id)
        
        assert report is not None
        assert report.session_id == test_session.id
        assert report.status == "pending"
    
    def test_returns_existing_report(self, db, test_session, test_messages):
        """Should return existing report if one exists."""
        # Create first report
        report1 = trigger_evaluation(db, test_session.id)
        
        # Trigger again
        report2 = trigger_evaluation(db, test_session.id)
        
        assert report1.id == report2.id
        assert report1.report_id == report2.report_id
    
    def test_computes_quick_stats(self, db, test_session, test_messages):
        """Should compute quick stats when creating report."""
        report = trigger_evaluation(db, test_session.id)
        
        # Refresh to get updated data
        db.refresh(report)
        
        assert report.quick_stats_json is not None
        assert "total_turns" in report.quick_stats_json
