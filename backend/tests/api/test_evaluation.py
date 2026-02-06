# backend/tests/api/test_evaluation.py
"""
API integration tests for evaluation endpoints.

Tests all evaluation API endpoints with mocked authentication.
"""

import pytest
from uuid import uuid4, UUID
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, JSON
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import JSONB

from backend.database import Base, get_db
from backend.main import app
from backend.models import User, Session as TrainingSession, Message, EmotionLog, EvaluationReport
from backend.models.persona import Persona
from backend.services.auth_service import get_current_user
from backend.services.evaluation_service import generate_report_id


# ═══════════════════════════════════════════════════════════════════════════════
# SQLite Compatibility - Compile JSONB as JSON for SQLite
# ═══════════════════════════════════════════════════════════════════════════════

from sqlalchemy.dialects import sqlite

def _compile_jsonb_sqlite(type_, compiler, **kw):
    return "JSON"

# Register the type compiler
sqlite.base.SQLiteTypeCompiler.visit_JSONB = _compile_jsonb_sqlite


# ═══════════════════════════════════════════════════════════════════════════════
# Test Database Setup
# ═══════════════════════════════════════════════════════════════════════════════

# Create test database - use file-based SQLite for session persistence
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_evaluation.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Global variable to hold the current test's database session
_test_db_session = None


def override_get_db():
    """Override database dependency for testing - returns the shared test session."""
    global _test_db_session
    if _test_db_session is not None:
        yield _test_db_session
    else:
        # Fallback if no session set
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Test Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="function")
def test_db():
    """
    Create a database session for each test.
    
    This session is shared between fixtures and API calls.
    """
    global _test_db_session
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    # Create and store session globally so override_get_db can access it
    session = TestingSessionLocal()
    _test_db_session = session
    
    try:
        yield session
    finally:
        _test_db_session = None
        session.close()
        # Drop all tables after test
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def test_user(test_db):
    """Create a test user."""
    user = User(
        id=uuid4(),
        email="testuser@example.com",
        password_hash="hashed_password",
        full_name="Test User"
    )
    test_db.add(user)
    test_db.commit()
    test_db.refresh(user)
    return user


@pytest.fixture
def other_user(test_db):
    """Create another test user for ownership tests."""
    user = User(
        id=uuid4(),
        email="other@example.com",
        password_hash="hashed_password",
        full_name="Other User"
    )
    test_db.add(user)
    test_db.commit()
    test_db.refresh(user)
    return user


@pytest.fixture
def test_persona(test_db):
    """Create a test persona."""
    persona = Persona(
        id="test_persona",
        name_ar="عميل تجريبي",
        name_en="Test Customer",
        personality_prompt="You are a test customer.",
        difficulty="medium"
    )
    test_db.add(persona)
    test_db.commit()
    return persona


@pytest.fixture
def completed_session(test_db, test_user, test_persona):
    """Create a completed training session."""
    session = TrainingSession(
        id=uuid4(),
        user_id=test_user.id,
        persona_id=test_persona.id,
        status="completed",
        difficulty="medium",
        duration_seconds=300,
        turn_count=6
    )
    test_db.add(session)
    test_db.commit()
    test_db.refresh(session)
    return session


@pytest.fixture
def active_session(test_db, test_user, test_persona):
    """Create an active (not completed) training session."""
    session = TrainingSession(
        id=uuid4(),
        user_id=test_user.id,
        persona_id=test_persona.id,
        status="active",
        difficulty="medium"
    )
    test_db.add(session)
    test_db.commit()
    test_db.refresh(session)
    return session


@pytest.fixture
def session_with_messages(test_db, completed_session):
    """Add messages to a session."""
    messages = [
        Message(
            session_id=completed_session.id,
            turn_number=1,
            speaker="customer",
            text="Hello, I need help with pricing."
        ),
        Message(
            session_id=completed_session.id,
            turn_number=2,
            speaker="salesperson",
            text="Of course! Let me explain our options."
        ),
        Message(
            session_id=completed_session.id,
            turn_number=3,
            speaker="customer",
            text="What's the difference between Basic and Pro?"
        ),
        Message(
            session_id=completed_session.id,
            turn_number=4,
            speaker="salesperson",
            text="Pro includes advanced features like analytics and priority support."
        ),
    ]
    for msg in messages:
        test_db.add(msg)
    test_db.commit()
    return messages


@pytest.fixture
def session_with_emotions(test_db, completed_session, session_with_messages):
    """Add emotion logs to a session."""
    logs = [
        EmotionLog(
            session_id=completed_session.id,
            customer_emotion="neutral",
            customer_mood_score=0,
            risk_level="low"
        ),
        EmotionLog(
            session_id=completed_session.id,
            customer_emotion="interested",
            customer_mood_score=30,
            risk_level="low"
        ),
    ]
    for log in logs:
        test_db.add(log)
    test_db.commit()
    return logs


@pytest.fixture
def existing_evaluation(test_db, completed_session):
    """Create an existing evaluation report."""
    report = EvaluationReport(
        session_id=completed_session.id,
        report_id=generate_report_id(),
        status="completed",
        mode="training",
        progress=100,
        overall_score=85,
        passed=True,
        report_json={"summary": "Good performance"},
        quick_stats_json={"total_turns": 4}
    )
    test_db.add(report)
    test_db.commit()
    test_db.refresh(report)
    return report


@pytest.fixture
def client(test_user):
    """Create a test client with mocked authentication."""
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = lambda: test_user
    
    with TestClient(app) as c:
        yield c
    
    app.dependency_overrides.clear()


@pytest.fixture
def client_other_user(other_user):
    """Create a test client authenticated as a different user."""
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = lambda: other_user
    
    with TestClient(app) as c:
        yield c
    
    app.dependency_overrides.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# Test: POST /evaluation/{session_id}/trigger
# ═══════════════════════════════════════════════════════════════════════════════

class TestTriggerEvaluationEndpoint:
    """Tests for trigger evaluation endpoint."""
    
    def test_trigger_evaluation_success(self, client, completed_session, session_with_messages):
        """Should trigger evaluation for a completed session."""
        response = client.post(f"/api/evaluation/{completed_session.id}/trigger")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"
        assert data["session_id"] == str(completed_session.id)
        assert data["report_id"] is not None
        assert data["report_id"].startswith("eval_")
    
    def test_trigger_with_custom_mode(self, client, completed_session, session_with_messages):
        """Should trigger evaluation with custom mode."""
        response = client.post(
            f"/api/evaluation/{completed_session.id}/trigger",
            params={"mode": "testing"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"
    
    def test_trigger_returns_existing_when_exists(self, client, completed_session, existing_evaluation):
        """Should return existing evaluation if one already exists."""
        response = client.post(f"/api/evaluation/{completed_session.id}/trigger")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "already_exists"
        assert data["report_id"] == existing_evaluation.report_id
    
    def test_trigger_fails_for_active_session(self, client, active_session):
        """Should fail for sessions that aren't completed."""
        response = client.post(f"/api/evaluation/{active_session.id}/trigger")
        
        assert response.status_code == 400
        assert "status" in response.json()["detail"].lower()
    
    def test_trigger_fails_for_nonexistent_session(self, client):
        """Should return 404 for non-existent session."""
        fake_id = uuid4()
        response = client.post(f"/api/evaluation/{fake_id}/trigger")
        
        assert response.status_code == 404
    
    def test_trigger_fails_for_other_users_session(self, client_other_user, completed_session):
        """Should return 403 when trying to access another user's session."""
        response = client_other_user.post(f"/api/evaluation/{completed_session.id}/trigger")
        
        assert response.status_code == 403


# ═══════════════════════════════════════════════════════════════════════════════
# Test: GET /evaluation/{session_id}/status
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetEvaluationStatusEndpoint:
    """Tests for get evaluation status endpoint."""
    
    def test_get_status_not_found(self, client, completed_session):
        """Should return 'not_found' status when no evaluation exists."""
        response = client.get(f"/api/evaluation/{completed_session.id}/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "not_found"
        assert data["session_id"] == str(completed_session.id)
    
    def test_get_status_pending(self, client, test_db, completed_session):
        """Should return pending status for new evaluation."""
        # Create pending report
        report = EvaluationReport(
            session_id=completed_session.id,
            report_id=generate_report_id(),
            status="pending",
            progress=0
        )
        test_db.add(report)
        test_db.commit()
        
        response = client.get(f"/api/evaluation/{completed_session.id}/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "pending"
        assert data["progress"] == 0
    
    def test_get_status_processing(self, client, test_db, completed_session):
        """Should return processing status with progress."""
        report = EvaluationReport(
            session_id=completed_session.id,
            report_id=generate_report_id(),
            status="processing",
            progress=50
        )
        test_db.add(report)
        test_db.commit()
        
        response = client.get(f"/api/evaluation/{completed_session.id}/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "processing"
        assert data["progress"] == 50
    
    def test_get_status_completed(self, client, completed_session, existing_evaluation):
        """Should return completed status."""
        response = client.get(f"/api/evaluation/{completed_session.id}/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["progress"] == 100
        assert data["report_id"] == existing_evaluation.report_id
    
    def test_get_status_failed_with_error(self, client, test_db, completed_session):
        """Should return error message for failed evaluations."""
        report = EvaluationReport(
            session_id=completed_session.id,
            report_id=generate_report_id(),
            status="failed",
            error_message="LLM timeout after 30s"
        )
        test_db.add(report)
        test_db.commit()
        
        response = client.get(f"/api/evaluation/{completed_session.id}/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert data["error"] == "LLM timeout after 30s"
    
    def test_get_status_forbidden_for_other_user(self, client_other_user, completed_session):
        """Should return 403 for other user's session."""
        response = client_other_user.get(f"/api/evaluation/{completed_session.id}/status")
        
        assert response.status_code == 403


# ═══════════════════════════════════════════════════════════════════════════════
# Test: GET /evaluation/{session_id}/report
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetEvaluationReportEndpoint:
    """Tests for get evaluation report endpoint."""
    
    def test_get_report_success(self, client, completed_session, existing_evaluation):
        """Should return full evaluation report."""
        response = client.get(f"/api/evaluation/{completed_session.id}/report")
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == str(completed_session.id)
        assert data["report_id"] == existing_evaluation.report_id
        assert data["status"] == "completed"
        assert data["overall_score"] == 85
        assert data["passed"] == True
        assert data["report"] == {"summary": "Good performance"}
        assert data["quick_stats"] == {"total_turns": 4}
    
    def test_get_report_not_found(self, client, completed_session):
        """Should return 404 when no evaluation exists."""
        response = client.get(f"/api/evaluation/{completed_session.id}/report")
        
        assert response.status_code == 404
        assert "No evaluation found" in response.json()["detail"]
    
    def test_get_report_forbidden_for_other_user(self, client_other_user, completed_session, existing_evaluation):
        """Should return 403 for other user's session."""
        response = client_other_user.get(f"/api/evaluation/{completed_session.id}/report")
        
        assert response.status_code == 403


# ═══════════════════════════════════════════════════════════════════════════════
# Test: GET /evaluation/{session_id}/quick-stats
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetQuickStatsEndpoint:
    """Tests for get quick stats endpoint."""
    
    def test_get_quick_stats_computed(self, client, completed_session, session_with_messages):
        """Should compute quick stats when not cached."""
        response = client.get(f"/api/evaluation/{completed_session.id}/quick-stats")
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == str(completed_session.id)
        assert "stats" in data
        assert data["stats"]["total_turns"] == 4
        assert data["stats"]["salesperson_turns"] == 2
        assert data["stats"]["customer_turns"] == 2
        assert data["evaluation_available"] == True
    
    def test_get_quick_stats_cached(self, client, completed_session, existing_evaluation):
        """Should return cached stats from evaluation report."""
        response = client.get(f"/api/evaluation/{completed_session.id}/quick-stats")
        
        assert response.status_code == 200
        data = response.json()
        assert data["stats"] == {"total_turns": 4}
        # evaluation_available should be False since report is completed
        assert data["evaluation_available"] == False
    
    def test_get_quick_stats_with_emotions(self, client, completed_session, session_with_emotions):
        """Should include emotion journey in stats."""
        response = client.get(f"/api/evaluation/{completed_session.id}/quick-stats")
        
        assert response.status_code == 200
        data = response.json()
        assert "emotion_journey" in data["stats"]
    
    def test_get_quick_stats_forbidden_for_other_user(self, client_other_user, completed_session):
        """Should return 403 for other user's session."""
        response = client_other_user.get(f"/api/evaluation/{completed_session.id}/quick-stats")
        
        assert response.status_code == 403
    
    def test_get_quick_stats_not_found_session(self, client):
        """Should return 404 for non-existent session."""
        fake_id = uuid4()
        response = client.get(f"/api/evaluation/{fake_id}/quick-stats")
        
        assert response.status_code == 404


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Full Integration Flow
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvaluationFlow:
    """Tests for the complete evaluation flow."""
    
    def test_full_evaluation_flow(self, client, test_db, completed_session, session_with_messages):
        """Test complete flow: trigger -> check status -> get report."""
        # Step 1: Trigger evaluation
        trigger_response = client.post(f"/api/evaluation/{completed_session.id}/trigger")
        assert trigger_response.status_code == 200
        assert trigger_response.json()["status"] == "started"
        report_id = trigger_response.json()["report_id"]
        
        # Step 2: Check status (should be pending)
        status_response = client.get(f"/api/evaluation/{completed_session.id}/status")
        assert status_response.status_code == 200
        assert status_response.json()["status"] == "pending"
        assert status_response.json()["report_id"] == report_id
        
        # Step 3: Simulate completion (manually update in DB)
        report = test_db.query(EvaluationReport).filter(
            EvaluationReport.session_id == completed_session.id
        ).first()
        report.status = "completed"
        report.progress = 100
        report.overall_score = 90
        report.passed = True
        report.report_json = {"summary": "Excellent sales call!"}
        test_db.commit()
        
        # Step 4: Get final report
        report_response = client.get(f"/api/evaluation/{completed_session.id}/report")
        assert report_response.status_code == 200
        data = report_response.json()
        assert data["status"] == "completed"
        assert data["overall_score"] == 90
        assert data["passed"] == True
