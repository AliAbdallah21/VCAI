# backend/tests/conftest.py
"""
Pytest configuration for backend tests.

This module sets up:
1. SQLite compatibility for PostgreSQL-specific types (JSONB, UUID)
2. Test database fixtures
3. Common test fixtures for models
"""

import pytest
from uuid import uuid4
from sqlalchemy import create_engine, event, JSON
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects import sqlite
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID

# ═══════════════════════════════════════════════════════════════════════════════
# SQLite Compatibility - Handle PostgreSQL types in SQLite
# ═══════════════════════════════════════════════════════════════════════════════

# Compile JSONB as JSON for SQLite
def _compile_jsonb_sqlite(type_, compiler, **kw):
    return "JSON"

# Compile UUID as VARCHAR for SQLite  
def _compile_uuid_sqlite(type_, compiler, **kw):
    return "VARCHAR(36)"

# Register custom type compilers for SQLite
sqlite.base.SQLiteTypeCompiler.visit_JSONB = _compile_jsonb_sqlite
sqlite.base.SQLiteTypeCompiler.visit_UUID = _compile_uuid_sqlite


# ═══════════════════════════════════════════════════════════════════════════════
# Database Engine Setup
# ═══════════════════════════════════════════════════════════════════════════════

from backend.database import Base

# Create test engine with SQLite
TEST_DATABASE_URL = "sqlite:///:memory:"
test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


# ═══════════════════════════════════════════════════════════════════════════════
# Test Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="function")
def db():
    """
    Create a fresh database for each test function.
    
    Creates all tables before the test and drops them after.
    """
    # Create all tables
    Base.metadata.create_all(bind=test_engine)
    
    # Create session
    session = TestingSessionLocal()
    
    try:
        yield session
    finally:
        session.close()
        # Drop all tables after test
        Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def test_user(db):
    """Create a test user."""
    from backend.models import User
    
    user = User(
        id=uuid4(),
        email="test@example.com",
        password_hash="hashed_password",
        full_name="Test User"
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture
def test_persona(db):
    """Create a test persona."""
    from backend.models.persona import Persona
    
    persona = Persona(
        id="test_persona",
        name_ar="عميل تجريبي",
        name_en="Test Customer",
        personality_prompt="You are a test customer.",
        difficulty="medium"
    )
    db.add(persona)
    db.commit()
    return persona


@pytest.fixture
def test_session(db, test_user, test_persona):
    """Create a test training session."""
    from backend.models import Session as TrainingSession
    
    session = TrainingSession(
        id=uuid4(),
        user_id=test_user.id,
        persona_id=test_persona.id,
        status="completed",
        difficulty="medium",
        duration_seconds=300,
        turn_count=6
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@pytest.fixture
def test_messages(db, test_session):
    """Create test messages for a session."""
    from backend.models import Message
    
    messages = [
        Message(
            session_id=test_session.id,
            turn_number=1,
            speaker="customer",
            text="Hi, I'm interested in your product."
        ),
        Message(
            session_id=test_session.id,
            turn_number=2,
            speaker="salesperson",
            text="Hello! Thanks for your interest."
        ),
        Message(
            session_id=test_session.id,
            turn_number=3,
            speaker="customer",
            text="What are the pricing options?"
        ),
        Message(
            session_id=test_session.id,
            turn_number=4,
            speaker="salesperson",
            text="We have Basic at $10/month and Pro at $25/month."
        ),
    ]
    for msg in messages:
        db.add(msg)
    db.commit()
    return messages


@pytest.fixture
def test_emotion_logs(db, test_session):
    """Create test emotion logs for a session."""
    from backend.models import EmotionLog
    
    logs = [
        EmotionLog(
            session_id=test_session.id,
            customer_emotion="curious",
            customer_mood_score=20,
            risk_level="low"
        ),
        EmotionLog(
            session_id=test_session.id,
            customer_emotion="interested",
            customer_mood_score=40,
            risk_level="low"
        ),
    ]
    for log in logs:
        db.add(log)
    db.commit()
    return logs