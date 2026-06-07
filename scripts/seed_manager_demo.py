# scripts/seed_manager_demo.py
"""
Seed a demo company for the Phase 5 manager dashboard: one manager + three
agents, each with several SCORED sessions (varying per-skill scores), a few
emotion logs, and a couple of abuse flags. Lets you log in as the manager and
verify every dashboard view against known numbers.

Idempotent: re-running removes the previously seeded demo company (by slug) and
re-creates it, so the numbers stay deterministic.

PREREQUISITES (run first, in order):
    <venv-python> -c "from alembic.config import main; main(argv=['upgrade','head'])"
    <venv-python> scripts/seed_plans.py
    <venv-python> scripts/seed_personas.py

Usage:
    <venv-python> scripts/seed_manager_demo.py

Then log in as:  manager@demo.test  /  demo1234
"""
from __future__ import annotations

import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import inspect as sa_inspect

from backend.database import SessionLocal
from backend.models import (
    Company, Subscription, Plan, User, UserStats, Persona,
    Session as TrainingSession, EmotionLog, AbuseFlag,
)
from backend.services.auth_service import get_password_hash


def _insert_filtered(db, model, values: dict):
    """
    Insert a row using only the columns that actually exist on the live table.
    Keeps this demo seed runnable on databases whose tables predate later
    additive columns (e.g. sessions.scenario / sessions.training_focus). Returns
    the inserted primary-key id.
    """
    live_cols = {c["name"] for c in sa_inspect(db.bind).get_columns(model.__tablename__)}
    cols = {k: v for k, v in values.items() if k in live_cols}
    result = db.execute(model.__table__.insert().values(**cols))
    return result.inserted_primary_key[0] if result.inserted_primary_key else values.get("id")

DEMO_SLUG = "manager-demo"
DEMO_PLAN = "growth"  # 150 sessions/mo, GaaS on
MANAGER_EMAIL = "manager@demo.test"
MANAGER_PW = "demo1234"
AGENTS = [
    ("Sara Hassan", "sara@demo.test"),
    ("Omar Farouk", "omar@demo.test"),
    ("Layla Adel", "layla@demo.test"),
]
SKILLS = ["communication", "product_knowledge", "objection_handling", "rapport", "closing"]
EMOTIONS = ["neutral", "interested", "happy", "frustrated", "angry", "curious"]


def _wipe_existing(db, company):
    """Delete the previous demo company; CASCADE clears subscription/users/sessions."""
    db.delete(company)
    db.commit()


def seed(db) -> None:
    if db.query(Plan).filter(Plan.name == DEMO_PLAN).first() is None:
        print(f"ERROR: plan '{DEMO_PLAN}' not found. Run seed_plans.py first.")
        return

    # Select only id + difficulty (avoids loading optional columns that may not
    # exist on older live `personas` tables). Returns (id, difficulty) rows.
    persona_rows = (
        db.query(Persona.id, Persona.difficulty)
        .filter(Persona.is_active == True)  # noqa: E712
        .limit(4)
        .all()
    )
    if not persona_rows:
        print("ERROR: no personas found. Run seed_personas.py first.")
        return

    existing = db.query(Company).filter(Company.slug == DEMO_SLUG).first()
    if existing:
        print(f"  [RESET] removing previous demo company {existing.id}")
        _wipe_existing(db, existing)

    random.seed(42)  # deterministic demo numbers

    company = Company(name="Manager Demo Co", slug=DEMO_SLUG, is_active=True)
    db.add(company)
    db.flush()
    db.add(Subscription(company_id=company.id, plan_name=DEMO_PLAN,
                        billing_cycle="monthly", billing_status="active"))

    manager = User(email=MANAGER_EMAIL, password_hash=get_password_hash(MANAGER_PW),
                   full_name="Demo Manager", company_id=company.id, role="manager")
    db.add(manager)
    db.flush()
    db.add(UserStats(user_id=manager.id))

    now = datetime.now(timezone.utc)
    total_sessions = 0

    for idx, (name, email) in enumerate(AGENTS):
        agent = User(email=email, password_hash=get_password_hash(MANAGER_PW),
                     full_name=name, company_id=company.id, role="salesperson")
        db.add(agent)
        db.flush()

        n = 4 + idx  # 4, 5, 6 sessions per agent
        overalls = []
        for k in range(n):
            persona_id, persona_difficulty = persona_rows[k % len(persona_rows)]
            overall = random.randint(45, 95)
            overalls.append(overall)
            started = now - timedelta(days=(n - k) * 2)
            session_id = uuid4()
            _insert_filtered(db, TrainingSession, dict(
                id=session_id, user_id=agent.id, persona_id=persona_id, status="completed",
                difficulty=persona_difficulty, overall_score=overall,
                started_at=started, ended_at=started + timedelta(minutes=8),
                duration_seconds=480, turn_count=random.randint(6, 14),
                communication_score=random.randint(40, 95),
                product_knowledge_score=random.randint(40, 95),
                objection_handling_score=random.randint(30, 90),
                rapport_score=random.randint(45, 95),
                closing_score=random.randint(35, 90),
            ))
            total_sessions += 1

            # A couple of emotion logs per session; last one drives risk share.
            for j in range(2):
                emo = random.choice(EMOTIONS)
                _insert_filtered(db, EmotionLog, dict(
                    id=uuid4(), session_id=session_id, customer_emotion=emo,
                    customer_mood_score=random.randint(-60, 60),
                    risk_level="high" if emo in ("angry", "frustrated") else "low",
                    created_at=started + timedelta(minutes=j * 3),
                ))

        avg = round(sum(overalls) / len(overalls), 1)
        db.add(UserStats(
            user_id=agent.id, total_sessions=n, completed_sessions=n,
            avg_overall_score=avg, current_streak=idx + 1,
            last_session_date=now - timedelta(days=2),
        ))

    # Two abuse flags (Phase 7 normally writes these; seed for the queue UI).
    first_agent = db.query(User).filter(User.company_id == company.id, User.role == "salesperson").first()
    db.add(AbuseFlag(company_id=company.id, user_id=first_agent.id, reason="rapid_fire",
                     severity="medium", status="open",
                     detail={"sessions_in_window": 9, "window_minutes": 10}))
    db.add(AbuseFlag(company_id=company.id, user_id=first_agent.id, reason="empty_sessions",
                     severity="low", status="open",
                     detail={"empty_count": 4}))

    db.commit()
    print(f"\nManager demo ready: company_id={company.id}")
    print(f"  Manager login: {MANAGER_EMAIL} / {MANAGER_PW}")
    print(f"  Agents: {len(AGENTS)}, sessions: {total_sessions}, abuse flags: 2")
    print("  Log in as the manager and open /dashboard to verify each view.")


if __name__ == "__main__":
    print("Seeding manager demo...")
    db = SessionLocal()
    try:
        seed(db)
    finally:
        db.close()
