# backend/tests/api/test_phase4_scoping.py
"""
Phase 4 acceptance tests: tenant scoping of sessions + usage metering +
free-plan persona gating.

These exercise the service layer (usage_service, scope_service, session_service)
directly against the SQLite test DB so they do not depend on the full app /
model preloading. They cover the acceptance criteria from
prompts/phase4_tenant_scoping_usage.md:

  - cross-user / cross-tenant session access -> 404
  - manager reads any session in own company, none in another
  - monthly session cap -> 429, usage increments exactly once per session
  - free-plan persona gating -> 403 + locked flags
  - suspended subscription -> 403
"""

from datetime import date
from uuid import uuid4

import pytest
from fastapi import HTTPException

from backend.models import (
    Company,
    Plan,
    Subscription,
    User,
    Persona,
    Session as TrainingSession,
    UsagePeriod,
)
from backend.plans import PLANS, FREE_PERSONA_IDS, UNLIMITED
from backend.services import session_service
from backend.services.scope_service import get_session_or_403
from backend.services import usage_service
from backend.schemas import SessionCreate


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def seed_plans(db):
    """Mirror backend/plans.py into the plans table the tests need."""
    for name in ("free", "starter", "scale"):
        spec = PLANS[name]
        db.add(Plan(
            name=name,
            display_name=spec["display_name"],
            seat_limit=spec["seat_limit"],
            session_limit_monthly=spec["session_limit_monthly"],
            gaas_enabled=spec["gaas_enabled"],
            price_monthly_usd=spec["price_monthly_usd"],
            price_annual_usd=spec["price_annual_usd"],
        ))
    db.commit()


def _make_company(db, *, plan_name="free", billing_status="active"):
    company = Company(id=uuid4(), name=f"Co-{plan_name}", slug=f"co-{uuid4().hex[:6]}", is_active=True)
    db.add(company)
    db.flush()
    db.add(Subscription(
        id=uuid4(),
        company_id=company.id,
        plan_name=plan_name,
        billing_cycle="monthly",
        billing_status=billing_status,
    ))
    db.commit()
    return company


def _make_user(db, *, company_id=None, role="salesperson", email=None):
    user = User(
        id=uuid4(),
        email=email or f"{uuid4().hex[:8]}@example.com",
        password_hash="x",
        full_name="U",
        company_id=company_id,
        role=role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _make_persona(db, persona_id, difficulty="easy"):
    p = Persona(
        id=persona_id,
        name_ar="x",
        name_en="x",
        personality_prompt="x",
        difficulty=difficulty,
    )
    db.add(p)
    db.commit()
    return p


def _make_session(db, user, persona_id):
    s = TrainingSession(
        id=uuid4(),
        user_id=user.id,
        persona_id=persona_id,
        status="active",
        difficulty="easy",
    )
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Scope: cross-user / cross-tenant session access
# ─────────────────────────────────────────────────────────────────────────────

def test_salesperson_cannot_read_other_users_session(db, seed_plans):
    company = _make_company(db, plan_name="scale")
    persona = _make_persona(db, "first_time_buyer")
    alice = _make_user(db, company_id=company.id)
    bob = _make_user(db, company_id=company.id)
    bobs_session = _make_session(db, bob, persona.id)

    with pytest.raises(HTTPException) as exc:
        get_session_or_403(db, bobs_session.id, alice)
    assert exc.value.status_code == 404


def test_salesperson_cannot_read_other_company_session(db, seed_plans):
    company_a = _make_company(db, plan_name="scale")
    company_b = _make_company(db, plan_name="scale")
    persona = _make_persona(db, "first_time_buyer")
    a_user = _make_user(db, company_id=company_a.id)
    b_user = _make_user(db, company_id=company_b.id)
    b_session = _make_session(db, b_user, persona.id)

    with pytest.raises(HTTPException) as exc:
        get_session_or_403(db, b_session.id, a_user)
    assert exc.value.status_code == 404


def test_manager_reads_any_session_in_own_company(db, seed_plans):
    company = _make_company(db, plan_name="scale")
    persona = _make_persona(db, "first_time_buyer")
    manager = _make_user(db, company_id=company.id, role="manager")
    agent = _make_user(db, company_id=company.id)
    agent_session = _make_session(db, agent, persona.id)

    # Manager can read the agent's session (same company).
    got = get_session_or_403(db, agent_session.id, manager)
    assert got.id == agent_session.id


def test_manager_cannot_read_other_company_session(db, seed_plans):
    company_a = _make_company(db, plan_name="scale")
    company_b = _make_company(db, plan_name="scale")
    persona = _make_persona(db, "first_time_buyer")
    manager_a = _make_user(db, company_id=company_a.id, role="manager")
    agent_b = _make_user(db, company_id=company_b.id)
    b_session = _make_session(db, agent_b, persona.id)

    with pytest.raises(HTTPException) as exc:
        get_session_or_403(db, b_session.id, manager_a)
    assert exc.value.status_code == 404


def test_superadmin_reads_any_session(db, seed_plans):
    company = _make_company(db, plan_name="scale")
    persona = _make_persona(db, "first_time_buyer")
    agent = _make_user(db, company_id=company.id)
    agent_session = _make_session(db, agent, persona.id)
    superadmin = _make_user(db, company_id=None, role="superadmin")

    got = get_session_or_403(db, agent_session.id, superadmin)
    assert got.id == agent_session.id


# ─────────────────────────────────────────────────────────────────────────────
# Usage metering: monthly cap + atomic increment
# ─────────────────────────────────────────────────────────────────────────────

def test_usage_increments_exactly_once_per_session(db, seed_plans):
    company = _make_company(db, plan_name="starter")  # 30/mo
    persona = _make_persona(db, "first_time_buyer")
    user = _make_user(db, company_id=company.id)

    for _ in range(3):
        session_service.create_session(db, user, SessionCreate(persona_id=persona.id, difficulty="easy"))

    assert usage_service.sessions_used_this_period(db, company.id) == 3


def test_monthly_limit_blocks_with_429(db, seed_plans):
    company = _make_company(db, plan_name="free")  # 5/mo
    persona = _make_persona(db, "first_time_buyer")
    user = _make_user(db, company_id=company.id)

    for _ in range(5):
        session_service.create_session(db, user, SessionCreate(persona_id=persona.id, difficulty="easy"))

    with pytest.raises(HTTPException) as exc:
        session_service.create_session(db, user, SessionCreate(persona_id=persona.id, difficulty="easy"))
    assert exc.value.status_code == 429
    # The blocked attempt did NOT increment the counter.
    assert usage_service.sessions_used_this_period(db, company.id) == 5


def test_unlimited_plan_skips_cap_but_records(db, seed_plans):
    company = _make_company(db, plan_name="scale")  # UNLIMITED
    persona = _make_persona(db, "first_time_buyer")
    user = _make_user(db, company_id=company.id)

    assert usage_service.session_limit_for(db, company.id) == UNLIMITED
    for _ in range(7):
        session_service.create_session(db, user, SessionCreate(persona_id=persona.id, difficulty="easy"))
    assert usage_service.sessions_used_this_period(db, company.id) == 7


# ─────────────────────────────────────────────────────────────────────────────
# Billing status gating
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad_status", ["suspended", "past_due", "cancelled"])
def test_inactive_subscription_blocks_with_403(db, seed_plans, bad_status):
    company = _make_company(db, plan_name="scale", billing_status=bad_status)
    persona = _make_persona(db, "first_time_buyer")
    user = _make_user(db, company_id=company.id)

    with pytest.raises(HTTPException) as exc:
        session_service.create_session(db, user, SessionCreate(persona_id=persona.id, difficulty="easy"))
    assert exc.value.status_code == 403


# ─────────────────────────────────────────────────────────────────────────────
# Free-plan persona gating
# ─────────────────────────────────────────────────────────────────────────────

def test_free_plan_blocks_gated_persona_with_403(db, seed_plans):
    company = _make_company(db, plan_name="free")
    gated = _make_persona(db, "tough_negotiator", difficulty="hard")
    user = _make_user(db, company_id=company.id)

    with pytest.raises(HTTPException) as exc:
        session_service.create_session(db, user, SessionCreate(persona_id=gated.id, difficulty="hard"))
    assert exc.value.status_code == 403


def test_free_plan_allows_easy_personas(db, seed_plans):
    company = _make_company(db, plan_name="free")
    persona = _make_persona(db, FREE_PERSONA_IDS[0])
    user = _make_user(db, company_id=company.id)

    s = session_service.create_session(db, user, SessionCreate(persona_id=persona.id, difficulty="easy"))
    assert s.persona_id == FREE_PERSONA_IDS[0]


def test_paid_plan_allows_any_persona(db, seed_plans):
    company = _make_company(db, plan_name="scale")
    gated = _make_persona(db, "tough_negotiator", difficulty="hard")
    user = _make_user(db, company_id=company.id)

    s = session_service.create_session(db, user, SessionCreate(persona_id=gated.id, difficulty="hard"))
    assert s.persona_id == "tough_negotiator"


def test_locked_flags_for_free_plan(db, seed_plans):
    company = _make_company(db, plan_name="free")
    _make_persona(db, FREE_PERSONA_IDS[0], difficulty="easy")
    _make_persona(db, "tough_negotiator", difficulty="hard")

    personas = session_service.get_all_personas_for_company(db, company.id)
    by_id = {p.id: p for p in personas}
    assert by_id[FREE_PERSONA_IDS[0]].locked is False
    assert by_id["tough_negotiator"].locked is True


def test_locked_flags_all_false_for_paid_plan(db, seed_plans):
    company = _make_company(db, plan_name="scale")
    _make_persona(db, FREE_PERSONA_IDS[0], difficulty="easy")
    _make_persona(db, "tough_negotiator", difficulty="hard")

    personas = session_service.get_all_personas_for_company(db, company.id)
    assert all(p.locked is False for p in personas)
