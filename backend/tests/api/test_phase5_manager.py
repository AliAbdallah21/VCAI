# backend/tests/api/test_phase5_manager.py
"""
Phase 5 acceptance tests: manager dashboard analytics + abuse review, scoped by
company. Exercises the service layer directly against the SQLite test DB
(consistent with the Phase 4 tests), covering:

  - manager sees only their company's agents/sessions/flags
  - analytics numbers reconcile with raw session rows
  - agent progress shows real per-skill + trend data
  - abuse list + resolve persists and writes an audit row
  - cross-company agent/flag access -> 404
"""

from datetime import datetime, timezone, timedelta
from uuid import uuid4

import pytest
from fastapi import HTTPException

from backend.models import (
    Company,
    Plan,
    Subscription,
    User,
    UserStats,
    Persona,
    Session as TrainingSession,
    EmotionLog,
    AbuseFlag,
    AuditLog,
)
from backend.services import analytics_service, abuse_service


# ─────────────────────────────────────────────────────────────────────────────
# Builders
# ─────────────────────────────────────────────────────────────────────────────

def _company(db, name="Co"):
    c = Company(id=uuid4(), name=name, slug=f"{name.lower()}-{uuid4().hex[:6]}", is_active=True)
    db.add(c)
    db.flush()
    db.add(Subscription(id=uuid4(), company_id=c.id, plan_name="scale", billing_cycle="monthly", billing_status="active"))
    db.commit()
    return c


@pytest.fixture
def seed_scale_plan(db):
    db.add(Plan(name="scale", display_name="Scale", seat_limit=100, session_limit_monthly=1_000_000,
                gaas_enabled=True, price_monthly_usd=299, price_annual_usd=2990))
    db.commit()


def _agent(db, company, *, role="salesperson", active=True, stats=None):
    u = User(id=uuid4(), email=f"{uuid4().hex[:8]}@x.com", password_hash="x",
             full_name="Agent X", company_id=company.id, role=role, is_active=active)
    db.add(u)
    db.flush()
    if stats is not None:
        db.add(UserStats(user_id=u.id, **stats))
    db.commit()
    db.refresh(u)
    return u


def _persona(db, pid="first_time_buyer", difficulty="easy"):
    p = db.query(Persona).filter(Persona.id == pid).first()
    if p is None:
        p = Persona(id=pid, name_ar="ع", name_en="P", personality_prompt="x", difficulty=difficulty)
        db.add(p)
        db.commit()
    return p


def _session(db, user, persona_id, *, overall=None, skills=None, status="completed", started=None):
    s = TrainingSession(
        id=uuid4(),
        user_id=user.id,
        persona_id=persona_id,
        status=status,
        difficulty="easy",
        overall_score=overall,
        started_at=started or datetime.now(timezone.utc),
        turn_count=4,
    )
    for k, v in (skills or {}).items():
        setattr(s, f"{k}_score", v)
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Roster scoping
# ─────────────────────────────────────────────────────────────────────────────

def test_list_agents_scoped_to_company(db, seed_scale_plan):
    co_a = _company(db, "Aco")
    co_b = _company(db, "Bco")
    _agent(db, co_a, stats={"total_sessions": 3, "completed_sessions": 2, "avg_overall_score": 70.0})
    _agent(db, co_a, stats={"total_sessions": 1})
    _agent(db, co_b)  # other company
    _agent(db, co_a, role="manager")  # managers excluded from agent list

    res = analytics_service.list_agents(db, co_a.id)
    assert len(res["agents"]) == 2  # two salespersons in A, manager excluded
    emails_b = {a["email"] for a in analytics_service.list_agents(db, co_b.id)["agents"]}
    assert all(a["email"] not in emails_b for a in res["agents"])


# ─────────────────────────────────────────────────────────────────────────────
# Analytics reconcile with raw rows
# ─────────────────────────────────────────────────────────────────────────────

def test_analytics_reconcile_with_raw_sessions(db, seed_scale_plan):
    co = _company(db)
    a1 = _agent(db, co)
    a2 = _agent(db, co)
    _persona(db, "first_time_buyer")

    # 4 sessions with known overall scores: 40, 60, 80, 100 -> avg 70.
    _session(db, a1, "first_time_buyer", overall=40)
    _session(db, a1, "first_time_buyer", overall=60)
    _session(db, a2, "first_time_buyer", overall=80)
    _session(db, a2, "first_time_buyer", overall=100)

    res = analytics_service.company_analytics(db, co.id)
    assert res["total_sessions"] == 4
    assert res["active_agents"] == 2
    assert res["avg_score"] == 70.0
    # Distribution buckets sum to the scored-session count.
    assert sum(b["count"] for b in res["score_distribution"]) == 4


def test_weakest_skill_is_lowest_average(db, seed_scale_plan):
    co = _company(db)
    a = _agent(db, co)
    _persona(db, "first_time_buyer")
    _session(db, a, "first_time_buyer", overall=70, skills={
        "communication": 90, "product_knowledge": 80,
        "objection_handling": 30, "rapport": 75, "closing": 60,
    })
    res = analytics_service.company_analytics(db, co.id)
    assert res["weakest_skill"]["skill"] == "objection_handling"
    assert res["weakest_skill"]["average"] == 30.0


def test_empty_company_analytics_safe(db, seed_scale_plan):
    co = _company(db)
    res = analytics_service.company_analytics(db, co.id)
    assert res["total_sessions"] == 0
    assert res["avg_score"] is None
    assert res["weakest_skill"] is None


# ─────────────────────────────────────────────────────────────────────────────
# Agent progress
# ─────────────────────────────────────────────────────────────────────────────

def test_agent_progress_real_data(db, seed_scale_plan):
    co = _company(db)
    agent = _agent(db, co)
    _persona(db, "first_time_buyer")
    t0 = datetime.now(timezone.utc) - timedelta(days=2)
    _session(db, agent, "first_time_buyer", overall=50, skills={"closing": 50}, started=t0)
    _session(db, agent, "first_time_buyer", overall=80, skills={"closing": 80},
             started=datetime.now(timezone.utc))

    res = analytics_service.agent_progress(db, co.id, agent.id)
    assert res["skill_averages"]["closing"] == 65.0
    assert [p["overall_score"] for p in res["score_trend"]] == [50, 80]  # oldest first
    assert len(res["sessions"]) == 2
    assert res["skill_profiles"] == []  # Phase 8 fills this


def test_agent_progress_cross_company_404(db, seed_scale_plan):
    co_a = _company(db, "Aco")
    co_b = _company(db, "Bco")
    agent_b = _agent(db, co_b)
    with pytest.raises(HTTPException) as exc:
        analytics_service.agent_progress(db, co_a.id, agent_b.id)
    assert exc.value.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# Emotion trends
# ─────────────────────────────────────────────────────────────────────────────

def test_emotion_trends_high_risk_share(db, seed_scale_plan):
    co = _company(db)
    agent = _agent(db, co)
    _persona(db, "first_time_buyer")
    s1 = _session(db, agent, "first_time_buyer", overall=60)
    s2 = _session(db, agent, "first_time_buyer", overall=60)

    now = datetime.now(timezone.utc)
    # s1 ends angry (high risk), s2 ends happy.
    db.add(EmotionLog(session_id=s1.id, customer_emotion="neutral", customer_mood_score=0, risk_level="low", created_at=now - timedelta(minutes=2)))
    db.add(EmotionLog(session_id=s1.id, customer_emotion="angry", customer_mood_score=-50, risk_level="high", created_at=now))
    db.add(EmotionLog(session_id=s2.id, customer_emotion="happy", customer_mood_score=40, risk_level="low", created_at=now))
    db.commit()

    res = analytics_service.emotion_trends(db, co.id)
    assert res["total_sessions_with_emotion"] == 2
    assert res["high_risk_session_share"] == 0.5  # one of two ended high-risk
    assert any(e["emotion"] == "angry" for e in res["emotion_distribution"])


# ─────────────────────────────────────────────────────────────────────────────
# Abuse review
# ─────────────────────────────────────────────────────────────────────────────

def test_list_flags_scoped_and_filtered(db, seed_scale_plan):
    co_a = _company(db, "Aco")
    co_b = _company(db, "Bco")
    db.add(AbuseFlag(id=uuid4(), company_id=co_a.id, reason="rapid_fire", severity="medium", status="open"))
    db.add(AbuseFlag(id=uuid4(), company_id=co_a.id, reason="empty_sessions", severity="low", status="reviewed"))
    db.add(AbuseFlag(id=uuid4(), company_id=co_b.id, reason="seat_sharing", severity="high", status="open"))
    db.commit()

    all_a = abuse_service.list_flags(db, co_a.id)
    assert len(all_a) == 2  # company B's flag not visible
    open_a = abuse_service.list_flags(db, co_a.id, status_filter="open")
    assert len(open_a) == 1 and open_a[0].reason == "rapid_fire"


def test_resolve_flag_persists_and_audits(db, seed_scale_plan):
    co = _company(db)
    manager = _agent(db, co, role="manager")
    flag = AbuseFlag(id=uuid4(), company_id=co.id, reason="rapid_fire", severity="medium", status="open")
    db.add(flag)
    db.commit()

    resolved = abuse_service.resolve_flag(
        db, company_id=co.id, actor=manager, flag_id=flag.id, new_status="dismissed", note="not abuse"
    )
    assert resolved.status == "dismissed"
    assert resolved.resolved_by == manager.id
    assert resolved.resolved_at is not None

    audit = db.query(AuditLog).filter(AuditLog.action == "abuse.resolved").first()
    assert audit is not None
    assert audit.detail["status"] == "dismissed"


def test_resolve_flag_cross_company_404(db, seed_scale_plan):
    co_a = _company(db, "Aco")
    co_b = _company(db, "Bco")
    manager_a = _agent(db, co_a, role="manager")
    flag_b = AbuseFlag(id=uuid4(), company_id=co_b.id, reason="rapid_fire", severity="medium", status="open")
    db.add(flag_b)
    db.commit()

    with pytest.raises(HTTPException) as exc:
        abuse_service.resolve_flag(db, company_id=co_a.id, actor=manager_a, flag_id=flag_b.id, new_status="reviewed")
    assert exc.value.status_code == 404


def test_resolve_flag_rejects_bad_status(db, seed_scale_plan):
    co = _company(db)
    manager = _agent(db, co, role="manager")
    flag = AbuseFlag(id=uuid4(), company_id=co.id, reason="rapid_fire", severity="medium", status="open")
    db.add(flag)
    db.commit()
    with pytest.raises(HTTPException) as exc:
        abuse_service.resolve_flag(db, company_id=co.id, actor=manager, flag_id=flag.id, new_status="open")
    assert exc.value.status_code == 400
