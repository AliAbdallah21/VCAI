# backend/tests/api/test_phase6_admin.py
"""
Phase 6 acceptance tests: super-admin platform views + tenant levers. Exercises
the service layer directly against the SQLite test DB (consistent with Phases
4/5), covering:

  - tenant list + detail reconcile with seeded data
  - platform usage rollups count across tenants
  - suspend flips billing_status to 'suspended' (Phase 4 gate) + is_active False, audited
  - reactivate restores active state
  - global abuse + audit span all tenants
"""

from datetime import date, datetime, timezone
from uuid import uuid4

import pytest
from fastapi import HTTPException

from backend.models import (
    Company, Subscription, Plan, User, UserStats,
    Session as TrainingSession, UsagePeriod, AbuseFlag, AuditLog,
)
from backend.services import analytics_service, admin_service


@pytest.fixture
def seed_plans(db):
    for name, sl, lim in [("free", 1, 5), ("scale", 100, 1_000_000)]:
        db.add(Plan(name=name, display_name=name.title(), seat_limit=sl,
                    session_limit_monthly=lim, gaas_enabled=(name == "scale"),
                    price_monthly_usd=0, price_annual_usd=0))
    db.commit()


def _company(db, name, *, plan="scale", billing="active", active=True):
    c = Company(id=uuid4(), name=name, slug=f"{name.lower()}-{uuid4().hex[:6]}", is_active=active)
    db.add(c)
    db.flush()
    db.add(Subscription(id=uuid4(), company_id=c.id, plan_name=plan,
                        billing_cycle="monthly", billing_status=billing))
    db.commit()
    return c


def _agent(db, company, *, role="salesperson", active=True):
    u = User(id=uuid4(), email=f"{uuid4().hex[:8]}@x.com", password_hash="x",
             full_name="A", company_id=company.id, role=role, is_active=active)
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


def _superadmin(db):
    u = User(id=uuid4(), email=f"super-{uuid4().hex[:6]}@x.com", password_hash="x",
             full_name="Super", company_id=None, role="superadmin")
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


def _session(db, user, overall=70):
    s = TrainingSession(id=uuid4(), user_id=user.id, persona_id="first_time_buyer",
                        status="completed", difficulty="easy", overall_score=overall,
                        started_at=datetime.now(timezone.utc), turn_count=5)
    db.add(s)
    db.commit()
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Tenant list / detail
# ─────────────────────────────────────────────────────────────────────────────

def test_list_tenants_reconciles(db, seed_plans):
    co_a = _company(db, "Alpha", plan="scale")
    co_b = _company(db, "Beta", plan="free")
    _agent(db, co_a); _agent(db, co_a)
    _agent(db, co_b)

    res = admin_service.list_tenants(db)
    assert res["total"] == 2
    by_name = {t["name"]: t for t in res["tenants"]}
    assert by_name["Alpha"]["total_agents"] == 2
    assert by_name["Alpha"]["plan_name"] == "scale"
    assert by_name["Beta"]["seat_limit"] == 1  # free plan seat limit


def test_list_tenants_search_and_plan_filter(db, seed_plans):
    _company(db, "Alpha", plan="scale")
    _company(db, "Beta", plan="free")
    assert admin_service.list_tenants(db, search="alph")["total"] == 1
    assert admin_service.list_tenants(db, plan="free")["total"] == 1


def test_tenant_detail_full(db, seed_plans):
    co = _company(db, "Alpha")
    _agent(db, co, role="manager")
    _agent(db, co)
    db.add(UsagePeriod(id=uuid4(), company_id=co.id, period_start=date.today().replace(day=1), sessions_used=4))
    db.add(AbuseFlag(id=uuid4(), company_id=co.id, reason="rapid_fire", severity="medium", status="open"))
    db.commit()

    detail = admin_service.tenant_detail(db, co.id)
    assert detail["company"]["name"] == "Alpha"
    assert detail["subscription"]["plan_name"] == "scale"
    assert len(detail["agents"]) == 2
    assert len(detail["usage_history"]) == 1
    assert len(detail["open_abuse_flags"]) == 1


def test_tenant_detail_missing_404(db, seed_plans):
    with pytest.raises(HTTPException) as exc:
        admin_service.tenant_detail(db, uuid4())
    assert exc.value.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# Platform usage
# ─────────────────────────────────────────────────────────────────────────────

def test_platform_usage_counts_across_tenants(db, seed_plans):
    co_a = _company(db, "Alpha")
    co_b = _company(db, "Beta")
    a1 = _agent(db, co_a)
    b1 = _agent(db, co_b)
    _session(db, a1); _session(db, a1); _session(db, b1)

    res = analytics_service.platform_usage(db)
    assert res["total_companies"] == 2
    assert res["active_companies"] == 2
    assert res["active_subscriptions"] == 2
    assert res["total_sessions"] == 3
    assert res["total_active_agents"] == 2
    # top tenant is Alpha with 2 sessions.
    assert res["top_tenants"][0]["name"] == "Alpha"
    assert res["top_tenants"][0]["sessions"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# Suspend / reactivate + audit (ties into Phase 4 gate)
# ─────────────────────────────────────────────────────────────────────────────

def test_suspend_flips_status_and_audits(db, seed_plans):
    co = _company(db, "Alpha")
    admin = _superadmin(db)

    company = admin_service.set_tenant_status(db, company_id=co.id, actor=admin, suspend=True)
    assert company.is_active is False
    sub = db.query(Subscription).filter(Subscription.company_id == co.id).first()
    assert sub.billing_status == "suspended"  # Phase 4 gate blocks new sessions

    audit = db.query(AuditLog).filter(AuditLog.action == "tenant.suspended").first()
    assert audit is not None and audit.company_id == co.id


def test_suspend_then_phase4_gate_blocks(db, seed_plans):
    """After suspend, assert_can_create_session must raise 403 (inactive)."""
    from backend.services.usage_service import assert_can_create_session
    co = _company(db, "Alpha")
    admin = _superadmin(db)
    admin_service.set_tenant_status(db, company_id=co.id, actor=admin, suspend=True)

    with pytest.raises(HTTPException) as exc:
        assert_can_create_session(db, co.id)
    assert exc.value.status_code == 403


def test_reactivate_restores(db, seed_plans):
    co = _company(db, "Alpha", billing="suspended", active=False)
    admin = _superadmin(db)
    company = admin_service.set_tenant_status(db, company_id=co.id, actor=admin, suspend=False)
    assert company.is_active is True
    sub = db.query(Subscription).filter(Subscription.company_id == co.id).first()
    assert sub.billing_status == "active"


# ─────────────────────────────────────────────────────────────────────────────
# Global abuse + audit span tenants
# ─────────────────────────────────────────────────────────────────────────────

def test_global_abuse_spans_tenants(db, seed_plans):
    co_a = _company(db, "Alpha")
    co_b = _company(db, "Beta")
    db.add(AbuseFlag(id=uuid4(), company_id=co_a.id, reason="rapid_fire", severity="high", status="open"))
    db.add(AbuseFlag(id=uuid4(), company_id=co_b.id, reason="empty_sessions", severity="low", status="reviewed"))
    db.commit()

    assert len(admin_service.global_abuse(db)) == 2
    assert len(admin_service.global_abuse(db, severity="high")) == 1
    assert len(admin_service.global_abuse(db, status_filter="open")) == 1


def test_global_audit_filterable(db, seed_plans):
    co = _company(db, "Alpha")
    admin = _superadmin(db)
    admin_service.set_tenant_status(db, company_id=co.id, actor=admin, suspend=True)
    admin_service.set_tenant_status(db, company_id=co.id, actor=admin, suspend=False)

    all_rows = admin_service.global_audit(db)
    assert len(all_rows) >= 2
    only_suspend = admin_service.global_audit(db, action="tenant.suspended")
    assert all(r.action == "tenant.suspended" for r in only_suspend)
    by_company = admin_service.global_audit(db, company_id=co.id)
    assert all(r.company_id == co.id for r in by_company)
