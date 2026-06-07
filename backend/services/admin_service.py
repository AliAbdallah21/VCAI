# backend/services/admin_service.py
"""
Super-admin (platform owner) service (Phase 6). Tenant listing/detail and the
manual suspend/reactivate levers for the mocked-billing era. Platform-wide (no
company scope); the router restricts every entry point to superadmin. All
rollups use SQL aggregates. See 00_ARCHITECTURE.md sections 3, 6, 10.
"""

from datetime import date
from typing import Optional
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import func
from sqlalchemy.orm import Session as DbSession

from backend.models import (
    Company,
    Subscription,
    Plan,
    User,
    UsagePeriod,
    AbuseFlag,
    AuditLog,
)
from backend.plans import get_plan
from backend.services.audit_service import record_audit


def _current_period_start() -> date:
    return date.today().replace(day=1)


def _seats_used(db: DbSession, company_id) -> int:
    return (
        db.query(func.count(User.id))
        .filter(
            User.company_id == company_id,
            User.role == "salesperson",
            User.is_active == True,  # noqa: E712
        )
        .scalar()
    ) or 0


def _total_agents(db: DbSession, company_id) -> int:
    return (
        db.query(func.count(User.id))
        .filter(User.company_id == company_id, User.role == "salesperson")
        .scalar()
    ) or 0


def _sessions_this_period(db: DbSession, company_id) -> int:
    row = (
        db.query(UsagePeriod.sessions_used)
        .filter(
            UsagePeriod.company_id == company_id,
            UsagePeriod.period_start == _current_period_start(),
        )
        .first()
    )
    return int(row[0]) if row and row[0] is not None else 0


def list_tenants(
    db: DbSession,
    *,
    search: Optional[str] = None,
    plan: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """All companies with plan/billing/usage summary. Supports search + plan filter."""
    q = (
        db.query(Company, Subscription)
        .outerjoin(Subscription, Subscription.company_id == Company.id)
    )
    if search:
        like = f"%{search.lower()}%"
        q = q.filter(func.lower(Company.name).like(like))
    if plan:
        q = q.filter(Subscription.plan_name == plan)

    total = q.count()
    rows = q.order_by(Company.created_at.desc()).offset(offset).limit(limit).all()

    tenants = []
    for company, sub in rows:
        plan_def = get_plan(sub.plan_name) if sub else None
        tenants.append({
            "company_id": company.id,
            "name": company.name,
            "slug": company.slug,
            "is_active": company.is_active,
            "created_at": company.created_at,
            "plan_name": sub.plan_name if sub else None,
            "billing_status": sub.billing_status if sub else None,
            "seats_used": _seats_used(db, company.id),
            "seat_limit": plan_def["seat_limit"] if plan_def else None,
            "sessions_this_period": _sessions_this_period(db, company.id),
            "total_agents": _total_agents(db, company.id),
        })

    return {"tenants": tenants, "total": total}


def tenant_detail(db: DbSession, company_id: UUID) -> dict:
    """One company's full detail: subscription, agents, usage history, flags, audit."""
    company = db.query(Company).filter(Company.id == company_id).first()
    if company is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Company not found")

    sub = db.query(Subscription).filter(Subscription.company_id == company_id).first()
    plan_def = get_plan(sub.plan_name) if sub else None

    agents = (
        db.query(User)
        .filter(User.company_id == company_id)
        .order_by(User.created_at.asc())
        .all()
    )

    usage_history = (
        db.query(UsagePeriod)
        .filter(UsagePeriod.company_id == company_id)
        .order_by(UsagePeriod.period_start.desc())
        .limit(6)
        .all()
    )

    open_flags = (
        db.query(AbuseFlag)
        .filter(AbuseFlag.company_id == company_id, AbuseFlag.status == "open")
        .order_by(AbuseFlag.created_at.desc())
        .all()
    )

    recent_audit = (
        db.query(AuditLog)
        .filter(AuditLog.company_id == company_id)
        .order_by(AuditLog.created_at.desc())
        .limit(20)
        .all()
    )

    return {
        "company": {
            "company_id": company.id,
            "name": company.name,
            "slug": company.slug,
            "is_active": company.is_active,
            "created_at": company.created_at,
        },
        "subscription": {
            "plan_name": sub.plan_name,
            "display_name": plan_def["display_name"] if plan_def else None,
            "billing_cycle": sub.billing_cycle,
            "billing_status": sub.billing_status,
            "trial_ends_at": sub.trial_ends_at,
            "seat_limit": plan_def["seat_limit"] if plan_def else None,
            "session_limit_monthly": plan_def["session_limit_monthly"] if plan_def else None,
        } if sub else None,
        "agents": [
            {
                "user_id": u.id,
                "email": u.email,
                "full_name": u.full_name,
                "role": u.role,
                "is_active": u.is_active,
                "created_at": u.created_at,
            }
            for u in agents
        ],
        "usage_history": [
            {
                "period_start": up.period_start,
                "sessions_used": up.sessions_used or 0,
                "seats_peak": up.seats_peak or 0,
            }
            for up in usage_history
        ],
        "open_abuse_flags": [
            {
                "id": f.id,
                "user_id": f.user_id,
                "reason": f.reason,
                "severity": f.severity,
                "status": f.status,
                "created_at": f.created_at,
            }
            for f in open_flags
        ],
        "recent_audit": [
            {
                "id": a.id,
                "action": a.action,
                "actor_role": a.actor_role,
                "target_type": a.target_type,
                "target_id": a.target_id,
                "created_at": a.created_at,
            }
            for a in recent_audit
        ],
    }


def set_tenant_status(
    db: DbSession, *, company_id: UUID, actor: User, suspend: bool
) -> Company:
    """
    Suspend or reactivate a tenant. Suspending flips the subscription
    billing_status to 'suspended' (which the Phase 4 gate blocks) and marks the
    company inactive; reactivating restores 'active' + active company. Audited.
    """
    company = db.query(Company).filter(Company.id == company_id).first()
    if company is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Company not found")

    sub = db.query(Subscription).filter(Subscription.company_id == company_id).first()

    if suspend:
        company.is_active = False
        if sub is not None:
            sub.billing_status = "suspended"
        action = "tenant.suspended"
    else:
        company.is_active = True
        if sub is not None:
            sub.billing_status = "active"
        action = "tenant.reactivated"

    record_audit(
        db,
        action=action,
        actor=actor,
        company_id=company_id,
        target_type="company",
        target_id=company_id,
        detail={"suspended": suspend},
    )
    db.commit()
    db.refresh(company)
    return company


def global_abuse(
    db: DbSession,
    *,
    severity: Optional[str] = None,
    status_filter: Optional[str] = None,
    limit: int = 100,
) -> list:
    """Global abuse queue across all tenants, newest first."""
    q = db.query(AbuseFlag)
    if severity:
        q = q.filter(AbuseFlag.severity == severity)
    if status_filter:
        q = q.filter(AbuseFlag.status == status_filter)
    return q.order_by(AbuseFlag.created_at.desc()).limit(limit).all()


def global_audit(
    db: DbSession,
    *,
    company_id: Optional[UUID] = None,
    action: Optional[str] = None,
    limit: int = 100,
) -> list:
    """Global audit log, filterable by company/action, newest first."""
    q = db.query(AuditLog)
    if company_id:
        q = q.filter(AuditLog.company_id == company_id)
    if action:
        q = q.filter(AuditLog.action == action)
    return q.order_by(AuditLog.created_at.desc()).limit(limit).all()
