# backend/services/company_service.py
"""
Company + subscription provisioning service. Creates a tenant (company), its
manager user, and its (mocked) subscription in a single transaction on onboarding.
Billing is mocked: no payment is captured. See 00_ARCHITECTURE.md sections 4, 5.
"""

import re
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from backend.config import get_settings
from backend.models import Company, Subscription, User, UserStats
from backend.plans import get_plan
from backend.services.auth_service import get_password_hash, create_access_token
from backend.services.audit_service import record_audit

settings = get_settings()


def _slugify(name: str) -> str:
    """Make a URL-safe slug from a company name."""
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return base or "company"


def _unique_slug(db: Session, name: str) -> str:
    """Return a slug not already used by another company."""
    base = _slugify(name)
    slug = base
    while db.query(Company).filter(Company.slug == slug).first() is not None:
        slug = f"{base}-{uuid.uuid4().hex[:6]}"
    return slug


def provision_is_allowed(plan_name: str) -> dict:
    """Validate the plan name, returning its definition. 400 on bad plan."""
    try:
        return get_plan(plan_name)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


def create_company_with_manager(
    db: Session,
    *,
    company_name: str,
    plan_name: str,
    billing_cycle: str,
    manager_name: str,
    manager_email: str,
    password: str,
) -> dict:
    """
    Provision a new tenant: company + manager user + subscription, in one
    transaction. Returns {company, manager_user, subscription, token}.

    This is the mocked-checkout completion - no payment is captured.
    """
    plan = provision_is_allowed(plan_name)

    if billing_cycle not in ("monthly", "annual"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="billing_cycle must be 'monthly' or 'annual'")

    # Email must be globally unique (users.email is unique).
    if db.query(User).filter(User.email == manager_email).first() is not None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Email already registered")

    is_free = plan["name"] == "free"

    try:
        company = Company(name=company_name, slug=_unique_slug(db, company_name), is_active=True)
        db.add(company)
        db.flush()  # assigns company.id

        subscription = Subscription(
            company_id=company.id,
            plan_name=plan["name"],
            billing_cycle=billing_cycle,
            billing_status="active" if is_free else "trial",
            trial_ends_at=None if is_free else datetime.now(timezone.utc) + timedelta(days=settings.trial_days),
        )
        db.add(subscription)

        manager = User(
            email=manager_email,
            password_hash=get_password_hash(password),
            full_name=manager_name,
            company=company_name,
            company_id=company.id,
            role="manager",
        )
        db.add(manager)
        db.flush()  # assigns manager.id

        db.add(UserStats(user_id=manager.id))

        record_audit(
            db,
            action="company.created",
            actor=manager,
            company_id=company.id,
            target_type="company",
            target_id=company.id,
            detail={"plan_name": plan["name"], "billing_cycle": billing_cycle},
        )

        db.commit()
    except HTTPException:
        db.rollback()
        raise
    except Exception:
        db.rollback()
        raise

    db.refresh(company)
    db.refresh(manager)
    db.refresh(subscription)

    token = create_access_token(str(manager.id))

    return {
        "company": company,
        "manager_user": manager,
        "subscription": subscription,
        "token": token,
    }
