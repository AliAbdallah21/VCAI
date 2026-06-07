# backend/services/usage_service.py
"""
Usage metering service - monthly session quota tracking and enforcement.

One usage_periods row per company per calendar month. The increment is atomic
(an UPDATE ... SET sessions_used = sessions_used + 1 issued at the database, with
the period row created first if missing) so concurrent session creations cannot
lose updates. See 00_ARCHITECTURE.md section 5.

Callers (session creation) must call:
  1. assert_can_create_session(db, company_id)   -- before creating the session
  2. record_session_usage(db, company_id)        -- inside the same transaction
The caller owns the commit; record_session_usage flushes but does not commit, so
the usage increment and the session insert commit (or roll back) together.
"""

from datetime import date

from fastapi import HTTPException, status
from sqlalchemy import update
from sqlalchemy.orm import Session

from backend.models import Plan, Subscription, UsagePeriod
from backend.plans import UNLIMITED

# Billing states that block new session creation.
_BLOCKED_BILLING_STATES = {"suspended", "past_due", "cancelled"}


def current_period_start() -> date:
    """First day of the current calendar month."""
    today = date.today()
    return today.replace(day=1)


def get_or_create_period(db: Session, company_id) -> UsagePeriod:
    """
    Return the usage_periods row for this company + current month, creating it
    if it does not exist. Flushes (so the row is queryable in this txn) but does
    not commit - it joins the caller's transaction.
    """
    period_start = current_period_start()
    period = (
        db.query(UsagePeriod)
        .filter(
            UsagePeriod.company_id == company_id,
            UsagePeriod.period_start == period_start,
        )
        .first()
    )
    if period is None:
        period = UsagePeriod(
            company_id=company_id,
            period_start=period_start,
            sessions_used=0,
            seats_peak=0,
        )
        db.add(period)
        try:
            db.flush()
        except Exception:
            # A concurrent request inserted the same (company_id, period_start)
            # first (unique constraint uq_usage_company_period). Roll back the
            # failed insert and re-read the winning row.
            db.rollback()
            period = (
                db.query(UsagePeriod)
                .filter(
                    UsagePeriod.company_id == company_id,
                    UsagePeriod.period_start == period_start,
                )
                .first()
            )
    return period


def sessions_used_this_period(db: Session, company_id) -> int:
    """Sessions already created by this company in the current month."""
    period = (
        db.query(UsagePeriod)
        .filter(
            UsagePeriod.company_id == company_id,
            UsagePeriod.period_start == current_period_start(),
        )
        .first()
    )
    return int(period.sessions_used or 0) if period else 0


def session_limit_for(db: Session, company_id) -> int:
    """
    Resolve the company's monthly session limit by JOINing subscription -> plan.
    UNLIMITED (1_000_000) means uncapped.
    """
    row = (
        db.query(Plan.session_limit_monthly)
        .join(Subscription, Subscription.plan_name == Plan.name)
        .filter(Subscription.company_id == company_id)
        .first()
    )
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No subscription for company",
        )
    return int(row[0])


def record_session_usage(db: Session, company_id) -> None:
    """
    Atomically increment sessions_used for the current period.

    Call this INSIDE the session-creation transaction (before the caller's
    commit). The increment is performed as a single SQL UPDATE so concurrent
    creations cannot lose updates - the row is locked by the database for the
    duration of the statement.
    """
    period = get_or_create_period(db, company_id)
    db.execute(
        update(UsagePeriod)
        .where(UsagePeriod.id == period.id)
        .values(sessions_used=UsagePeriod.sessions_used + 1)
    )


def assert_can_create_session(db: Session, company_id) -> None:
    """
    Gate session creation on billing status and the monthly quota.

    Raises:
      403 if the subscription billing_status is suspended/past_due/cancelled.
      429 if sessions_used >= limit and the plan is not unlimited.
    Returns silently otherwise.
    """
    subscription = (
        db.query(Subscription)
        .filter(Subscription.company_id == company_id)
        .first()
    )
    if subscription is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No subscription for company",
        )

    if subscription.billing_status in _BLOCKED_BILLING_STATES:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Subscription inactive",
        )

    limit = session_limit_for(db, company_id)
    if limit >= UNLIMITED:
        # Unlimited plans skip the cap check but still record usage for analytics.
        return

    used = sessions_used_this_period(db, company_id)
    if used >= limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Monthly session limit reached",
        )
