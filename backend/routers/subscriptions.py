# backend/routers/subscriptions.py
"""
Subscription API (manager-only). View the current subscription with resolved plan
limits + usage, and change plan (MOCKED - no payment). See Phase 3.
"""

from datetime import date

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import Company, Subscription, UsagePeriod, User
from backend.plans import get_plan
from backend.schemas import SubscriptionInfo, PlanChange
from backend.services import (
    get_current_manager,
    assert_same_company,
    count_active_seats,
    record_audit,
)

router = APIRouter(prefix="/subscriptions", tags=["Subscriptions"])


def _manager_company(db: Session, current_user: User) -> Company:
    if current_user.company_id is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="User is not attached to a company")
    assert_same_company(current_user, current_user.company_id)
    company = db.query(Company).filter(Company.id == current_user.company_id).first()
    if company is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Company not found")
    return company


def _sessions_used_this_period(db: Session, company_id) -> int:
    period_start = date.today().replace(day=1)
    row = (
        db.query(UsagePeriod.sessions_used)
        .filter(UsagePeriod.company_id == company_id, UsagePeriod.period_start == period_start)
        .first()
    )
    return int(row[0]) if row and row[0] is not None else 0


def _build_info(db: Session, company: Company, sub: Subscription) -> SubscriptionInfo:
    plan = get_plan(sub.plan_name)
    return SubscriptionInfo(
        plan_name=sub.plan_name,
        display_name=plan["display_name"],
        billing_cycle=sub.billing_cycle,
        billing_status=sub.billing_status,
        trial_ends_at=sub.trial_ends_at,
        seat_limit=plan["seat_limit"],
        session_limit_monthly=plan["session_limit_monthly"],
        gaas_enabled=plan["gaas_enabled"],
        price_monthly_usd=plan["price_monthly_usd"],
        price_annual_usd=plan["price_annual_usd"],
        seats_used=count_active_seats(db, company.id),
        sessions_used_this_period=_sessions_used_this_period(db, company.id),
    )


@router.get("/me", response_model=SubscriptionInfo)
def my_subscription(db: Session = Depends(get_db),
                    current_user: User = Depends(get_current_manager)):
    """Current company subscription + resolved plan limits + usage summary."""
    company = _manager_company(db, current_user)
    sub = db.query(Subscription).filter(Subscription.company_id == company.id).first()
    if sub is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No subscription")
    return _build_info(db, company, sub)


@router.post("/change-plan", response_model=SubscriptionInfo)
def change_plan(payload: PlanChange, db: Session = Depends(get_db),
                current_user: User = Depends(get_current_manager)):
    """
    MOCKED plan change. Updates the subscription row and re-derives limits. If
    downgrading below current active seats, returns 409 with how many to free.
    """
    company = _manager_company(db, current_user)

    try:
        new_plan = get_plan(payload.plan_name)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    if payload.billing_cycle not in ("monthly", "annual"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="billing_cycle must be 'monthly' or 'annual'")

    sub = db.query(Subscription).filter(Subscription.company_id == company.id).first()
    if sub is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No subscription")

    active_seats = count_active_seats(db, company.id)
    if active_seats > new_plan["seat_limit"]:
        must_free = active_seats - new_plan["seat_limit"]
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Plan '{new_plan['name']}' allows {new_plan['seat_limit']} seats but you have "
                f"{active_seats} active agents. Deactivate {must_free} before downgrading."
            ),
        )

    old_plan = sub.plan_name
    sub.plan_name = new_plan["name"]
    sub.billing_cycle = payload.billing_cycle
    record_audit(
        db,
        action="subscription.changed",
        actor=current_user,
        company_id=company.id,
        target_type="subscription",
        target_id=sub.id,
        detail={"from": old_plan, "to": new_plan["name"], "billing_cycle": payload.billing_cycle},
    )
    db.commit()
    db.refresh(sub)
    return _build_info(db, company, sub)
