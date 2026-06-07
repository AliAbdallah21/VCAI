# backend/routers/plans.py
"""
Public Plans API endpoint.

Serves the subscription plan matrix from backend/plans.py (the single source of
truth). No auth required — used by the public landing/pricing page.
"""

from fastapi import APIRouter

from backend.plans import all_plans

router = APIRouter(prefix="/plans", tags=["Plans"])


@router.get("")
def list_plans():
    """
    List all subscription plans, ordered free -> enterprise.

    Each plan includes: name, display_name, seat_limit, session_limit_monthly,
    gaas_enabled, price_monthly_usd, price_annual_usd.
    """
    return all_plans()
