# backend/plans.py
"""Single source of truth for subscription plans. Mirrored to the `plans` table
via scripts/seed_plans.py. Frontend reads these via GET /plans."""
from __future__ import annotations

UNLIMITED = 1_000_000  # sentinel for "unlimited sessions"

PLANS: dict[str, dict] = {
    "free":       {"display_name": "Free",       "seat_limit": 1,   "session_limit_monthly": 5,         "gaas_enabled": False, "price_monthly_usd": 0,    "price_annual_usd": 0},
    "starter":    {"display_name": "Starter",    "seat_limit": 5,   "session_limit_monthly": 30,        "gaas_enabled": False, "price_monthly_usd": 29,   "price_annual_usd": 290},
    "growth":     {"display_name": "Growth",     "seat_limit": 20,  "session_limit_monthly": 150,       "gaas_enabled": True,  "price_monthly_usd": 99,   "price_annual_usd": 990},
    "scale":      {"display_name": "Scale",      "seat_limit": 100, "session_limit_monthly": UNLIMITED, "gaas_enabled": True,  "price_monthly_usd": 299,  "price_annual_usd": 2990},
    "enterprise": {"display_name": "Enterprise", "seat_limit": UNLIMITED, "session_limit_monthly": UNLIMITED, "gaas_enabled": True, "price_monthly_usd": None, "price_annual_usd": None},
}

PLAN_ORDER = ["free", "starter", "growth", "scale", "enterprise"]
FREE_PERSONA_IDS = ["first_time_buyer", "friendly_customer"]  # Easy-only gating for free plan


def get_plan(name: str) -> dict:
    if name not in PLANS:
        raise ValueError(f"Unknown plan '{name}'. Valid: {', '.join(PLAN_ORDER)}")
    return {"name": name, **PLANS[name]}


def all_plans() -> list[dict]:
    return [get_plan(n) for n in PLAN_ORDER]
