# scripts/seed_plans.py
"""
Seed the subscription plans into the plans table, mirrored from backend/plans.py.

Existing rows are updated (upsert via merge) so this script is safe to re-run.

Usage:
    python scripts/seed_plans.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make sure project root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from backend.database import SessionLocal
from backend.models.plan import Plan
from backend.plans import PLANS, PLAN_ORDER


def seed(db) -> None:
    inserted = 0
    updated = 0
    for name in PLAN_ORDER:
        data = PLANS[name]
        row = db.query(Plan).filter(Plan.name == name).first()
        if row is None:
            row = Plan(name=name)
            db.add(row)
            action = "INSERT"
            inserted += 1
        else:
            action = "UPDATE"
            updated += 1

        row.display_name = data["display_name"]
        row.seat_limit = data["seat_limit"]
        row.session_limit_monthly = data["session_limit_monthly"]
        row.gaas_enabled = data["gaas_enabled"]
        row.price_monthly_usd = data["price_monthly_usd"]
        row.price_annual_usd = data["price_annual_usd"]

        print(f"  [{action}] {name} (seats={data['seat_limit']}, "
              f"sessions/mo={data['session_limit_monthly']}, gaas={data['gaas_enabled']})")

    db.commit()
    print(f"\nSeeded {len(PLAN_ORDER)} plans successfully "
          f"({inserted} inserted, {updated} updated).")


if __name__ == "__main__":
    print("Seeding plans...")
    db = SessionLocal()
    try:
        seed(db)
    finally:
        db.close()

    # Quick verification read-back
    db2 = SessionLocal()
    try:
        print("\nVerification — plans in DB:")
        for p in db2.query(Plan).all():
            print(f"  {p.name:<12} {p.display_name:<12} "
                  f"seats={p.seat_limit:<8} sessions/mo={p.session_limit_monthly}")
    finally:
        db2.close()
