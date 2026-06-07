# scripts/seed_test_company.py
"""
Seed a dedicated TEST company with an unlimited plan (scale: 1,000,000 sessions/mo).

Use this for internal testing / demos where session and seat limits must not get in
the way. Idempotent: re-running updates the existing test company/subscription rather
than creating duplicates.

PREREQUISITES (run first, in order):
    alembic upgrade head        # creates companies/plans/subscriptions tables
    python scripts/seed_plans.py  # the "scale" plan row must exist (FK target)

Usage:
    python scripts/seed_test_company.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make sure project root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from backend.database import SessionLocal
from backend.models.company import Company
from backend.models.subscription import Subscription
from backend.models.plan import Plan

# Identity of the test company. Kept stable so the script is idempotent.
TEST_COMPANY_NAME = "VCAI Test Company"
TEST_COMPANY_SLUG = "vcai-test"
TEST_PLAN = "scale"  # unlimited sessions (1,000,000 sentinel), 100 seats, GaaS on


def seed(db) -> None:
    # The "scale" plan must be seeded (FK target for subscriptions.plan_name).
    if db.query(Plan).filter(Plan.name == TEST_PLAN).first() is None:
        print(f"ERROR: plan '{TEST_PLAN}' not found. Run `python scripts/seed_plans.py` first.")
        return

    company = db.query(Company).filter(Company.slug == TEST_COMPANY_SLUG).first()
    if company is None:
        company = Company(name=TEST_COMPANY_NAME, slug=TEST_COMPANY_SLUG, is_active=True)
        db.add(company)
        db.flush()  # assign company.id
        print(f"  [INSERT] company '{TEST_COMPANY_NAME}' (slug={TEST_COMPANY_SLUG})")
    else:
        company.name = TEST_COMPANY_NAME
        company.is_active = True
        print(f"  [UPDATE] company '{TEST_COMPANY_NAME}' (slug={TEST_COMPANY_SLUG})")

    subscription = db.query(Subscription).filter(
        Subscription.company_id == company.id
    ).first()
    if subscription is None:
        subscription = Subscription(company_id=company.id)
        db.add(subscription)
        action = "INSERT"
    else:
        action = "UPDATE"

    subscription.plan_name = TEST_PLAN
    subscription.billing_cycle = "annual"
    subscription.billing_status = "active"
    print(f"  [{action}] subscription -> plan='{TEST_PLAN}' (unlimited sessions), status=active")

    db.commit()
    print(f"\nTest company ready: '{TEST_COMPANY_NAME}' on the '{TEST_PLAN}' plan.")
    print(f"  company_id = {company.id}")
    print("  Assign a user to it with:  UPDATE users SET company_id = '<id>' WHERE email = '<you>';")


if __name__ == "__main__":
    print("Seeding test company...")
    db = SessionLocal()
    try:
        seed(db)
    finally:
        db.close()
