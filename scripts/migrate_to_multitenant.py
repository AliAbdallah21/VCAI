# scripts/migrate_to_multitenant.py
"""
One-time data migration: convert the single-tenant prototype data into the
multi-tenant model.

What it does (all in ONE transaction, idempotent, safe to re-run):
  1. For each distinct non-empty User.company string, create a Company (slug
     derived from the name) if one does not already exist.
  2. Create one Subscription per new company (plan_name="free",
     billing_status="active").
  3. Set each user's company_id to its company (by matching the legacy
     User.company string). Users with no company string are left NULL and logged.
  4. Migrate role: any user with role == "admin" becomes "manager".

PREREQUISITES:
  - Run `alembic upgrade head` first (creates companies/plans/subscriptions/etc.).
  - Run `python scripts/seed_plans.py` first (the "free" plan row must exist,
    since subscriptions.plan_name has a FK to plans.name).

This script is GUARDED by a confirmation prompt and never runs unattended.

Usage:
    python scripts/migrate_to_multitenant.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# Make sure project root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from backend.database import SessionLocal
from backend.models.user import User
from backend.models.company import Company
from backend.models.subscription import Subscription
from backend.models.plan import Plan


def slugify(name: str) -> str:
    """lowercase, non-alnum -> hyphen, collapse repeats, strip ends."""
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s)
    return s.strip("-") or "company"


def unique_slug(db, base: str, taken: set[str]) -> str:
    """Return a slug not present in `taken` nor in the companies table."""
    slug = base
    suffix = 1
    while slug in taken or db.query(Company).filter(Company.slug == slug).first() is not None:
        suffix += 1
        slug = f"{base}-{suffix}"
    taken.add(slug)
    return slug


def migrate(db) -> None:
    # Guard: the "free" plan must be seeded (FK target for subscriptions).
    if db.query(Plan).filter(Plan.name == "free").first() is None:
        print("ERROR: plan 'free' not found. Run `python scripts/seed_plans.py` first.")
        return

    companies_created = 0
    subscriptions_created = 0
    users_migrated = 0
    users_unassigned = 0
    roles_changed = 0

    taken_slugs: set[str] = set()

    # ── 1+2: create a Company (+ free Subscription) per distinct company string ──
    distinct_companies = [
        row[0] for row in db.query(User.company).distinct().all()
        if row[0] is not None and row[0].strip() != ""
    ]

    # Map normalized company name -> Company row.
    name_to_company: dict[str, Company] = {}
    for raw_name in distinct_companies:
        name = raw_name.strip()
        existing = db.query(Company).filter(Company.name == name).first()
        if existing is not None:
            name_to_company[name] = existing
            continue

        slug = unique_slug(db, slugify(name), taken_slugs)
        company = Company(name=name, slug=slug, is_active=True)
        db.add(company)
        db.flush()  # assign company.id
        companies_created += 1

        subscription = Subscription(
            company_id=company.id,
            plan_name="free",
            billing_cycle="monthly",
            billing_status="active",
        )
        db.add(subscription)
        subscriptions_created += 1

        name_to_company[name] = company
        print(f"  [COMPANY] '{name}' -> slug='{slug}' (+ free subscription)")

    # ── 3: assign users to their company ──
    for user in db.query(User).all():
        if user.company is not None and user.company.strip() != "":
            company = name_to_company.get(user.company.strip())
            if company is not None:
                if user.company_id != company.id:
                    user.company_id = company.id
                    users_migrated += 1
            else:
                user.company_id = None
                users_unassigned += 1
        else:
            users_unassigned += 1
            print(f"  [UNASSIGNED] user {user.email} has no company string -> company_id NULL")

        # ── 4: role migration admin -> manager ──
        if user.role == "admin":
            user.role = "manager"
            roles_changed += 1
            print(f"  [ROLE] {user.email}: admin -> manager")

    db.commit()

    # ── 6: summary ──
    print("\n" + "=" * 50)
    print("Migration summary")
    print("=" * 50)
    print(f"  Companies created:     {companies_created}")
    print(f"  Subscriptions created: {subscriptions_created}")
    print(f"  Users migrated:        {users_migrated}")
    print(f"  Users left unassigned: {users_unassigned}")
    print(f"  Roles changed:         {roles_changed}")


if __name__ == "__main__":
    print("Multi-tenant data migration")
    print("Prerequisites: `alembic upgrade head` and `python scripts/seed_plans.py` "
          "must have been run.")
    confirm = input("Proceed? [y/N] ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        sys.exit(0)

    db = SessionLocal()
    try:
        migrate(db)
    except Exception:
        db.rollback()
        print("\nMigration FAILED and was rolled back.")
        raise
    finally:
        db.close()
