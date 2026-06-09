# scripts/fix_schema.py
"""
Idempotent schema reconciler — brings a drifted database up to the current
models WITHOUT relying on alembic's recorded revision.

Use this when a deployment's DB predates the multi-tenant redesign and is
missing tables/columns the code expects (e.g. the classic
`column users.company_id does not exist` 500 on /api/auth/login).

It is SAFE to run repeatedly:
  - create_all() only creates tables that don't already exist.
  - every ALTER uses ADD COLUMN IF NOT EXISTS, so present columns are untouched.
  - no data is dropped or rewritten.

Run on the box that owns the database (e.g. EC2), from the repo root, in the
same env the server uses:

    conda activate vcai          # or whatever env runs uvicorn
    python scripts/fix_schema.py

Then restart the backend so it reconnects to the corrected schema.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import text

from backend.database import engine, Base
# Import every model so Base.metadata knows about all tables before create_all.
from backend.models import (  # noqa: F401
    User, Session, Persona, Message, EmotionLog,
)
# These may not all be exported from backend.models; import directly to be safe.
import backend.models.company        # noqa: F401
import backend.models.subscription   # noqa: F401
import backend.models.plan           # noqa: F401
import backend.models.seat_invite    # noqa: F401
import backend.models.usage_period   # noqa: F401
import backend.models.audit_log      # noqa: F401
import backend.models.abuse_flag     # noqa: F401
import backend.models.evaluation     # noqa: F401


# Additive columns the code expects on pre-existing tables. (table, column, type-DDL)
ADDITIVE_COLUMNS = [
    ("users",        "company_id",     "UUID"),
    ("seat_invites", "invite_code",    "VARCHAR(12)"),
    ("sessions",     "scenario",       "JSONB"),
    ("sessions",     "training_focus", "VARCHAR(100)"),
    ("personas",     "gender",         "VARCHAR(10) NOT NULL DEFAULT 'male'"),
    ("personas",     "gender_mode",    "VARCHAR(20) NOT NULL DEFAULT 'male_only'"),
]


def main() -> None:
    print(f"Target DB engine: {engine.url.render_as_string(hide_password=True)}")

    # 1) Create any tables that don't exist yet (companies, subscriptions,
    #    seat_invites, usage_periods, plans, audit_logs, abuse_flags,
    #    evaluation_reports, ...). Existing tables are left alone.
    print("\n[1/3] Ensuring all model tables exist (create_all, checkfirst)...")
    Base.metadata.create_all(bind=engine, checkfirst=True)
    print("      done.")

    # 2) Add missing additive columns idempotently.
    print("\n[2/3] Adding missing additive columns (IF NOT EXISTS)...")
    with engine.begin() as conn:
        for table, col, ddl in ADDITIVE_COLUMNS:
            # Only attempt if the table exists at all.
            exists_tbl = conn.execute(
                text("SELECT to_regclass(:t)"), {"t": f"public.{table}"}
            ).scalar()
            if exists_tbl is None:
                print(f"      - {table}.{col}: table missing, skipped (create_all should have made it)")
                continue
            conn.execute(text(
                f'ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} {ddl}'
            ))
            print(f"      - {table}.{col}: ensured")

        # FK + index for users.company_id (only if missing).
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_users_company_id ON users (company_id)"
        ))
        fk = conn.execute(text(
            "SELECT 1 FROM information_schema.table_constraints "
            "WHERE constraint_name='fk_users_company_id_companies' AND table_name='users'"
        )).scalar()
        if not fk:
            conn.execute(text(
                "ALTER TABLE users ADD CONSTRAINT fk_users_company_id_companies "
                "FOREIGN KEY (company_id) REFERENCES companies(id) ON DELETE CASCADE"
            ))
            print("      - users.company_id FK -> companies: added")
        else:
            print("      - users.company_id FK -> companies: already present")

        # invite_code unique index (matches model).
        conn.execute(text(
            "CREATE UNIQUE INDEX IF NOT EXISTS ix_seat_invites_invite_code "
            "ON seat_invites (invite_code)"
        ))

    # 3) Verify the column that was breaking login.
    print("\n[3/3] Verifying users.company_id ...")
    with engine.connect() as conn:
        ok = conn.execute(text(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name='users' AND column_name='company_id'"
        )).scalar()
    print("      users.company_id present:", bool(ok))
    print("\nSchema reconcile complete. Restart the backend now.")


if __name__ == "__main__":
    main()
