"""
scripts/migrate_add_gender.py

One-time migration: add gender and gender_mode columns to the personas table.
Safe to re-run — uses IF NOT EXISTS checks.

Usage:
    python scripts/migrate_add_gender.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from backend.database import engine
from sqlalchemy import text


def run() -> None:
    with engine.connect() as conn:
        conn.execute(text("""
            ALTER TABLE personas
            ADD COLUMN IF NOT EXISTS gender VARCHAR(10) NOT NULL DEFAULT 'male';
        """))
        conn.execute(text("""
            ALTER TABLE personas
            ADD COLUMN IF NOT EXISTS gender_mode VARCHAR(20) NOT NULL DEFAULT 'male_only';
        """))
        conn.commit()
    print("Migration complete: gender and gender_mode columns added.")


if __name__ == "__main__":
    run()
