"""
Add the `scenario` JSONB column to the sessions table.

Idempotent — safe to run multiple times (uses ADD COLUMN IF NOT EXISTS).
This is the project's standard way of applying a schema change (see
seed_personas.py / cleanup_old_audio.py); the scaffolded Alembic setup is
not wired up.

Usage:
    python scripts/add_scenario_column.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import text
from backend.database import engine


def main() -> None:
    print("[migrate] Adding sessions.scenario (JSONB, nullable) if missing...")
    with engine.begin() as conn:
        conn.execute(text(
            "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS scenario JSONB"
        ))
        # Verify
        row = conn.execute(text(
            "SELECT column_name, data_type, is_nullable "
            "FROM information_schema.columns "
            "WHERE table_name = 'sessions' AND column_name = 'scenario'"
        )).fetchone()

    if row:
        print(f"[migrate] OK — column present: {row[0]} ({row[1]}, nullable={row[2]})")
    else:
        print("[migrate] FAILED — column not found after ALTER. Check DB connection.")
        sys.exit(1)


if __name__ == "__main__":
    main()
