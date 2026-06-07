"""
Migration: add training_focus column to sessions table.

training_focus records which skill the adaptive system targeted for that session
(e.g. "closing", "objection_handling"). NULL means a free-choice session.

Safe to run multiple times (ADD COLUMN IF NOT EXISTS).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

import psycopg2

DATABASE_URL = os.environ["DATABASE_URL"]


def run():
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                ALTER TABLE sessions
                ADD COLUMN IF NOT EXISTS training_focus VARCHAR(100);
            """)
        conn.commit()
        print("Migration complete: training_focus column added to sessions.")
    finally:
        conn.close()


if __name__ == "__main__":
    run()
