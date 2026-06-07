"""reconcile drifted additive columns with the models

Revision ID: 0006_reconcile_model_columns
Revises: 0005_abuse_flags
Create Date: 2026-06-07

Some live databases reached alembic head while missing additive columns that
exist on the SQLAlchemy models (they were applied via create_all on other
machines but never shipped as a migration). This revision reconciles that drift
idempotently with ADD COLUMN IF NOT EXISTS, so a DB already carrying the columns
is unaffected and a drifted DB is brought in line.

Columns reconciled:
  - sessions.scenario        (JSONB,   nullable)
  - sessions.training_focus  (String,  nullable)
  - personas.gender          (String,  default 'male')
  - personas.gender_mode     (String,  default 'male_only')

Roll back with: alembic downgrade -1  (drops the columns IF EXISTS)
"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "0006_reconcile_model_columns"
down_revision: Union[str, None] = "0005_abuse_flags"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Postgres supports IF NOT EXISTS on ADD COLUMN, making this safe on DBs that
    # already have the columns. Plain SQL keeps it dialect-explicit.
    op.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS scenario JSONB")
    op.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS training_focus VARCHAR(100)")
    op.execute("ALTER TABLE personas ADD COLUMN IF NOT EXISTS gender VARCHAR(10) NOT NULL DEFAULT 'male'")
    op.execute("ALTER TABLE personas ADD COLUMN IF NOT EXISTS gender_mode VARCHAR(20) NOT NULL DEFAULT 'male_only'")


def downgrade() -> None:
    # Only drop what this revision may have added; IF EXISTS keeps it safe.
    op.execute("ALTER TABLE personas DROP COLUMN IF EXISTS gender_mode")
    op.execute("ALTER TABLE personas DROP COLUMN IF EXISTS gender")
    op.execute("ALTER TABLE sessions DROP COLUMN IF EXISTS training_focus")
    op.execute("ALTER TABLE sessions DROP COLUMN IF EXISTS scenario")
