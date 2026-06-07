"""usage indexes: composite index on sessions(user_id, created_at) (Phase 4)

Revision ID: 0004_usage_indexes
Revises: 0002_audit_logs
Create Date: 2026-06-07

Phase 4 adds tenant scoping + monthly usage metering around the existing
session endpoints. No table reshapes. The only schema change is a composite
index supporting the per-user / per-period session scans used by usage
counting and the manager/scope queries.

The usage_periods unique index (company_id, period_start) already exists from
revision 0001_tenancy_baseline, so it is NOT recreated here.

Roll back with: alembic downgrade -1
"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "0004_usage_indexes"
down_revision: Union[str, None] = "0002_audit_logs"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Supports "sessions for this user, newest first" and per-period counting
    # scans introduced by usage metering and the scope queries.
    op.create_index(
        "ix_sessions_user_id_created_at",
        "sessions",
        ["user_id", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_sessions_user_id_created_at", table_name="sessions")
