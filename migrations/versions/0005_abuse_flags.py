"""abuse flags: add abuse_flags table (Phase 5 read/triage; Phase 7 writes)

Revision ID: 0005_abuse_flags
Revises: 0004_usage_indexes
Create Date: 2026-06-07

Adds the abuse_flags table from 00_ARCHITECTURE.md sections 6/9. Phase 5 builds
the manager review/triage UI + endpoints that READ this table; the rule-based
detection engine that writes rows lands in Phase 7. Additive only.

Roll back with: alembic downgrade -1
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "0005_abuse_flags"
down_revision: Union[str, None] = "0004_usage_indexes"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "abuse_flags",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("company_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("reason", sa.String(length=50), nullable=False),
        sa.Column("severity", sa.String(length=20), nullable=False),
        sa.Column("detail", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolved_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.ForeignKeyConstraint(["company_id"], ["companies.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["resolved_by"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_abuse_flags_company_id"), "abuse_flags", ["company_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_abuse_flags_company_id"), table_name="abuse_flags")
    op.drop_table("abuse_flags")
