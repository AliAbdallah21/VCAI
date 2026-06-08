"""add invite_code to seat_invites

Revision ID: 0007_seat_invite_code
Revises: 0006_reconcile_model_columns
Create Date: 2026-06-08

Adds a short, human-typeable invite_code (6 alphanumeric chars) to seat_invites so
an invitee can join a company by pasting the code at registration or in settings,
as an alternative to the full invite link. Column is nullable so pre-existing
invites are unaffected; a unique index enforces code uniqueness.

Idempotent (ADD COLUMN IF NOT EXISTS) so a DB that already has the column via
create_all is unaffected.

Roll back with: alembic downgrade -1  (drops the column + index IF EXISTS)
"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "0007_seat_invite_code"
down_revision: Union[str, None] = "0006_reconcile_model_columns"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE seat_invites ADD COLUMN IF NOT EXISTS invite_code VARCHAR(12)")
    # Unique index (partial: NULLs are allowed and not considered equal in Postgres,
    # but a plain unique index already permits multiple NULLs, so this is fine).
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ix_seat_invites_invite_code "
        "ON seat_invites (invite_code)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_seat_invites_invite_code")
    op.execute("ALTER TABLE seat_invites DROP COLUMN IF EXISTS invite_code")
