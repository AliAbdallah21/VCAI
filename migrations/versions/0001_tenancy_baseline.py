"""tenancy baseline: add multi-tenancy tables + users.company_id

Revision ID: 0001_tenancy_baseline
Revises:
Create Date: 2026-06-07

This is a HAND-WRITTEN baseline. The live database already has the existing
tables (users, sessions, personas, messages, emotion_logs, checkpoints,
user_stats, evaluation_reports), so this revision must NOT recreate them.

It only:
  (a) creates the 5 new tenancy/plan tables, and
  (b) adds users.company_id (nullable) + its index + FK to companies.

Roll back with: alembic downgrade -1
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "0001_tenancy_baseline"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── plans (no FKs; mirrors backend/plans.py, seeded separately) ──────────
    op.create_table(
        "plans",
        sa.Column("name", sa.String(length=50), nullable=False),
        sa.Column("display_name", sa.String(length=100), nullable=True),
        sa.Column("seat_limit", sa.Integer(), nullable=True),
        sa.Column("session_limit_monthly", sa.Integer(), nullable=True),
        sa.Column("gaas_enabled", sa.Boolean(), nullable=True),
        sa.Column("price_monthly_usd", sa.Integer(), nullable=True),
        sa.Column("price_annual_usd", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("name"),
    )

    # ── companies (tenant root; no FKs) ─────────────────────────────────────
    op.create_table(
        "companies",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("slug", sa.String(length=120), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("slug"),
    )
    op.create_index(op.f("ix_companies_slug"), "companies", ["slug"], unique=True)

    # ── subscriptions (→ companies, plans) ──────────────────────────────────
    op.create_table(
        "subscriptions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("company_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("plan_name", sa.String(length=50), nullable=False),
        sa.Column("billing_cycle", sa.String(length=20), nullable=True),
        sa.Column("billing_status", sa.String(length=30), nullable=True),
        sa.Column("trial_ends_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("current_period_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("stripe_customer_id", sa.String(length=255), nullable=True),
        sa.Column("stripe_subscription_id", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.ForeignKeyConstraint(["company_id"], ["companies.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["plan_name"], ["plans.name"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("company_id"),
    )

    # ── seat_invites (→ companies, users) ───────────────────────────────────
    op.create_table(
        "seat_invites",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("company_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("role", sa.String(length=50), nullable=True),
        sa.Column("token", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=True),
        sa.Column("invited_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.ForeignKeyConstraint(["company_id"], ["companies.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["invited_by"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("token"),
    )
    op.create_index(op.f("ix_seat_invites_company_id"), "seat_invites", ["company_id"], unique=False)
    op.create_index(op.f("ix_seat_invites_token"), "seat_invites", ["token"], unique=True)

    # ── usage_periods (→ companies) ─────────────────────────────────────────
    op.create_table(
        "usage_periods",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("company_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("period_start", sa.Date(), nullable=False),
        sa.Column("sessions_used", sa.Integer(), nullable=True),
        sa.Column("seats_peak", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.ForeignKeyConstraint(["company_id"], ["companies.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("company_id", "period_start", name="uq_usage_company_period"),
    )

    # ── users.company_id (additive column on existing table) ────────────────
    op.add_column(
        "users",
        sa.Column("company_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_index(op.f("ix_users_company_id"), "users", ["company_id"], unique=False)
    op.create_foreign_key(
        "fk_users_company_id_companies",
        "users",
        "companies",
        ["company_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    # Reverse order. Only touch what upgrade() created.
    op.drop_constraint("fk_users_company_id_companies", "users", type_="foreignkey")
    op.drop_index(op.f("ix_users_company_id"), table_name="users")
    op.drop_column("users", "company_id")

    op.drop_table("usage_periods")

    op.drop_index(op.f("ix_seat_invites_token"), table_name="seat_invites")
    op.drop_index(op.f("ix_seat_invites_company_id"), table_name="seat_invites")
    op.drop_table("seat_invites")

    op.drop_table("subscriptions")

    op.drop_index(op.f("ix_companies_slug"), table_name="companies")
    op.drop_table("companies")

    op.drop_table("plans")
