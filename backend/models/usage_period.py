# backend/models/usage_period.py
"""
UsagePeriod SQLAlchemy model - one row per company per calendar month, tracking
sessions used and peak seats for limit enforcement and analytics.
"""

from sqlalchemy import Column, Integer, Date, DateTime, ForeignKey, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID
import uuid

from backend.database import Base


class UsagePeriod(Base):
    """Usage period model - per-company per-month metering row."""

    __tablename__ = "usage_periods"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    company_id = Column(
        UUID(as_uuid=True),
        ForeignKey("companies.id", ondelete="CASCADE"),
        nullable=False,
    )
    period_start = Column(Date, nullable=False)
    sessions_used = Column(Integer, default=0)
    seats_peak = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint("company_id", "period_start", name="uq_usage_company_period"),
    )

    def __repr__(self):
        return f"<UsagePeriod {self.company_id} {self.period_start} used={self.sessions_used}>"
