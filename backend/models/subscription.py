# backend/models/subscription.py
"""
Subscription SQLAlchemy model. One subscription per company. Stripe columns are
nullable and unused until the Stripe phase (billing is mocked for now).
"""

from sqlalchemy import Column, String, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid

from backend.database import Base


class Subscription(Base):
    """Subscription model - one per company. Billing is mocked, Stripe-ready."""

    __tablename__ = "subscriptions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    company_id = Column(
        UUID(as_uuid=True),
        ForeignKey("companies.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    plan_name = Column(String(50), ForeignKey("plans.name"), nullable=False)
    billing_cycle = Column(String(20), default="monthly")  # monthly, annual
    billing_status = Column(String(30), default="active")  # active, trial, past_due, suspended, cancelled
    trial_ends_at = Column(DateTime(timezone=True), nullable=True)
    current_period_end = Column(DateTime(timezone=True), nullable=True)
    stripe_customer_id = Column(String(255), nullable=True)
    stripe_subscription_id = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    company = relationship("Company", back_populates="subscription")

    def __repr__(self):
        return f"<Subscription {self.plan_name} ({self.billing_status})>"
