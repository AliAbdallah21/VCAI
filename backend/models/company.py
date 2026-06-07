# backend/models/company.py
"""
Company SQLAlchemy model - the tenant root.
"""

from sqlalchemy import Column, String, Boolean, DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid

from backend.database import Base


class Company(Base):
    """Company model - the tenant root. Owns users, one subscription, usage, etc.

    Plan/limit fields live on Subscription, not here, to keep billing separable.
    """

    __tablename__ = "companies"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    slug = Column(String(120), unique=True, index=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    users = relationship("User", back_populates="company_ref")
    subscription = relationship(
        "Subscription",
        back_populates="company",
        uselist=False,
        cascade="all, delete-orphan",
    )

    def __repr__(self):
        return f"<Company {self.name} ({self.slug})>"
