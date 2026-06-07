# backend/models/plan.py
"""
Plan SQLAlchemy model. Mirrors backend/plans.py (seeded via scripts/seed_plans.py)
so enforcement can JOIN without importing Python.
"""

from sqlalchemy import Column, String, Integer, Boolean

from backend.database import Base


class Plan(Base):
    """Plan model - subscription tier definitions mirrored from plans.py."""

    __tablename__ = "plans"

    name = Column(String(50), primary_key=True)
    display_name = Column(String(100))
    seat_limit = Column(Integer)
    session_limit_monthly = Column(Integer)
    gaas_enabled = Column(Boolean)
    price_monthly_usd = Column(Integer, nullable=True)
    price_annual_usd = Column(Integer, nullable=True)

    def __repr__(self):
        return f"<Plan {self.name}>"
