# backend/models/abuse_flag.py
"""
AbuseFlag SQLAlchemy model - rule-based abuse/misuse flags surfaced to managers
(own company) and superadmins (all). The detection engine that WRITES these is
Phase 7; Phase 5 only reads + triages them. See 00_ARCHITECTURE.md sections 6, 9.
"""

from sqlalchemy import Column, String, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid

from backend.database import Base


class AbuseFlag(Base):
    """Abuse flag model - one row per detected misuse event."""

    __tablename__ = "abuse_flags"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    company_id = Column(
        UUID(as_uuid=True),
        ForeignKey("companies.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    reason = Column(String(50), nullable=False)      # seat_sharing, rapid_fire, ...
    severity = Column(String(20), nullable=False)    # low, medium, high
    detail = Column(JSONB, nullable=True)
    status = Column(String(20), default="open")      # open, reviewed, dismissed
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    resolved_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)

    def __repr__(self):
        return f"<AbuseFlag {self.reason} ({self.severity}/{self.status})>"
