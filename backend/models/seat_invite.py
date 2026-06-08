# backend/models/seat_invite.py
"""
SeatInvite SQLAlchemy model - pending invitations for agents (salespersons) to
join a company within its seat limit.
"""

from sqlalchemy import Column, String, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID
import uuid

from backend.database import Base


class SeatInvite(Base):
    """Seat invite model - an emailed invitation to join a company."""

    __tablename__ = "seat_invites"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    company_id = Column(
        UUID(as_uuid=True),
        ForeignKey("companies.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    email = Column(String(255), nullable=False)
    role = Column(String(50), default="salesperson")
    token = Column(String(128), unique=True, index=True, nullable=False)
    # Short, human-typeable code the invitee can paste at registration or in
    # settings (alternative to the full link). Nullable for pre-existing invites.
    invite_code = Column(String(12), unique=True, index=True, nullable=True)
    status = Column(String(20), default="pending")  # pending, accepted, revoked, expired
    invited_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    expires_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<SeatInvite {self.email} ({self.status})>"
