# backend/models/audit_log.py
"""
AuditLog SQLAlchemy model - records every privileged action (seat invites, role
changes, plan changes, deactivations, etc.). Also the substrate for the future
GaaS RL layer. See 00_ARCHITECTURE.md section 7.
"""

from sqlalchemy import Column, String, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid

from backend.database import Base


class AuditLog(Base):
    """Audit log model - one row per privileged action."""

    __tablename__ = "audit_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    company_id = Column(
        UUID(as_uuid=True),
        ForeignKey("companies.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    actor_user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    actor_role = Column(String(50), nullable=True)
    action = Column(String(100), nullable=False)
    target_type = Column(String(100), nullable=True)
    target_id = Column(String(255), nullable=True)
    detail = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<AuditLog {self.action} ({self.target_type}:{self.target_id})>"
