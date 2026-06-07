# backend/services/audit_service.py
"""
Audit logging helper. Every privileged action writes an audit_logs row.
See 00_ARCHITECTURE.md section 7. Callers are responsible for committing the
session (this only adds the row, so it participates in the caller's transaction).
"""

from typing import Optional
from uuid import UUID

from sqlalchemy.orm import Session

from backend.models import AuditLog, User


def record_audit(
    db: Session,
    *,
    action: str,
    actor: Optional[User] = None,
    company_id=None,
    target_type: Optional[str] = None,
    target_id=None,
    detail: Optional[dict] = None,
) -> AuditLog:
    """Append an audit_logs row to the current session (does not commit)."""
    log = AuditLog(
        action=action,
        actor_user_id=(actor.id if actor else None),
        actor_role=(actor.role if actor else None),
        company_id=company_id,
        target_type=target_type,
        target_id=(str(target_id) if target_id is not None else None),
        detail=detail,
    )
    db.add(log)
    return log
