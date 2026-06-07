# backend/services/abuse_service.py
"""
Abuse-flag review service (Phase 5). Lists a company's abuse_flags and lets a
manager resolve them (reviewed/dismissed). The detection engine that writes
flags is Phase 7. Every resolution writes an audit_logs row. See
00_ARCHITECTURE.md sections 6, 7.
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy.orm import Session as DbSession

from backend.models import AbuseFlag, User
from backend.services.audit_service import record_audit

_VALID_STATUSES = {"open", "reviewed", "dismissed"}
_RESOLVE_STATUSES = {"reviewed", "dismissed"}


def list_flags(db: DbSession, company_id, status_filter: Optional[str] = None) -> list:
    """Abuse flags for a company, newest first, optionally filtered by status."""
    q = db.query(AbuseFlag).filter(AbuseFlag.company_id == company_id)
    if status_filter:
        if status_filter not in _VALID_STATUSES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status filter '{status_filter}'",
            )
        q = q.filter(AbuseFlag.status == status_filter)
    return q.order_by(AbuseFlag.created_at.desc()).all()


def resolve_flag(
    db: DbSession,
    *,
    company_id,
    actor: User,
    flag_id: UUID,
    new_status: str,
    note: Optional[str] = None,
) -> AbuseFlag:
    """
    Update a flag's status (reviewed|dismissed) within the actor's company,
    stamp resolved_by/resolved_at, and audit the action.
    """
    if new_status not in _RESOLVE_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="status must be 'reviewed' or 'dismissed'",
        )

    flag = (
        db.query(AbuseFlag)
        .filter(AbuseFlag.id == flag_id, AbuseFlag.company_id == company_id)
        .first()
    )
    if flag is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Flag not found")

    flag.status = new_status
    flag.resolved_by = actor.id
    flag.resolved_at = datetime.now(timezone.utc)

    record_audit(
        db,
        action="abuse.resolved",
        actor=actor,
        company_id=company_id,
        target_type="abuse_flag",
        target_id=flag.id,
        detail={"status": new_status, "note": note},
    )
    db.commit()
    db.refresh(flag)
    return flag
