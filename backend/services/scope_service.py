# backend/services/scope_service.py
"""
Tenant-scoping helper for session-owned resources (Phase 4).

A single chokepoint, get_session_or_403, applies the role rules from
00_ARCHITECTURE.md section 3 to any session lookup so the checks are not
duplicated across routers:

  - salesperson : may only access their OWN sessions (session.user_id == me).
  - manager     : may access any session within their OWN company.
  - superadmin  : unrestricted.

The owning company is resolved via the session owner's company_id. Anything
outside scope raises 404 (not 403) so a salesperson cannot probe which session
ids exist in another tenant - existence is not leaked across the boundary.
"""

from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy.orm import Session as DbSession

from backend.models import Session as TrainingSession, User

# 404 for cross-scope reads: do not leak whether the id exists in another tenant.
_NOT_FOUND = HTTPException(
    status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
)


def get_session_or_403(
    db: DbSession, session_id: UUID, current_user: User
) -> TrainingSession:
    """
    Fetch a session and enforce the caller's access scope, or raise 404.

    Use this everywhere a session (or its messages/emotion logs/checkpoints/
    stats/evaluation) is read or written, instead of re-implementing the
    ownership checks per endpoint.
    """
    session = (
        db.query(TrainingSession)
        .filter(TrainingSession.id == session_id)
        .first()
    )
    if session is None:
        raise _NOT_FOUND

    # Superadmin: unrestricted.
    if current_user.role == "superadmin":
        return session

    # Salesperson: own sessions only.
    if current_user.role == "salesperson":
        if session.user_id != current_user.id:
            raise _NOT_FOUND
        return session

    # Manager: any session within the manager's own company. Resolve the
    # owning company via the session's user.
    if current_user.role == "manager":
        owner = db.query(User).filter(User.id == session.user_id).first()
        owner_company_id = owner.company_id if owner else None
        if (
            owner_company_id is None
            or current_user.company_id is None
            or str(owner_company_id) != str(current_user.company_id)
        ):
            raise _NOT_FOUND
        return session

    # Any other/unknown role has no access.
    raise _NOT_FOUND
