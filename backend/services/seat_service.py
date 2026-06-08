# backend/services/seat_service.py
"""
Seat management service - invite, accept, revoke, and deactivate seats within a
company's plan seat limit. Email sending is out of scope (stubbed): invite links
are returned/logged. See 00_ARCHITECTURE.md sections 4, 5.
"""

import logging
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from backend.config import get_settings
from backend.models import Company, Plan, SeatInvite, Subscription, User, UserStats
from backend.services.auth_service import get_password_hash, create_access_token
from backend.services.audit_service import record_audit

logger = logging.getLogger(__name__)
settings = get_settings()

INVITE_TTL_DAYS = 7


def _now() -> datetime:
    """Timezone-aware current time (columns are DateTime(timezone=True))."""
    return datetime.now(timezone.utc)


def _is_expired(expires_at) -> bool:
    """Compare an invite's expiry to now, tolerating naive timestamps from the DB."""
    if not expires_at:
        return False
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    return expires_at < _now()


def _invite_link(token: str, base_url: str | None = None) -> str:
    """Build the accept link. Prefer the caller-supplied origin (derived from the
    incoming request) so links are correct through tunnels / any serving host;
    fall back to the configured frontend_base_url when no request origin is known
    (e.g. background logging)."""
    base = (base_url or settings.frontend_base_url).rstrip("/")
    return f"{base}/invite/{token}"


# Code shown to the invitee — short, human-typeable, unambiguous. Excludes
# easily-confused chars (0/O, 1/I/L) so it can be read aloud / copied by hand.
_INVITE_CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
_INVITE_CODE_LEN = 6


def _generate_invite_code(db: Session) -> str:
    """Generate a unique 6-char alphanumeric invite code."""
    while True:
        code = "".join(secrets.choice(_INVITE_CODE_ALPHABET) for _ in range(_INVITE_CODE_LEN))
        if db.query(SeatInvite).filter(SeatInvite.invite_code == code).first() is None:
            return code


def count_active_seats(db: Session, company_id) -> int:
    """Active salesperson users in a company (managers do not consume seats)."""
    return (
        db.query(User)
        .filter(
            User.company_id == company_id,
            User.role == "salesperson",
            User.is_active == True,  # noqa: E712
        )
        .count()
    )


def seat_limit_for(db: Session, company_id) -> int:
    """Resolve the company's seat limit by JOINing subscription -> plan."""
    row = (
        db.query(Plan.seat_limit)
        .join(Subscription, Subscription.plan_name == Plan.name)
        .filter(Subscription.company_id == company_id)
        .first()
    )
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="No subscription for company")
    return int(row[0])


def invite_seat(db: Session, *, company: Company, inviter: User, email: str,
                role: str = "salesperson") -> SeatInvite:
    """Create a pending seat invite, enforcing the seat limit. 409 if full."""
    used = count_active_seats(db, company.id)
    limit = seat_limit_for(db, company.id)
    # Pending invites also tentatively occupy seats so a manager cannot
    # over-invite past the limit before anyone accepts.
    pending = (
        db.query(SeatInvite)
        .filter(SeatInvite.company_id == company.id, SeatInvite.status == "pending")
        .count()
    )
    if used + pending >= limit:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="Seat limit reached")

    invite = SeatInvite(
        company_id=company.id,
        email=email,
        role=role,
        token=secrets.token_urlsafe(32),
        invite_code=_generate_invite_code(db),
        status="pending",
        invited_by=inviter.id,
        expires_at=_now() + timedelta(days=INVITE_TTL_DAYS),
    )
    db.add(invite)
    record_audit(
        db,
        action="seat.invited",
        actor=inviter,
        company_id=company.id,
        target_type="seat_invite",
        target_id=email,
        detail={"email": email, "role": role},
    )
    db.commit()
    db.refresh(invite)

    # Email is stubbed; surface the link via logs for the demo.
    logger.info("Seat invite created for %s: %s", email, _invite_link(invite.token))
    return invite


def get_invite_info(db: Session, *, token: str) -> dict:
    """Public lookup for the accept page. 404 if missing, 410 if unusable."""
    invite = db.query(SeatInvite).filter(SeatInvite.token == token).first()
    if invite is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invite not found")
    if invite.status != "pending":
        raise HTTPException(status_code=status.HTTP_410_GONE, detail="Invite is no longer valid")
    if _is_expired(invite.expires_at):
        invite.status = "expired"
        db.commit()
        raise HTTPException(status_code=status.HTTP_410_GONE, detail="Invite has expired")

    company = db.query(Company).filter(Company.id == invite.company_id).first()
    return {
        "company_name": company.name if company else "",
        "email": invite.email,
        "role": invite.role,
        "status": invite.status,
        "expires_at": invite.expires_at,
    }


def accept_invite(db: Session, *, token: str, full_name: str, password: str) -> dict:
    """Validate a pending invite and create the salesperson. Returns {user, jwt}."""
    invite = db.query(SeatInvite).filter(SeatInvite.token == token).first()
    if invite is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invite not found")
    if invite.status != "pending":
        raise HTTPException(status_code=status.HTTP_410_GONE, detail="Invite is no longer valid")
    if _is_expired(invite.expires_at):
        invite.status = "expired"
        db.commit()
        raise HTTPException(status_code=status.HTTP_410_GONE, detail="Invite has expired")

    if db.query(User).filter(User.email == invite.email).first() is not None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Email already registered")

    # Race guard: re-check the seat limit at acceptance time, counting only
    # active users (this pending invite is about to convert to a user).
    used = count_active_seats(db, invite.company_id)
    limit = seat_limit_for(db, invite.company_id)
    if used >= limit:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="Seat limit reached")

    try:
        user = User(
            email=invite.email,
            password_hash=get_password_hash(password),
            full_name=full_name,
            company_id=invite.company_id,
            role=invite.role or "salesperson",
        )
        db.add(user)
        db.flush()
        db.add(UserStats(user_id=user.id))

        invite.status = "accepted"
        record_audit(
            db,
            action="seat.accepted",
            actor=user,
            company_id=invite.company_id,
            target_type="user",
            target_id=user.id,
            detail={"email": invite.email},
        )
        db.commit()
    except HTTPException:
        db.rollback()
        raise
    except Exception:
        db.rollback()
        raise

    db.refresh(user)
    return {"user": user, "jwt": create_access_token(str(user.id))}


def _resolve_usable_invite_by_code(db: Session, *, code: str) -> SeatInvite:
    """Look up a pending, unexpired invite by its short code. Raises 404/410.

    Codes are matched case-insensitively so a hand-typed lowercase code still
    works (they are stored uppercase from _INVITE_CODE_ALPHABET).
    """
    normalized = (code or "").strip().upper()
    invite = db.query(SeatInvite).filter(SeatInvite.invite_code == normalized).first()
    if invite is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="Invalid or expired invite code")
    if invite.status != "pending":
        raise HTTPException(status_code=status.HTTP_410_GONE,
                            detail="Invalid or expired invite code")
    if _is_expired(invite.expires_at):
        invite.status = "expired"
        db.commit()
        raise HTTPException(status_code=status.HTTP_410_GONE,
                            detail="Invalid or expired invite code")
    return invite


def attach_user_to_company(db: Session, *, user: User, invite: SeatInvite) -> User:
    """Attach an EXISTING user to the invite's company, consuming the invite.

    Used when an already-registered (solo) user joins via a code. Enforces the
    seat limit and that the user is not already in a company. The user's email
    need NOT match the invite's target email (the manager invited a placeholder
    or a different address).
    """
    if user.company_id is not None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="You already belong to a company")

    used = count_active_seats(db, invite.company_id)
    limit = seat_limit_for(db, invite.company_id)
    if used >= limit:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="Seat limit reached")

    company = db.query(Company).filter(Company.id == invite.company_id).first()
    if company is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Company not found")

    try:
        user.company_id = invite.company_id
        user.company = company.name
        user.role = invite.role or "salesperson"
        invite.status = "accepted"
        record_audit(
            db,
            action="seat.accepted",
            actor=user,
            company_id=invite.company_id,
            target_type="user",
            target_id=user.id,
            detail={"email": user.email, "via": "code"},
        )
        db.commit()
    except HTTPException:
        db.rollback()
        raise
    except Exception:
        db.rollback()
        raise

    db.refresh(user)
    return user


def join_company_by_code(db: Session, *, user: User, code: str) -> User:
    """Existing logged-in user joins a company by pasting a 6-char invite code."""
    invite = _resolve_usable_invite_by_code(db, code=code)
    return attach_user_to_company(db, user=user, invite=invite)


def validate_invite_code(db: Session, *, code: str) -> SeatInvite:
    """Validate a code during registration. Returns the usable invite or raises.

    The register handler calls this BEFORE creating the user so an invalid code
    rejects the whole registration instead of silently creating a solo account.
    """
    return _resolve_usable_invite_by_code(db, code=code)


def consume_invite_for_new_user(db: Session, *, user: User, invite: SeatInvite) -> None:
    """Mark an invite accepted for a brand-new user created during registration.

    The User row is created + committed by register_user with company_id already
    set; here we only flip the invite to accepted and audit, in the same session.
    """
    invite.status = "accepted"
    record_audit(
        db,
        action="seat.accepted",
        actor=user,
        company_id=invite.company_id,
        target_type="user",
        target_id=user.id,
        detail={"email": user.email, "via": "register_code"},
    )
    db.commit()


def revoke_invite(db: Session, *, company: Company, inviter: User, invite_id) -> SeatInvite:
    """Revoke a pending invite for this company."""
    invite = (
        db.query(SeatInvite)
        .filter(SeatInvite.id == invite_id, SeatInvite.company_id == company.id)
        .first()
    )
    if invite is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invite not found")
    if invite.status != "pending":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Only pending invites can be revoked")

    invite.status = "revoked"
    record_audit(
        db,
        action="seat.revoked",
        actor=inviter,
        company_id=company.id,
        target_type="seat_invite",
        target_id=invite.id,
        detail={"email": invite.email},
    )
    db.commit()
    db.refresh(invite)
    return invite


def deactivate_user(db: Session, *, company: Company, actor: User, user_id) -> User:
    """Deactivate a user (frees a seat). Never hard-delete."""
    user = (
        db.query(User)
        .filter(User.id == user_id, User.company_id == company.id)
        .first()
    )
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    if user.id == actor.id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="You cannot deactivate yourself")

    user.is_active = False
    record_audit(
        db,
        action="seat.deactivated",
        actor=actor,
        company_id=company.id,
        target_type="user",
        target_id=user.id,
        detail={"email": user.email},
    )
    db.commit()
    db.refresh(user)
    return user


def get_roster(db: Session, *, company: Company) -> dict:
    """Active users + pending invites + {used, limit} for the company."""
    users = (
        db.query(User)
        .filter(User.company_id == company.id, User.is_active == True)  # noqa: E712
        .order_by(User.created_at.asc())
        .all()
    )
    pending = (
        db.query(SeatInvite)
        .filter(SeatInvite.company_id == company.id, SeatInvite.status == "pending")
        .order_by(SeatInvite.created_at.desc())
        .all()
    )
    return {
        "users": users,
        "pending_invites": pending,
        "used": count_active_seats(db, company.id),
        "limit": seat_limit_for(db, company.id),
    }


def serialize_invite(invite: SeatInvite, base_url: str | None = None) -> dict:
    """Shape a SeatInvite for InviteResponse (adds the stubbed invite link + code).

    Pass the request origin as base_url so the link matches the serving host.
    """
    return {
        "id": invite.id,
        "email": invite.email,
        "role": invite.role,
        "status": invite.status,
        "token": invite.token,
        "invite_code": invite.invite_code,
        "invite_link": _invite_link(invite.token, base_url),
        "expires_at": invite.expires_at,
        "created_at": invite.created_at,
    }
