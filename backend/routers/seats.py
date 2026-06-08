# backend/routers/seats.py
"""
Seat management API (manager-only). Roster, invite, revoke, deactivate. Email
sending is stubbed: invite responses include the accept link. See Phase 3.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import Company, User
from backend.schemas import (
    InviteCreate,
    InviteResponse,
    JoinByCode,
    SeatRoster,
    SeatUser,
    UserResponse,
)
from backend.services import (
    get_current_manager,
    get_current_user,
    assert_same_company,
    invite_seat,
    join_company_by_code,
    revoke_invite,
    deactivate_user,
    get_roster,
    serialize_invite,
)

router = APIRouter(prefix="/seats", tags=["Seats"])


def _request_origin(request: Request) -> str:
    """Origin (scheme://host[:port]) of the incoming request, honoring proxy
    headers so invite links are correct behind a tunnel / reverse proxy. Falls
    back to the request's own base_url, then (in serialize) to frontend_base_url."""
    forwarded_host = request.headers.get("x-forwarded-host")
    if forwarded_host:
        proto = request.headers.get("x-forwarded-proto", request.url.scheme)
        return f"{proto}://{forwarded_host}"
    # request.base_url is like "http://host:8000/" — strip the trailing slash.
    return str(request.base_url).rstrip("/")


def _manager_company(db: Session, current_user: User) -> Company:
    """Resolve and authorize the manager's own company."""
    if current_user.company_id is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="User is not attached to a company")
    assert_same_company(current_user, current_user.company_id)
    company = db.query(Company).filter(Company.id == current_user.company_id).first()
    if company is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Company not found")
    return company


@router.get("", response_model=SeatRoster)
def roster(request: Request, db: Session = Depends(get_db),
           current_user: User = Depends(get_current_manager)):
    """Active users + pending invites + {used, limit} for the manager's company."""
    company = _manager_company(db, current_user)
    data = get_roster(db, company=company)
    origin = _request_origin(request)
    return SeatRoster(
        users=[SeatUser.model_validate(u) for u in data["users"]],
        pending_invites=[InviteResponse(**serialize_invite(i, origin)) for i in data["pending_invites"]],
        used=data["used"],
        limit=data["limit"],
    )


@router.post("/invite", response_model=InviteResponse, status_code=status.HTTP_201_CREATED)
def invite(payload: InviteCreate, request: Request, db: Session = Depends(get_db),
           current_user: User = Depends(get_current_manager)):
    """Invite an agent by email. 409 if the seat limit is reached."""
    company = _manager_company(db, current_user)
    invite = invite_seat(
        db, company=company, inviter=current_user, email=payload.email, role=payload.role
    )
    return InviteResponse(**serialize_invite(invite, _request_origin(request)))


@router.post("/join", response_model=UserResponse)
def join(payload: JoinByCode, db: Session = Depends(get_db),
         current_user: User = Depends(get_current_user)):
    """Existing logged-in user joins a company by pasting a 6-char invite code.

    Attaches the current account to the invite's company (no new user). 404/410
    on a bad/expired code, 400 if already in a company, 409 if the seat is full.
    """
    user = join_company_by_code(db, user=current_user, code=payload.code)
    return UserResponse.model_validate(user)


@router.delete("/invite/{invite_id}", response_model=InviteResponse)
def revoke(invite_id: UUID, db: Session = Depends(get_db),
           current_user: User = Depends(get_current_manager)):
    """Revoke a pending invite."""
    company = _manager_company(db, current_user)
    invite = revoke_invite(db, company=company, inviter=current_user, invite_id=invite_id)
    return InviteResponse(**serialize_invite(invite))


@router.post("/{user_id}/deactivate", response_model=SeatUser)
def deactivate(user_id: UUID, db: Session = Depends(get_db),
               current_user: User = Depends(get_current_manager)):
    """Deactivate an agent, freeing a seat. Never hard-deletes."""
    company = _manager_company(db, current_user)
    user = deactivate_user(db, company=company, actor=current_user, user_id=user_id)
    return SeatUser.model_validate(user)
