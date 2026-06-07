# backend/routers/seats.py
"""
Seat management API (manager-only). Roster, invite, revoke, deactivate. Email
sending is stubbed: invite responses include the accept link. See Phase 3.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import Company, User
from backend.schemas import (
    InviteCreate,
    InviteResponse,
    SeatRoster,
    SeatUser,
)
from backend.services import (
    get_current_manager,
    assert_same_company,
    invite_seat,
    revoke_invite,
    deactivate_user,
    get_roster,
    serialize_invite,
)

router = APIRouter(prefix="/seats", tags=["Seats"])


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
def roster(db: Session = Depends(get_db), current_user: User = Depends(get_current_manager)):
    """Active users + pending invites + {used, limit} for the manager's company."""
    company = _manager_company(db, current_user)
    data = get_roster(db, company=company)
    return SeatRoster(
        users=[SeatUser.model_validate(u) for u in data["users"]],
        pending_invites=[InviteResponse(**serialize_invite(i)) for i in data["pending_invites"]],
        used=data["used"],
        limit=data["limit"],
    )


@router.post("/invite", response_model=InviteResponse, status_code=status.HTTP_201_CREATED)
def invite(payload: InviteCreate, db: Session = Depends(get_db),
           current_user: User = Depends(get_current_manager)):
    """Invite an agent by email. 409 if the seat limit is reached."""
    company = _manager_company(db, current_user)
    invite = invite_seat(
        db, company=company, inviter=current_user, email=payload.email, role=payload.role
    )
    return InviteResponse(**serialize_invite(invite))


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
