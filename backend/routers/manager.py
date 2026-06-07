# backend/routers/manager.py
"""
Manager / supervisor dashboard API (manager auth). Roster, per-agent progress,
company analytics, emotion trends, and the abuse-flag review queue.

All endpoints scope to the manager's own company_id. A superadmin may pass
`?company_id=` to view any company. See 00_ARCHITECTURE.md sections 3, 6, 10.
"""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import Company, User
from backend.schemas import (
    AgentListResponse,
    AgentProgress,
    CompanyAnalytics,
    EmotionTrends,
    AbuseFlagOut,
    AbuseResolve,
)
from backend.services import (
    get_current_manager,
    assert_same_company,
    list_agents,
    agent_progress,
    company_analytics,
    emotion_trends,
    list_flags,
    resolve_flag,
)

router = APIRouter(prefix="/manager", tags=["Manager"])


def _resolve_company_id(
    db: Session, current_user: User, company_id: Optional[UUID]
) -> UUID:
    """
    Resolve the company to operate on:
      - superadmin may target any company via ?company_id=, else their own (must
        provide one since superadmin.company_id may be NULL).
      - manager is locked to their own company; a mismatched ?company_id= is 403.
    """
    if current_user.role == "superadmin":
        target = company_id or current_user.company_id
        if target is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Superadmin must specify company_id",
            )
        if db.query(Company).filter(Company.id == target).first() is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Company not found")
        return target

    # Manager (or any non-superadmin allowed through get_current_manager).
    if current_user.company_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User is not attached to a company",
        )
    if company_id is not None:
        # Reject cross-company access explicitly (403, not silent rescope).
        assert_same_company(current_user, company_id)
    return current_user.company_id


@router.get("/agents", response_model=AgentListResponse)
def agents(
    company_id: Optional[UUID] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_manager),
):
    """List the company's agents with summary stats + pending-invite count."""
    cid = _resolve_company_id(db, current_user, company_id)
    return list_agents(db, cid)


@router.get("/agents/{user_id}/progress", response_model=AgentProgress)
def agent_detail(
    user_id: UUID,
    company_id: Optional[UUID] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_manager),
):
    """One agent's per-skill averages, score trend, and session history."""
    cid = _resolve_company_id(db, current_user, company_id)
    return agent_progress(db, cid, user_id)


@router.get("/analytics", response_model=CompanyAnalytics)
def analytics(
    company_id: Optional[UUID] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_manager),
):
    """Company-level rollups for the overview tab."""
    cid = _resolve_company_id(db, current_user, company_id)
    return company_analytics(db, cid)


@router.get("/emotion-trends", response_model=EmotionTrends)
def emotion_trends_endpoint(
    company_id: Optional[UUID] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_manager),
):
    """Company-level emotion aggregates from emotion_logs."""
    cid = _resolve_company_id(db, current_user, company_id)
    return emotion_trends(db, cid)


@router.get("/abuse", response_model=list[AbuseFlagOut])
def abuse(
    status_filter: Optional[str] = Query(None, alias="status"),
    company_id: Optional[UUID] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_manager),
):
    """Abuse flags for the company, newest first; filter by ?status=."""
    cid = _resolve_company_id(db, current_user, company_id)
    flags = list_flags(db, cid, status_filter=status_filter)
    return [AbuseFlagOut.model_validate(f) for f in flags]


@router.post("/abuse/{flag_id}/resolve", response_model=AbuseFlagOut)
def resolve_abuse(
    flag_id: UUID,
    payload: AbuseResolve,
    company_id: Optional[UUID] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_manager),
):
    """Mark a flag reviewed/dismissed; stamps resolver + audits the action."""
    cid = _resolve_company_id(db, current_user, company_id)
    flag = resolve_flag(
        db,
        company_id=cid,
        actor=current_user,
        flag_id=flag_id,
        new_status=payload.status,
        note=payload.note,
    )
    return AbuseFlagOut.model_validate(flag)
