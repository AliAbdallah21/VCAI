# backend/routers/onboarding.py
"""
Onboarding API - mocked checkout (company + manager provisioning) and the public
invite accept flow. Billing is MOCKED: no payment is captured. See Phase 3.
"""

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.schemas import (
    OnboardingSignup,
    OnboardingResult,
    InviteInfo,
    InviteAccept,
    Token,
    UserResponse,
    CompanyResponse,
)
from backend.services import (
    create_company_with_manager,
    get_invite_info,
    accept_invite,
)

router = APIRouter(prefix="/onboarding", tags=["Onboarding"])


@router.post("/signup", response_model=OnboardingResult, status_code=status.HTTP_201_CREATED)
def signup(payload: OnboardingSignup, db: Session = Depends(get_db)):
    """
    Mocked-checkout completion: create a company, its manager, and a (mocked)
    subscription, then return a JWT for the manager. No payment is captured.
    """
    result = create_company_with_manager(
        db,
        company_name=payload.company_name,
        plan_name=payload.plan_name,
        billing_cycle=payload.billing_cycle,
        manager_name=payload.manager_name,
        manager_email=payload.manager_email,
        password=payload.password,
    )
    return OnboardingResult(
        access_token=result["token"],
        token_type="bearer",
        user=UserResponse.model_validate(result["manager_user"]),
        company=CompanyResponse.model_validate(result["company"]),
        plan_name=result["subscription"].plan_name,
        billing_status=result["subscription"].billing_status,
    )


@router.get("/invite/{token}", response_model=InviteInfo)
def invite_info(token: str, db: Session = Depends(get_db)):
    """Public: resolve an invite for the accept page. 404/410 if unusable."""
    return InviteInfo(**get_invite_info(db, token=token))


@router.post("/accept", response_model=Token)
def accept(payload: InviteAccept, db: Session = Depends(get_db)):
    """Public: accept an invite, creating a salesperson and returning a JWT."""
    result = accept_invite(
        db,
        token=payload.token,
        full_name=payload.full_name,
        password=payload.password,
    )
    return Token(
        access_token=result["jwt"],
        token_type="bearer",
        user=UserResponse.model_validate(result["user"]),
    )
