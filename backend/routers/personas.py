# backend/routers/personas.py
"""
Personas API endpoints.
"""

from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.schemas import PersonaResponse, PersonaListResponse
from backend.services import (
    get_all_personas,
    get_all_personas_for_company,
    get_personas_by_difficulty,
    get_personas_by_gender,
    get_persona,
    get_current_user_optional,
)
from backend.models import User

router = APIRouter(prefix="/personas", tags=["Personas"])


def _apply_lock_flags(db: Session, personas, current_user: Optional[User]) -> None:
    """
    Annotate each persona with a transient `locked` flag based on the caller's
    company plan. For free plans, personas outside FREE_PERSONA_IDS are locked
    so the UI can grey them out. Unauthenticated / no-company callers see
    everything unlocked (public landing/pricing preview).
    """
    from backend.plans import FREE_PERSONA_IDS
    from backend.services.session_service import _company_plan_name

    company_id = current_user.company_id if current_user else None
    plan_name = _company_plan_name(db, company_id) if company_id else None
    is_free = company_id is not None and plan_name in (None, "free")
    for p in personas:
        p.locked = is_free and p.id not in FREE_PERSONA_IDS


@router.get("", response_model=PersonaListResponse)
def list_personas(
    difficulty: Optional[str] = Query(None, description="Filter by difficulty: easy, medium, hard"),
    gender: Optional[str] = Query(None, description="Filter by gender: male, female"),
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """
    Get all available personas.

    Optionally filter by difficulty and/or gender. When the caller is an
    authenticated free-plan user, gated personas are returned with `locked:
    true` (shown greyed out in the UI) rather than hidden.
    """
    if difficulty and gender:
        personas = [
            p for p in get_personas_by_difficulty(db, difficulty)
            if p.gender == gender
        ]
        _apply_lock_flags(db, personas, current_user)
    elif difficulty:
        personas = get_personas_by_difficulty(db, difficulty)
        _apply_lock_flags(db, personas, current_user)
    elif gender:
        personas = get_personas_by_gender(db, gender)
        _apply_lock_flags(db, personas, current_user)
    elif current_user is not None:
        # Unfiltered + authenticated: use the company-scoped listing which sets
        # the lock flags from the company plan.
        personas = get_all_personas_for_company(db, current_user.company_id)
    else:
        personas = get_all_personas(db)
        _apply_lock_flags(db, personas, current_user)

    return PersonaListResponse(
        personas=[PersonaResponse.model_validate(p) for p in personas],
        total=len(personas)
    )


@router.get("/{persona_id}", response_model=PersonaResponse)
def get_persona_detail(persona_id: str, db: Session = Depends(get_db)):
    """
    Get a specific persona by ID.
    """
    persona = get_persona(db, persona_id)
    return PersonaResponse.model_validate(persona)