# backend/routers/personas.py
"""
Personas API endpoints.
"""

from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.schemas import PersonaResponse, PersonaListResponse
from backend.services import get_all_personas, get_personas_by_difficulty, get_persona

router = APIRouter(prefix="/personas", tags=["Personas"])


@router.get("", response_model=PersonaListResponse)
def list_personas(
    difficulty: Optional[str] = Query(None, description="Filter by difficulty: easy, medium, hard"),
    db: Session = Depends(get_db)
):
    """
    Get all available personas.
    
    Optionally filter by difficulty level.
    """
    if difficulty:
        personas = get_personas_by_difficulty(db, difficulty)
    else:
        personas = get_all_personas(db)
    
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