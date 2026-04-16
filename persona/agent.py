# persona/agent.py
"""
Persona agent — loads persona data from the PostgreSQL database.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from shared.types import Persona

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

_log = logging.getLogger(__name__)


# Map persona difficulty → sensible default starting emotion for the LLM
_DIFFICULTY_EMOTION = {
    "easy":   "friendly",
    "medium": "neutral",
    "hard":   "hesitant",
}


def get_persona(persona_id: str, db_session: "Session") -> Persona:
    """
    Load a persona from the personas table and return it as a Persona TypedDict.

    Args:
        persona_id:  Primary key of the persona (e.g. "difficult_customer").
        db_session:  Active SQLAlchemy Session.

    Returns:
        Persona TypedDict with all fields populated.

    Raises:
        ValueError: if persona_id is not found in the personas table.
    """
    from backend.models.persona import Persona as PersonaModel

    row = db_session.query(PersonaModel).filter(PersonaModel.id == persona_id).first()
    if row is None:
        raise ValueError(
            f"Persona '{persona_id}' not found in the personas table. "
            "Run scripts/seed_personas.py to populate it."
        )

    return Persona(
        id=row.id,
        name=row.name_ar or row.name_en,
        name_en=row.name_en,
        description=row.description_ar or row.description_en or "",
        personality_prompt=row.personality_prompt,
        voice_id=row.voice_id or "",
        default_emotion=_DIFFICULTY_EMOTION.get(row.difficulty, "neutral"),
        difficulty=row.difficulty,
        traits=row.traits or [],
        avatar_url=row.avatar_url,
    )
