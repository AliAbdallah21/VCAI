"""
Learning & progress API router.

Endpoints:
  GET /api/learning/profile           — current skill profile + next-session recommendation
  GET /api/learning/progress          — full skill timeline (grouped by session|week|month|year)
  GET /api/learning/insights          — Arabic coaching observations
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session as DBSession

from backend.database import get_db
from backend.models import User
from backend.schemas.learning import InsightsResponse, LearningProfile, ProgressData
from backend.services import get_current_user
from backend.services.learning_service import (
    compute_learning_profile,
    get_insights,
    get_progress_data,
)

router = APIRouter(prefix="/learning", tags=["learning"])


@router.get("/profile", response_model=LearningProfile)
def profile(
    last_n: int = Query(default=5, ge=2, le=20, description="Sessions to analyse"),
    current_user: User = Depends(get_current_user),
    db: DBSession = Depends(get_db),
):
    """
    Return the user's current learning profile: skill averages, trends, and
    a recommendation for the next training session.
    """
    return compute_learning_profile(str(current_user.id), db, last_n=last_n)


@router.get("/progress", response_model=ProgressData)
def progress(
    group_by: str = Query(
        default="month",
        description="Grouping: session | week | month | year",
    ),
    current_user: User = Depends(get_current_user),
    db: DBSession = Depends(get_db),
):
    """
    Return the full skill-progress timeline grouped by the requested period.
    Use group_by=session for a per-session chart, month for a monthly overview.
    """
    return get_progress_data(str(current_user.id), db, group_by=group_by)


@router.get("/insights", response_model=InsightsResponse)
def insights(
    current_user: User = Depends(get_current_user),
    db: DBSession = Depends(get_db),
):
    """
    Return a list of Arabic coaching insights derived from the user's history.
    """
    return get_insights(str(current_user.id), db)
