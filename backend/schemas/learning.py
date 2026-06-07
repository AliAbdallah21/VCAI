"""
Pydantic schemas for the learning / progress API.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel


# ── Skill constants ───────────────────────────────────────────────────────────

SKILL_KEYS: list[str] = [
    "communication",
    "product_knowledge",
    "objection_handling",
    "rapport",
    "closing",
]

SKILL_NAMES_AR: dict[str, str] = {
    "communication": "التواصل",
    "product_knowledge": "معرفة المنتج",
    "objection_handling": "التعامل مع الاعتراضات",
    "rapport": "بناء العلاقة",
    "closing": "إغلاق الصفقة",
}


# ── Learning profile ──────────────────────────────────────────────────────────

class SkillStatus(BaseModel):
    skill_key: str
    skill_name_ar: str
    current_avg: float
    trend: str                    # "improving" | "plateau" | "declining" | "insufficient_data"
    last_score: Optional[int]
    sessions_count: int


class SessionRecommendation(BaseModel):
    focus_skill: str
    focus_skill_name_ar: str
    reason: str
    recommended_persona_id: Optional[str]
    recommended_difficulty: str
    scenario_hint: str


class LearningProfile(BaseModel):
    user_id: str
    sessions_analyzed: int
    has_enough_data: bool         # False when < 2 completed sessions with scores
    weakest_skills: list[str]     # ordered weakest → strongest
    strongest_skills: list[str]   # ordered strongest → weakest
    skill_statuses: list[SkillStatus]
    recommendation: Optional[SessionRecommendation]


# ── Progress data ─────────────────────────────────────────────────────────────

class PeriodData(BaseModel):
    label: str                    # "يناير 2026", "أسبوع 3", "جلسة 5", …
    period_key: str               # sortable key, e.g. "2026-01"
    session_count: int
    overall_avg: Optional[float]
    scores: dict[str, Optional[float]]   # skill_key → avg score in this period
    focus_skill: Optional[str]    # dominant training_focus among sessions in period


class SkillSummary(BaseModel):
    skill_key: str
    skill_name_ar: str
    first_score: Optional[int]
    current_score: Optional[int]
    best_score: Optional[int]
    worst_score: Optional[int]
    total_improvement: Optional[float]  # current - first (negative = declined)
    trend: str
    focus_sessions: int           # sessions where this skill was training_focus


class ProgressData(BaseModel):
    group_by: str
    total_sessions: int
    periods: list[PeriodData]
    skill_summaries: list[SkillSummary]
    date_from: Optional[datetime]
    date_to: Optional[datetime]


# ── Insights ──────────────────────────────────────────────────────────────────

class Insight(BaseModel):
    type: str       # "improvement" | "strength" | "plateau" | "milestone" | "best_score"
    text: str       # Arabic human-readable text
    skill_key: Optional[str]
    value: Optional[float]


class InsightsResponse(BaseModel):
    insights: list[Insight]
    generated_at: datetime
