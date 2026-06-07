"""
Learning service — computes learning profiles, skill-progress timelines,
and coaching insights from completed session history.

All computation is derived from the existing sessions table.
No new tables required for Phase 1.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session as DBSession

from backend.models.session import Session as SessionModel
from backend.schemas.learning import (
    SKILL_KEYS,
    SKILL_NAMES_AR,
    Insight,
    InsightsResponse,
    LearningProfile,
    ProgressData,
    SessionRecommendation,
    SkillStatus,
    SkillSummary,
    PeriodData,
)


# ── Constants ─────────────────────────────────────────────────────────────────

# Maps skill_key → recommended persona_id and scenario hint (Arabic)
_SKILL_PERSONA_MAP: dict[str, dict] = {
    "closing": {
        "persona_id": "first_time_buyer",
        "scenario_hint": "عميل مهتم لكن يحتاج إقناع للإغلاق",
        "difficulty_if_below_50": "medium",
        "difficulty_if_above_50": "hard",
    },
    "objection_handling": {
        "persona_id": "price_focused_customer",
        "scenario_hint": "عميل لديه اعتراضات متعددة على السعر والشروط",
        "difficulty_if_below_50": "medium",
        "difficulty_if_above_50": "hard",
    },
    "product_knowledge": {
        "persona_id": "detail_oriented_customer",
        "scenario_hint": "عميل يسأل أسئلة تفصيلية عن المشروع والمواصفات",
        "difficulty_if_below_50": "medium",
        "difficulty_if_above_50": "hard",
    },
    "rapport": {
        "persona_id": "difficult_customer",
        "scenario_hint": "عميل يبدأ بارداً ومتحفظاً ويحتاج بناء علاقة",
        "difficulty_if_below_50": "easy",
        "difficulty_if_above_50": "medium",
    },
    "communication": {
        "persona_id": "rushed_customer",
        "scenario_hint": "محادثة سريعة تتطلب وضوحاً ودقة في التواصل",
        "difficulty_if_below_50": "easy",
        "difficulty_if_above_50": "medium",
    },
}

_AR_MONTHS = [
    "يناير", "فبراير", "مارس", "إبريل", "مايو", "يونيو",
    "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر",
]

# Score difference threshold for trend classification
_TREND_THRESHOLD = 5.0


# ── Private helpers ───────────────────────────────────────────────────────────

def _skill_score(session: SessionModel, skill_key: str) -> Optional[int]:
    """Return the score for a skill from a session row, or None."""
    return getattr(session, f"{skill_key}_score", None)


def _strip_tz(dt: Optional[datetime]) -> Optional[datetime]:
    """Return a timezone-naive datetime so comparisons don't raise."""
    if dt is None:
        return None
    return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt


def _detect_trend(scores: list[float]) -> str:
    """
    Classify a skill's trajectory from chronological score list.

    Compares the average of the last 2 sessions against the average of all
    earlier sessions.  Requires at least 3 data points; anything less is
    reported as "insufficient_data" so the UI doesn't show misleading arrows.
    """
    if len(scores) < 3:
        return "insufficient_data"
    recent_avg = sum(scores[-2:]) / 2
    older_avg = sum(scores[:-2]) / max(len(scores) - 2, 1)
    diff = recent_avg - older_avg
    if diff > _TREND_THRESHOLD:
        return "improving"
    if diff < -_TREND_THRESHOLD:
        return "declining"
    return "plateau"


def _period_key_and_label(dt: datetime, group_by: str, session_index: int) -> tuple[str, str]:
    """Derive a sortable key and a human-readable Arabic label for a period."""
    if group_by == "session":
        return f"{session_index:05d}", f"جلسة {session_index}"
    if group_by == "week":
        year, week, _ = dt.isocalendar()
        return f"{year}-W{week:02d}", f"أسبوع {week} ({year})"
    if group_by == "year":
        return str(dt.year), str(dt.year)
    # Default: month
    return f"{dt.year}-{dt.month:02d}", f"{_AR_MONTHS[dt.month - 1]} {dt.year}"


def _build_recommendation(focus_skill: str, avg_score: float) -> SessionRecommendation:
    cfg = _SKILL_PERSONA_MAP.get(focus_skill, {})
    difficulty = (
        cfg.get("difficulty_if_below_50", "medium")
        if avg_score < 50
        else cfg.get("difficulty_if_above_50", "hard")
    )
    return SessionRecommendation(
        focus_skill=focus_skill,
        focus_skill_name_ar=SKILL_NAMES_AR[focus_skill],
        reason=f"متوسط درجتك في {SKILL_NAMES_AR[focus_skill]} هو {int(avg_score)}/100 — تحتاج تركيزاً هنا",
        recommended_persona_id=cfg.get("persona_id"),
        recommended_difficulty=difficulty,
        scenario_hint=cfg.get("scenario_hint", ""),
    )


# ── Public API ────────────────────────────────────────────────────────────────

def compute_learning_profile(
    user_id: str,
    db: DBSession,
    last_n: int = 5,
) -> LearningProfile:
    """
    Compute the current learning profile from the last `last_n` scored sessions.

    Returns a profile with weakest/strongest skill rankings, per-skill trends,
    and a session recommendation.  Returns has_enough_data=False when fewer
    than 2 completed sessions exist (no meaningful pattern yet).
    """
    sessions = (
        db.query(SessionModel)
        .filter(
            SessionModel.user_id == user_id,
            SessionModel.status == "completed",
            SessionModel.communication_score.isnot(None),
        )
        .order_by(SessionModel.ended_at.desc())
        .limit(last_n)
        .all()
    )

    if len(sessions) < 2:
        return LearningProfile(
            user_id=str(user_id),
            sessions_analyzed=len(sessions),
            has_enough_data=False,
            weakest_skills=list(SKILL_KEYS),
            strongest_skills=list(reversed(SKILL_KEYS)),
            skill_statuses=[
                SkillStatus(
                    skill_key=sk,
                    skill_name_ar=SKILL_NAMES_AR[sk],
                    current_avg=0.0,
                    trend="insufficient_data",
                    last_score=None,
                    sessions_count=0,
                )
                for sk in SKILL_KEYS
            ],
            recommendation=None,
        )

    # sessions[0] is the most recent; reverse for chronological trend detection
    chronological = list(reversed(sessions))

    skill_avgs: dict[str, float] = {}
    skill_last: dict[str, Optional[int]] = {}
    skill_trends: dict[str, str] = {}
    skill_counts: dict[str, int] = {}

    for skill in SKILL_KEYS:
        chron_scores = [
            float(score)
            for s in chronological
            if (score := _skill_score(s, skill)) is not None
        ]
        skill_avgs[skill] = sum(chron_scores) / len(chron_scores) if chron_scores else 0.0
        skill_last[skill] = _skill_score(sessions[0], skill)
        skill_trends[skill] = _detect_trend(chron_scores)
        skill_counts[skill] = len(chron_scores)

    sorted_skills = sorted(SKILL_KEYS, key=lambda sk: skill_avgs[sk])
    weakest = sorted_skills
    strongest = list(reversed(sorted_skills))

    statuses = [
        SkillStatus(
            skill_key=sk,
            skill_name_ar=SKILL_NAMES_AR[sk],
            current_avg=round(skill_avgs[sk], 1),
            trend=skill_trends[sk],
            last_score=skill_last[sk],
            sessions_count=skill_counts[sk],
        )
        for sk in SKILL_KEYS
    ]

    return LearningProfile(
        user_id=str(user_id),
        sessions_analyzed=len(sessions),
        has_enough_data=True,
        weakest_skills=weakest,
        strongest_skills=strongest,
        skill_statuses=statuses,
        recommendation=_build_recommendation(weakest[0], skill_avgs[weakest[0]]),
    )


def get_progress_data(
    user_id: str,
    db: DBSession,
    group_by: str = "month",
) -> ProgressData:
    """
    Return the full skill-progress timeline grouped by session / week / month / year.

    All completed sessions with scores are included (not capped at last_n),
    so the chart always shows the full history.
    """
    if group_by not in ("session", "week", "month", "year"):
        group_by = "month"

    sessions = (
        db.query(SessionModel)
        .filter(
            SessionModel.user_id == user_id,
            SessionModel.status == "completed",
            SessionModel.communication_score.isnot(None),
        )
        .order_by(SessionModel.ended_at.asc())
        .all()
    )

    if not sessions:
        return ProgressData(
            group_by=group_by,
            total_sessions=0,
            periods=[],
            skill_summaries=[
                SkillSummary(
                    skill_key=sk,
                    skill_name_ar=SKILL_NAMES_AR[sk],
                    first_score=None, current_score=None,
                    best_score=None, worst_score=None,
                    total_improvement=None,
                    trend="insufficient_data",
                    focus_sessions=0,
                )
                for sk in SKILL_KEYS
            ],
            date_from=None,
            date_to=None,
        )

    periods = _build_periods(sessions, group_by)
    skill_summaries = _build_skill_summaries(sessions)

    date_from = _strip_tz(sessions[0].ended_at or sessions[0].started_at)
    date_to = _strip_tz(sessions[-1].ended_at or sessions[-1].started_at)

    return ProgressData(
        group_by=group_by,
        total_sessions=len(sessions),
        periods=periods,
        skill_summaries=skill_summaries,
        date_from=date_from,
        date_to=date_to,
    )


def get_insights(user_id: str, db: DBSession) -> InsightsResponse:
    """
    Generate Arabic coaching insights from the user's full session history.

    Combines the learning profile (last 5 sessions) with the full progress
    timeline to surface meaningful patterns.
    """
    profile = compute_learning_profile(user_id, db, last_n=5)
    progress = get_progress_data(user_id, db, group_by="session")

    insights = _build_insights(profile, progress)

    return InsightsResponse(
        insights=insights,
        generated_at=datetime.utcnow(),
    )


# ── Private computation helpers ───────────────────────────────────────────────

def _build_periods(sessions: list[SessionModel], group_by: str) -> list[PeriodData]:
    groups: dict[str, list[SessionModel]] = defaultdict(list)
    labels: dict[str, str] = {}

    for idx, session in enumerate(sessions, start=1):
        dt = _strip_tz(session.ended_at or session.started_at)
        if dt is None:
            continue
        key, label = _period_key_and_label(dt, group_by, idx)
        groups[key].append(session)
        labels[key] = label

    periods: list[PeriodData] = []

    for key in sorted(groups.keys()):
        group = groups[key]
        skill_score_lists: dict[str, list[float]] = defaultdict(list)
        overall_scores: list[float] = []
        focus_counter: Counter = Counter()

        for s in group:
            for skill in SKILL_KEYS:
                score = _skill_score(s, skill)
                if score is not None:
                    skill_score_lists[skill].append(float(score))
            if s.overall_score is not None:
                overall_scores.append(float(s.overall_score))
            if s.training_focus:
                focus_counter[s.training_focus] += 1

        avg_scores: dict[str, Optional[float]] = {
            skill: (round(sum(v) / len(v), 1) if v else None)
            for skill, v in skill_score_lists.items()
        }
        for skill in SKILL_KEYS:
            avg_scores.setdefault(skill, None)

        periods.append(PeriodData(
            label=labels[key],
            period_key=key,
            session_count=len(group),
            overall_avg=round(sum(overall_scores) / len(overall_scores), 1) if overall_scores else None,
            scores=avg_scores,
            focus_skill=focus_counter.most_common(1)[0][0] if focus_counter else None,
        ))

    return periods


def _build_skill_summaries(sessions: list[SessionModel]) -> list[SkillSummary]:
    summaries: list[SkillSummary] = []

    for skill in SKILL_KEYS:
        all_scores: list[int] = [
            score
            for s in sessions
            if (score := _skill_score(s, skill)) is not None
        ]
        focus_sessions = sum(1 for s in sessions if s.training_focus == skill)

        if not all_scores:
            summaries.append(SkillSummary(
                skill_key=skill,
                skill_name_ar=SKILL_NAMES_AR[skill],
                first_score=None, current_score=None,
                best_score=None, worst_score=None,
                total_improvement=None,
                trend="insufficient_data",
                focus_sessions=focus_sessions,
            ))
            continue

        summaries.append(SkillSummary(
            skill_key=skill,
            skill_name_ar=SKILL_NAMES_AR[skill],
            first_score=all_scores[0],
            current_score=all_scores[-1],
            best_score=max(all_scores),
            worst_score=min(all_scores),
            total_improvement=float(all_scores[-1] - all_scores[0]),
            trend=_detect_trend([float(s) for s in all_scores]),
            focus_sessions=focus_sessions,
        ))

    return summaries


def _build_insights(profile: LearningProfile, progress: ProgressData) -> list[Insight]:
    insights: list[Insight] = []

    summary_by_skill = {s.skill_key: s for s in progress.skill_summaries}

    # 1. Focus → improvement: focused on skill N times and score actually went up
    for skill in SKILL_KEYS:
        summary = summary_by_skill.get(skill)
        if (summary
                and summary.focus_sessions >= 2
                and summary.total_improvement is not None
                and summary.total_improvement > 8):
            insights.append(Insight(
                type="improvement",
                text=(
                    f"ركّزت على {SKILL_NAMES_AR[skill]} في {summary.focus_sessions} جلسات"
                    f" — تحسّن بمقدار {int(summary.total_improvement)} نقطة"
                ),
                skill_key=skill,
                value=summary.total_improvement,
            ))

    # 2. Consistent strength: top skill
    if profile.has_enough_data and profile.strongest_skills:
        top = profile.strongest_skills[0]
        top_summary = summary_by_skill.get(top)
        if top_summary and top_summary.current_score is not None:
            insights.append(Insight(
                type="strength",
                text=f"{SKILL_NAMES_AR[top]} نقطة قوة ثابتة لديك (درجة {top_summary.current_score})",
                skill_key=top,
                value=float(top_summary.current_score),
            ))

    # 3. Plateau warning: stagnant skill with enough sessions to confirm it
    if progress.total_sessions >= 4:
        for skill in SKILL_KEYS:
            summary = summary_by_skill.get(skill)
            if summary and summary.trend == "plateau":
                insights.append(Insight(
                    type="plateau",
                    text=f"{SKILL_NAMES_AR[skill]} ثابت بدون تقدم ملحوظ — جرّب رفع مستوى الصعوبة",
                    skill_key=skill,
                    value=None,
                ))

    # 4. Personal best on the weakest skill (encouraging)
    if profile.has_enough_data and profile.weakest_skills:
        weakest = profile.weakest_skills[0]
        w_summary = summary_by_skill.get(weakest)
        if (w_summary
                and w_summary.current_score is not None
                and w_summary.best_score is not None
                and w_summary.current_score == w_summary.best_score
                and progress.total_sessions > 1):
            insights.append(Insight(
                type="best_score",
                text=f"أفضل أداء على الإطلاق في {SKILL_NAMES_AR[weakest]}! استمر في التركيز",
                skill_key=weakest,
                value=float(w_summary.best_score),
            ))

    # 5. Session count milestones
    for milestone in (5, 10, 20, 50, 100):
        if progress.total_sessions == milestone:
            insights.append(Insight(
                type="milestone",
                text=f"أكملت {milestone} جلسة تدريبية! إنجاز رائع — استمر",
                skill_key=None,
                value=float(milestone),
            ))

    # 6. Overall improvement since first session
    if progress.total_sessions >= 3:
        improvements = [
            s.total_improvement
            for s in progress.skill_summaries
            if s.total_improvement is not None
        ]
        if improvements:
            avg_imp = sum(improvements) / len(improvements)
            if avg_imp > 5:
                insights.append(Insight(
                    type="improvement",
                    text=f"تحسّن إجمالي منذ بدايتك: +{int(avg_imp)} نقطة في المتوسط عبر جميع المهارات",
                    skill_key=None,
                    value=avg_imp,
                ))

    return insights
