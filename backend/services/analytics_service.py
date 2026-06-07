# backend/services/analytics_service.py
"""
Manager analytics aggregation service (Phase 5).

All rollups are computed with SQLAlchemy aggregate queries scoped by company_id
(never Python loops over every row). A company's sessions are reached via the
session owner's company_id (sessions.user_id -> users.company_id). See
00_ARCHITECTURE.md sections 3, 5, 6, 10.
"""

from datetime import date, datetime, timedelta, timezone
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import func, case, cast, Date
from sqlalchemy.orm import Session as DbSession

from backend.models import (
    User,
    UserStats,
    Session as TrainingSession,
    EmotionLog,
    SeatInvite,
    Persona,
    Company,
    Subscription,
    UsagePeriod,
    AbuseFlag,
)

# Per-skill score columns on the sessions table.
SKILL_COLUMNS = {
    "communication": TrainingSession.communication_score,
    "product_knowledge": TrainingSession.product_knowledge_score,
    "objection_handling": TrainingSession.objection_handling_score,
    "rapport": TrainingSession.rapport_score,
    "closing": TrainingSession.closing_score,
}

# Customer emotions treated as "ended badly" for the risk-share metric.
_HIGH_RISK_EMOTIONS = {"angry", "frustrated", "annoyed"}


def _current_period_start() -> date:
    return date.today().replace(day=1)


def _company_user_ids(db: DbSession, company_id) -> list:
    """All user ids belonging to a company (active or not)."""
    rows = db.query(User.id).filter(User.company_id == company_id).all()
    return [r[0] for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Agents roster
# ─────────────────────────────────────────────────────────────────────────────

def list_agents(db: DbSession, company_id) -> dict:
    """
    Company agents (salespersons) with their summary stats from user_stats.
    Managers are excluded from the agent list. Includes pending-invite count.
    """
    rows = (
        db.query(User, UserStats)
        .outerjoin(UserStats, UserStats.user_id == User.id)
        .filter(User.company_id == company_id, User.role == "salesperson")
        .order_by(User.created_at.asc())
        .all()
    )

    agents = []
    for user, stats in rows:
        agents.append({
            "user_id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "is_active": user.is_active,
            "total_sessions": (stats.total_sessions if stats else 0) or 0,
            "completed_sessions": (stats.completed_sessions if stats else 0) or 0,
            "avg_overall_score": stats.avg_overall_score if stats else None,
            "current_streak": (stats.current_streak if stats else 0) or 0,
            "last_session_date": stats.last_session_date if stats else None,
        })

    pending_invites = (
        db.query(func.count(SeatInvite.id))
        .filter(SeatInvite.company_id == company_id, SeatInvite.status == "pending")
        .scalar()
    ) or 0

    return {"agents": agents, "pending_invites": pending_invites}


# ─────────────────────────────────────────────────────────────────────────────
# Per-agent progress
# ─────────────────────────────────────────────────────────────────────────────

def agent_progress(db: DbSession, company_id, user_id: UUID) -> dict:
    """
    One agent's per-skill averages, score trend, and session list. The agent
    must belong to `company_id` (else 404). skill_profiles are Phase 8; an empty
    list is returned until then.
    """
    agent = (
        db.query(User)
        .filter(User.id == user_id, User.company_id == company_id)
        .first()
    )
    if agent is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")

    # Per-skill averages (single aggregate query).
    skill_avgs = db.query(
        *[func.avg(col).label(name) for name, col in SKILL_COLUMNS.items()]
    ).filter(TrainingSession.user_id == user_id).first()
    skill_averages = {
        name: (round(float(val), 1) if val is not None else None)
        for name, val in zip(SKILL_COLUMNS.keys(), skill_avgs)
    }

    # Score trend: completed sessions with an overall score, oldest first.
    trend_rows = (
        db.query(TrainingSession.started_at, TrainingSession.overall_score)
        .filter(
            TrainingSession.user_id == user_id,
            TrainingSession.overall_score.isnot(None),
        )
        .order_by(TrainingSession.started_at.asc())
        .all()
    )
    score_trend = [
        {"date": started.isoformat() if started else None, "overall_score": score}
        for started, score in trend_rows
    ]

    # Session list (most recent first), with persona name. Explicit columns
    # only (never the full ORM entity) so optional columns are not loaded.
    session_rows = (
        db.query(
            TrainingSession.id,
            TrainingSession.persona_id,
            Persona.name_ar,
            TrainingSession.status,
            TrainingSession.difficulty,
            TrainingSession.started_at,
            TrainingSession.duration_seconds,
            TrainingSession.overall_score,
            TrainingSession.turn_count,
        )
        .outerjoin(Persona, Persona.id == TrainingSession.persona_id)
        .filter(TrainingSession.user_id == user_id)
        .order_by(TrainingSession.started_at.desc())
        .limit(100)
        .all()
    )
    sessions = [
        {
            "id": r.id,
            "persona_id": r.persona_id,
            "persona_name": r.name_ar,
            "status": r.status,
            "difficulty": r.difficulty,
            "started_at": r.started_at,
            "duration_seconds": r.duration_seconds,
            "overall_score": r.overall_score,
            "turn_count": r.turn_count or 0,
        }
        for r in session_rows
    ]

    return {
        "user_id": agent.id,
        "full_name": agent.full_name,
        "email": agent.email,
        "is_active": agent.is_active,
        "skill_averages": skill_averages,
        "score_trend": score_trend,
        "sessions": sessions,
        "skill_profiles": [],  # Phase 8 populates skill_profiles; tolerate empty.
    }


# ─────────────────────────────────────────────────────────────────────────────
# Company analytics
# ─────────────────────────────────────────────────────────────────────────────

def company_analytics(db: DbSession, company_id) -> dict:
    """Company-level rollups. All aggregates are scoped by company_id via SQL.

    Every query selects explicit columns/aggregates (never the full ORM entity),
    so the rollups stay correct and avoid loading optional columns.
    """
    user_ids = _company_user_ids(db, company_id)

    if not user_ids:
        return {
            "total_sessions": 0,
            "sessions_this_period": 0,
            "active_agents": 0,
            "avg_score": None,
            "score_distribution": [],
            "sessions_per_day": [],
            "persona_usage": [],
            "weakest_skill": None,
        }

    total_sessions = (
        db.query(func.count(TrainingSession.id))
        .filter(TrainingSession.user_id.in_(user_ids))
        .scalar()
    ) or 0

    period_start = _current_period_start()
    sessions_this_period = (
        db.query(func.count(TrainingSession.id))
        .filter(
            TrainingSession.user_id.in_(user_ids),
            cast(TrainingSession.started_at, Date) >= period_start,
        )
        .scalar()
    ) or 0

    active_agents = (
        db.query(func.count(User.id))
        .filter(
            User.company_id == company_id,
            User.role == "salesperson",
            User.is_active == True,  # noqa: E712
        )
        .scalar()
    ) or 0

    avg_score = (
        db.query(func.avg(TrainingSession.overall_score))
        .filter(
            TrainingSession.user_id.in_(user_ids),
            TrainingSession.overall_score.isnot(None),
        )
        .scalar()
    )
    avg_score = round(float(avg_score), 1) if avg_score is not None else None

    # Score distribution in 20-point buckets (0-19, 20-39, ... 80-100).
    bucket = case(
        (TrainingSession.overall_score < 20, "0-19"),
        (TrainingSession.overall_score < 40, "20-39"),
        (TrainingSession.overall_score < 60, "40-59"),
        (TrainingSession.overall_score < 80, "60-79"),
        else_="80-100",
    )
    dist_rows = (
        db.query(bucket.label("bucket"), func.count(TrainingSession.id))
        .filter(
            TrainingSession.user_id.in_(user_ids),
            TrainingSession.overall_score.isnot(None),
        )
        .group_by(bucket)
        .all()
    )
    dist_map = {b: c for b, c in dist_rows}
    score_distribution = [
        {"bucket": b, "count": dist_map.get(b, 0)}
        for b in ["0-19", "20-39", "40-59", "60-79", "80-100"]
    ]

    # Sessions per day, last 30 days.
    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).date()
    day = cast(TrainingSession.started_at, Date).label("day")
    spd_rows = (
        db.query(day, func.count(TrainingSession.id))
        .filter(
            TrainingSession.user_id.in_(user_ids),
            cast(TrainingSession.started_at, Date) >= cutoff,
        )
        .group_by(day)
        .order_by(day)
        .all()
    )
    sessions_per_day = [
        {"date": d.isoformat() if hasattr(d, "isoformat") else str(d), "count": c}
        for d, c in spd_rows
    ]

    # Persona usage (most/least practised) with names.
    persona_rows = (
        db.query(
            TrainingSession.persona_id,
            Persona.name_ar,
            func.count(TrainingSession.id).label("count"),
        )
        .outerjoin(Persona, Persona.id == TrainingSession.persona_id)
        .filter(TrainingSession.user_id.in_(user_ids))
        .group_by(TrainingSession.persona_id, Persona.name_ar)
        .order_by(func.count(TrainingSession.id).desc())
        .all()
    )
    persona_usage = [
        {"persona_id": pid, "persona_name": name, "count": cnt}
        for pid, name, cnt in persona_rows
    ]

    # Weakest team skill: the skill with the lowest average across the company.
    skill_avgs = db.query(
        *[func.avg(col).label(name) for name, col in SKILL_COLUMNS.items()]
    ).filter(TrainingSession.user_id.in_(user_ids)).first()
    team_skill_averages = {
        name: (round(float(val), 1) if val is not None else None)
        for name, val in zip(SKILL_COLUMNS.keys(), skill_avgs)
    }
    scored = {k: v for k, v in team_skill_averages.items() if v is not None}
    weakest_skill = (
        {"skill": min(scored, key=scored.get), "average": scored[min(scored, key=scored.get)]}
        if scored else None
    )

    return {
        "total_sessions": total_sessions,
        "sessions_this_period": sessions_this_period,
        "active_agents": active_agents,
        "avg_score": avg_score,
        "score_distribution": score_distribution,
        "sessions_per_day": sessions_per_day,
        "persona_usage": persona_usage,
        "team_skill_averages": team_skill_averages,
        "weakest_skill": weakest_skill,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Emotion trends
# ─────────────────────────────────────────────────────────────────────────────

def emotion_trends(db: DbSession, company_id) -> dict:
    """
    Aggregate emotion_logs across the company: emotion distribution, average
    mood, and the share of sessions whose LAST emotion is high-risk.
    """
    user_ids = _company_user_ids(db, company_id)
    if not user_ids:
        return {
            "emotion_distribution": [],
            "avg_mood_score": None,
            "high_risk_session_share": 0.0,
            "total_sessions_with_emotion": 0,
        }

    # emotion_logs joined to their session, scoped to the company's users.
    logs_q = (
        db.query(EmotionLog)
        .join(TrainingSession, TrainingSession.id == EmotionLog.session_id)
        .filter(TrainingSession.user_id.in_(user_ids))
    )

    dist_rows = (
        logs_q.with_entities(EmotionLog.customer_emotion, func.count(EmotionLog.id))
        .group_by(EmotionLog.customer_emotion)
        .order_by(func.count(EmotionLog.id).desc())
        .all()
    )
    emotion_distribution = [
        {"emotion": emo, "count": cnt} for emo, cnt in dist_rows
    ]

    avg_mood = logs_q.with_entities(func.avg(EmotionLog.customer_mood_score)).scalar()
    avg_mood = round(float(avg_mood), 1) if avg_mood is not None else None

    # High-risk share: of sessions that have emotion logs, how many end in a
    # high-risk emotion. The "last" emotion per session is the one with the
    # max created_at. Computed in SQL via a per-session latest-id subquery.
    latest_per_session = (
        db.query(
            EmotionLog.session_id.label("sid"),
            func.max(EmotionLog.created_at).label("last_at"),
        )
        .join(TrainingSession, TrainingSession.id == EmotionLog.session_id)
        .filter(TrainingSession.user_id.in_(user_ids))
        .group_by(EmotionLog.session_id)
        .subquery()
    )
    last_emotions = (
        db.query(EmotionLog.customer_emotion)
        .join(
            latest_per_session,
            (EmotionLog.session_id == latest_per_session.c.sid)
            & (EmotionLog.created_at == latest_per_session.c.last_at),
        )
        .all()
    )
    total_with_emotion = len(last_emotions)
    high_risk = sum(1 for (emo,) in last_emotions if emo in _HIGH_RISK_EMOTIONS)
    high_risk_share = round(high_risk / total_with_emotion, 3) if total_with_emotion else 0.0

    return {
        "emotion_distribution": emotion_distribution,
        "avg_mood_score": avg_mood,
        "high_risk_session_share": high_risk_share,
        "total_sessions_with_emotion": total_with_emotion,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Platform-wide rollups (Phase 6, superadmin only — NO company scope)
# ─────────────────────────────────────────────────────────────────────────────

def platform_usage(db: DbSession) -> dict:
    """
    Cross-tenant rollups for the super-admin dashboard. Platform-wide (no company
    scope); the router restricts access to superadmin. All aggregates are SQL.
    """
    total_companies = db.query(func.count(Company.id)).scalar() or 0
    active_companies = (
        db.query(func.count(Company.id)).filter(Company.is_active == True).scalar()  # noqa: E712
    ) or 0

    # Active subscriptions by plan (active/trial count as active).
    plan_rows = (
        db.query(Subscription.plan_name, func.count(Subscription.id))
        .filter(Subscription.billing_status.in_(["active", "trial"]))
        .group_by(Subscription.plan_name)
        .all()
    )
    active_subs_by_plan = [{"plan": p, "count": c} for p, c in plan_rows]
    active_subscriptions = sum(c for _, c in plan_rows)

    total_sessions = db.query(func.count(TrainingSession.id)).scalar() or 0

    # Platform sessions per day, last 30 days.
    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).date()
    day = cast(TrainingSession.started_at, Date).label("day")
    spd_rows = (
        db.query(day, func.count(TrainingSession.id))
        .filter(cast(TrainingSession.started_at, Date) >= cutoff)
        .group_by(day)
        .order_by(day)
        .all()
    )
    sessions_per_day = [
        {"date": d.isoformat() if hasattr(d, "isoformat") else str(d), "count": c}
        for d, c in spd_rows
    ]

    # Sessions this period (current month) platform-wide.
    period_start = _current_period_start()
    sessions_this_period = (
        db.query(func.coalesce(func.sum(UsagePeriod.sessions_used), 0))
        .filter(UsagePeriod.period_start == period_start)
        .scalar()
    ) or 0

    # Seat utilisation: active salesperson seats platform-wide.
    total_active_agents = (
        db.query(func.count(User.id))
        .filter(User.role == "salesperson", User.is_active == True)  # noqa: E712
        .scalar()
    ) or 0

    open_flags = (
        db.query(func.count(AbuseFlag.id)).filter(AbuseFlag.status == "open").scalar()
    ) or 0

    # Top tenants by total session count.
    top_rows = (
        db.query(Company.id, Company.name, func.count(TrainingSession.id).label("sessions"))
        .join(User, User.company_id == Company.id)
        .join(TrainingSession, TrainingSession.user_id == User.id)
        .group_by(Company.id, Company.name)
        .order_by(func.count(TrainingSession.id).desc())
        .limit(10)
        .all()
    )
    top_tenants = [
        {"company_id": cid, "name": name, "sessions": sessions}
        for cid, name, sessions in top_rows
    ]

    return {
        "total_companies": total_companies,
        "active_companies": active_companies,
        "active_subscriptions": active_subscriptions,
        "active_subs_by_plan": active_subs_by_plan,
        "total_sessions": total_sessions,
        "sessions_this_period": int(sessions_this_period),
        "total_active_agents": total_active_agents,
        "open_abuse_flags": open_flags,
        "sessions_per_day": sessions_per_day,
        "top_tenants": top_tenants,
    }
