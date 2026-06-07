# backend/services/__init__.py
"""
Business logic services.
"""

from backend.services.auth_service import (
    verify_password,
    get_password_hash,
    create_access_token,
    decode_token,
    get_current_user,
    get_current_user_optional,
    register_user,
    authenticate_user,
    login_user,
    require_role,
    get_current_manager,
    get_current_superadmin,
    assert_same_company
)

from backend.services.session_service import (
    get_persona,
    get_all_personas,
    get_personas_by_difficulty,
    get_personas_by_gender,
    create_session,
    get_session,
    get_user_sessions,
    add_message,
    add_emotion_log,
    end_session,
    get_session_messages,
    assert_persona_allowed,
    get_all_personas_for_company,
)

from backend.services.usage_service import (
    current_period_start,
    get_or_create_period,
    sessions_used_this_period,
    session_limit_for as session_limit_for_usage,
    record_session_usage,
    assert_can_create_session,
)

from backend.services.scope_service import get_session_or_403

from backend.services.analytics_service import (
    list_agents,
    agent_progress,
    company_analytics,
    emotion_trends,
    platform_usage,
)

from backend.services.admin_service import (
    list_tenants,
    tenant_detail,
    set_tenant_status,
    global_abuse,
    global_audit,
)

from backend.services.abuse_service import (
    list_flags,
    resolve_flag,
)

from backend.services.audit_service import record_audit

from backend.services.company_service import (
    create_company_with_manager,
    provision_is_allowed,
)

from backend.services.seat_service import (
    count_active_seats,
    seat_limit_for,
    invite_seat,
    get_invite_info,
    accept_invite,
    revoke_invite,
    deactivate_user,
    get_roster,
    serialize_invite,
)

__all__ = [
    # Auth
    "verify_password",
    "get_password_hash",
    "create_access_token",
    "decode_token",
    "get_current_user",
    "get_current_user_optional",
    "register_user",
    "authenticate_user",
    "login_user",
    "require_role",
    "get_current_manager",
    "get_current_superadmin",
    "assert_same_company",

    # Session
    "get_persona",
    "get_all_personas",
    "get_personas_by_difficulty",
    "get_personas_by_gender",
    "create_session",
    "get_session",
    "get_user_sessions",
    "add_message",
    "add_emotion_log",
    "end_session",
    "get_session_messages",
    "assert_persona_allowed",
    "get_all_personas_for_company",

    # Usage metering (Phase 4)
    "current_period_start",
    "get_or_create_period",
    "sessions_used_this_period",
    "session_limit_for_usage",
    "record_session_usage",
    "assert_can_create_session",

    # Tenant scoping (Phase 4)
    "get_session_or_403",

    # Manager analytics (Phase 5)
    "list_agents",
    "agent_progress",
    "company_analytics",
    "emotion_trends",

    # Super-admin (Phase 6)
    "platform_usage",
    "list_tenants",
    "tenant_detail",
    "set_tenant_status",
    "global_abuse",
    "global_audit",

    # Abuse review (Phase 5)
    "list_flags",
    "resolve_flag",

    # Audit
    "record_audit",

    # Company / onboarding
    "create_company_with_manager",
    "provision_is_allowed",

    # Seats
    "count_active_seats",
    "seat_limit_for",
    "invite_seat",
    "get_invite_info",
    "accept_invite",
    "revoke_invite",
    "deactivate_user",
    "get_roster",
    "serialize_invite",
]