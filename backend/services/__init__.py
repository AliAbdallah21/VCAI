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
    register_user,
    authenticate_user,
    login_user
)

from backend.services.session_service import (
    get_persona,
    get_all_personas,
    get_personas_by_difficulty,
    create_session,
    get_session,
    get_user_sessions,
    add_message,
    add_emotion_log,
    end_session,
    get_session_messages
)

__all__ = [
    # Auth
    "verify_password",
    "get_password_hash",
    "create_access_token",
    "decode_token",
    "get_current_user",
    "register_user",
    "authenticate_user",
    "login_user",
    
    # Session
    "get_persona",
    "get_all_personas",
    "get_personas_by_difficulty",
    "create_session",
    "get_session",
    "get_user_sessions",
    "add_message",
    "add_emotion_log",
    "end_session",
    "get_session_messages"
]