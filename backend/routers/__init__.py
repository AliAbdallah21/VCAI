# backend/routers/__init__.py
"""
API routers.
"""

from backend.routers.auth import router as auth_router
from backend.routers.personas import router as personas_router
from backend.routers.sessions import router as sessions_router

__all__ = [
    "auth_router",
    "personas_router",
    "sessions_router"
]