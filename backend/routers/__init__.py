# backend/routers/__init__.py
"""
API routers.
"""

from backend.routers.auth import router as auth_router
from backend.routers.personas import router as personas_router
from backend.routers.sessions import router as sessions_router
from backend.routers.evaluation import router as evaluation_router
from backend.routers.plans import router as plans_router
from backend.routers.onboarding import router as onboarding_router
from backend.routers.seats import router as seats_router
from backend.routers.subscriptions import router as subscriptions_router
from backend.routers.learning import router as learning_router
from backend.routers.manager import router as manager_router
from backend.routers.admin import router as admin_router

__all__ = [
    "auth_router",
    "personas_router",
    "sessions_router",
    "evaluation_router",
    "plans_router",
    "onboarding_router",
    "seats_router",
    "subscriptions_router",
    "learning_router",
    "manager_router",
    "admin_router",
]