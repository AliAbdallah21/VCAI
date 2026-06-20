# backend/main.py
"""
VCAI Backend - FastAPI Application

Run with:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

Or:
    python -m backend.main
"""

from dotenv import load_dotenv
load_dotenv()

import logging
import traceback
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from backend.config import get_settings
from backend.rate_limit import limiter
from backend.routers import (
    auth_router,
    personas_router,
    sessions_router,
    evaluation_router, learning_router,
    plans_router,
    onboarding_router,
    seats_router,
    subscriptions_router,
    manager_router,
    admin_router,
)
from backend.routers.websocket import router as websocket_router
from backend.routers.chatbot import router as chatbot_router

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    print("=" * 60)
    print("VCAI Backend Starting...")
    print("=" * 60)

    if settings.preload_models:
        print("[Startup] Preloading ML models...")

        # STT
        try:
            from stt.realtime_stt import load_model
            load_model()
            print("[Startup] STT model loaded")
        except Exception as e:
            print(f"[Startup] Warning: Could not preload STT: {e}")

        # LLM (local Qwen only; OpenRouter is stateless and needs no preload)
        if not settings.use_mocks and not settings.use_openrouter:
            try:
                print("[Startup] Loading LLM model (this may take ~40 seconds)...")
                from llm.agent import _load_model
                _load_model()
                print("[Startup] LLM model loaded")
            except Exception as e:
                print(f"[Startup] Warning: Could not preload LLM: {e}")
        elif settings.use_openrouter:
            print("[Startup] LLM: OpenRouter mode — skipping local model load")

        # TTS
        try:
            print("[Startup] Loading TTS model (this may take ~30 seconds)...")
            from tts.agent import _get_model
            _get_model()
            print("[Startup] TTS model loaded")
        except Exception as e:
            print(f"[Startup] Warning: Could not preload TTS: {e}")

        # Emotion
        try:
            print("[Startup] Loading Emotion models...")
            from emotion.voice_emotion import EmotionDetector
            EmotionDetector.get_instance()
            print("[Startup] Voice emotion model loaded")

            from emotion.text_emotion import TextEmotionDetector
            TextEmotionDetector.get_instance()
            print("[Startup] Text emotion model loaded")
        except Exception as e:
            print(f"[Startup] Warning: Could not preload Emotion models: {e}")

    # ── Health checks ─────────────────────────────────────────────────────────
    # Run after all models are loaded so results reflect true readiness.
    print("[Startup] Running health checks...")
    from backend.health import run_all_checks, print_health_report
    results = run_all_checks()
    print_health_report(results)
    # ──────────────────────────────────────────────────────────────────────────

    print(f"[Startup] Server ready at http://{settings.host}:{settings.port}")
    print(f"[Startup] API docs at http://{settings.host}:{settings.port}/docs")
    print(f"[Startup] Health    at http://{settings.host}:{settings.port}/health")
    print("=" * 60)

    yield

    print("\n[Shutdown] VCAI Backend shutting down...")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="VCAI - Virtual Customer AI",
    description="Sales Training Platform with AI-powered Virtual Customers",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Rate limiting ─────────────────────────────────────────────────────────────
# Default cap: 60 req/min/user (or per-IP for anonymous). Specific endpoints
# can override via @limiter.limit("N/timewindow") in the router.
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request, exc):
    return JSONResponse(
        status_code=429,
        content={
            "detail": "Too many requests — slow down and try again in a moment.",
            "retry_after_seconds": getattr(exc, "retry_after", None),
        },
    )

# Catch-all exception handler — logs the full traceback server-side with a
# correlation id, but returns a clean, generic message to the client so we
# never leak SQL errors, file paths, or stack traces to the browser.
@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    # HTTPException is FastAPI's intentional error response — let those flow
    # through with their author-chosen detail.
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    eid = uuid.uuid4().hex[:12]
    logger.error(
        "[unhandled] error_id=%s method=%s path=%s\n%s",
        eid, request.method, request.url.path,
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    )
    # In debug mode include the exception message so devs can see what went
    # wrong without hitting the log file; in prod, just the error_id.
    if settings.debug:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal error ({type(exc).__name__}): {str(exc)[:200]}", "error_id": eid},
        )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error — please try again. If this keeps happening, contact support with the error_id below.",
            "error_id": eid,
        },
    )

app.add_middleware(SlowAPIMiddleware)

app.include_router(auth_router, prefix="/api")
app.include_router(personas_router, prefix="/api")
app.include_router(sessions_router, prefix="/api")
app.include_router(evaluation_router, prefix="/api")
app.include_router(plans_router, prefix="/api")
app.include_router(onboarding_router, prefix="/api")
app.include_router(seats_router, prefix="/api")
app.include_router(subscriptions_router, prefix="/api")
app.include_router(learning_router, prefix="/api")
app.include_router(manager_router, prefix="/api")
app.include_router(admin_router, prefix="/api")
app.include_router(websocket_router)
app.include_router(chatbot_router)


# ── Endpoints ─────────────────────────────────────────────────────────────────

# Note: GET / used to return a JSON status. It's now handled by the SPA mount
# at the bottom of this file — the React app serves at /. Use /health for
# JSON status or /docs for the API explorer.


@app.get("/health")
def health_check(fresh: bool = False):
    """
    Per-module health status.

    By default returns cached results from startup checks (fast). Pass
    ?fresh=true to re-run live checks — useful for load balancers / cron
    probes that need real-time reachability data.

    Returns HTTP 503 when any critical dependency is in 'error' state
    (so external monitors can detect outages). Returns 200 for healthy
    or degraded-but-functional ('warn').
    """
    from fastapi.responses import JSONResponse
    from backend.health import get_health_status, run_all_checks

    if fresh:
        run_all_checks()
    payload = get_health_status()

    # 503 if any check errored; 200 otherwise.
    any_error = any(c.get("status") == "error" for c in payload.get("checks", {}).values())
    if any_error:
        return JSONResponse(status_code=503, content=payload)
    return payload


# ── SPA static frontend ───────────────────────────────────────────────────────
# Serve the built React app from frontend/dist/ at the root. This mount must
# come AFTER every API route so /api, /health, /docs, /ws etc. take priority.
# We don't use StaticFiles directly because it returns 404 for client-side
# routes like /dashboard or /compare — instead we manually fall back to
# index.html for any unknown GET so React Router can take over.
from pathlib import Path as _Path
from fastapi.responses import FileResponse as _FileResponse
from fastapi.staticfiles import StaticFiles as _StaticFiles

_FRONTEND_DIST = _Path(__file__).resolve().parent.parent / "frontend" / "dist"

if _FRONTEND_DIST.is_dir():
    # Bundled JS/CSS/assets — served with the right cache headers automatically
    _ASSETS_DIR = _FRONTEND_DIST / "assets"
    if _ASSETS_DIR.is_dir():
        app.mount("/assets", _StaticFiles(directory=str(_ASSETS_DIR)), name="assets")

    # Other top-level files Vite might emit (favicon, manifest, robots.txt, etc.)
    @app.get("/{static_file:path}", include_in_schema=False)
    async def _spa_fallback(static_file: str):
        """
        SPA fallback: a top-level file (favicon.svg, robots.txt, ...) is
        served directly; anything else returns index.html so React Router
        handles the route on the client.
        """
        # Empty path = root → index.html
        if not static_file:
            return _FileResponse(str(_FRONTEND_DIST / "index.html"))

        # Direct file hit (only one path segment, e.g. "favicon.svg")
        candidate = _FRONTEND_DIST / static_file
        if "/" not in static_file and candidate.is_file():
            return _FileResponse(str(candidate))

        # Anything else → index.html so the React router handles it
        return _FileResponse(str(_FRONTEND_DIST / "index.html"))

    print(f"[Startup] SPA mounted from {_FRONTEND_DIST}")
else:
    # No build yet — keep a placeholder root so the server doesn't 404 on /
    @app.get("/")
    def _no_frontend_root():
        return {
            "status": "online",
            "service": "VCAI Backend",
            "version": "1.0.0",
            "note": "Frontend not built. Run `cd frontend && npm run build` then restart.",
        }
    print(f"[Startup] No SPA build found at {_FRONTEND_DIST} — API-only mode")


# ── Direct run ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
