# backend/main.py
"""
VCAI Backend - FastAPI Application

Run with:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

Or:
    python -m backend.main
"""

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
from backend.routers import auth_router, personas_router, sessions_router, evaluation_router
from backend.routers.websocket import router as websocket_router

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

        # LLM (local Qwen only; OpenRouter is stateless)
        if not settings.use_mocks:
            try:
                print("[Startup] Loading LLM model (this may take ~40 seconds)...")
                from llm.agent import _load_model
                _load_model()
                print("[Startup] LLM model loaded")
            except Exception as e:
                print(f"[Startup] Warning: Could not preload LLM: {e}")

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
app.include_router(websocket_router)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Root endpoint."""
    return {"status": "online", "service": "VCAI Backend", "version": "1.0.0"}


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


# ── Direct run ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
