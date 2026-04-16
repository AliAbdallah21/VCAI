# backend/main.py
"""
VCAI Backend - FastAPI Application

Run with:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

Or:
    python -m backend.main
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from backend.routers import auth_router, personas_router, sessions_router, evaluation_router
from backend.routers.websocket import router as websocket_router

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
def health_check():
    """
    Per-module health status.

    Returns cached results from startup checks (fast, no inference on each call).
    If the server started with preload_models=False the checks run on first request.

    Example response:
        {
          "status": "healthy" | "degraded",
          "checked_at": "2026-04-16T...",
          "checks": {
            "stt":     {"status": "ok",    "message": "Working"},
            "tts":     {"status": "warn",  "message": "Base model only (...)"},
            "emotion": {"status": "ok",    "message": "Working (detected: neutral)"},
            "rag":     {"status": "ok",    "message": "83 documents indexed"},
            "memory":  {"status": "ok",    "message": "PostgreSQL connected"},
            "llm":     {"status": "ok",    "message": "OpenRouter connected (...)"}
          }
        }
    """
    from backend.health import get_health_status
    return get_health_status()


# ── Direct run ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
