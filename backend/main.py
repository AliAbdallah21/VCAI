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
from backend.routers import auth_router, personas_router, sessions_router
from backend.routers.websocket import router as websocket_router

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events.
    """
    # Startup
    print("=" * 60)
    print("VCAI Backend Starting...")
    print("=" * 60)
    
    if settings.preload_models:
        print("[Startup] Preloading ML models...")
        try:
            # Preload STT model
            from stt.realtime_stt import load_model
            load_model()
            print("[Startup] STT model loaded")
        except Exception as e:
            print(f"[Startup] Warning: Could not preload STT: {e}")
    
    print(f"[Startup] Server ready at http://{settings.host}:{settings.port}")
    print(f"[Startup] API docs at http://{settings.host}:{settings.port}/docs")
    print("=" * 60)
    
    yield
    
    # Shutdown
    print("\n[Shutdown] VCAI Backend shutting down...")


# Create FastAPI app
app = FastAPI(
    title="VCAI - Virtual Customer AI",
    description="Sales Training Platform with AI-powered Virtual Customers",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router, prefix="/api")
app.include_router(personas_router, prefix="/api")
app.include_router(sessions_router, prefix="/api")
app.include_router(websocket_router)  # WebSocket at root level


@app.get("/")
def root():
    """Root endpoint - health check."""
    return {
        "status": "online",
        "service": "VCAI Backend",
        "version": "1.0.0"
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "database": "connected",
        "models": "loaded" if settings.preload_models else "lazy"
    }


# Run with Python directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )