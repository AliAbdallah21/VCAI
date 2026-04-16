# backend/health.py
"""
Lightweight health checks for each VCAI module.

Each check returns:
    {"status": "ok" | "warn" | "error", "message": str}

Results are cached at startup via run_all_checks() and served from
the GET /health endpoint without re-running checks on every request.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

_log = logging.getLogger(__name__)

# ── result cache ─────────────────────────────────────────────────────────────
_last_results: dict[str, dict] = {}
_last_checked: datetime | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Individual checks
# ─────────────────────────────────────────────────────────────────────────────

def _check_stt() -> dict[str, str]:
    try:
        from stt.realtime_stt import _model_loaded  # module-level flag
        if _model_loaded:
            return {"status": "ok", "message": "Working"}
        return {"status": "warn", "message": "Not loaded (lazy)"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def _check_tts() -> dict[str, str]:
    try:
        from tts.agent import _tts_model, _egyptian_active
        if _tts_model is None:
            return {"status": "warn", "message": "Not loaded (lazy)"}
        if _egyptian_active:
            return {"status": "ok", "message": "Egyptian dialect"}
        return {"status": "warn", "message": "Base model only (Egyptian checkpoint missing)"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def _check_emotion() -> dict[str, str]:
    """Run a quick emotion detection pass with dummy audio to confirm the full stack works."""
    try:
        import numpy as np
        from emotion import detect_emotion
        dummy = np.zeros(16000, dtype=np.float32)   # 1 s of silence at 16 kHz
        result = detect_emotion(audio_array=dummy, text="مرحبا")
        return {"status": "ok", "message": f"Working (detected: {result['emotion']})"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def _check_rag() -> dict[str, str]:
    """Verify the FAISS index exists and report how many docs are indexed."""
    try:
        from rag.config import FAISS_DIRECT_INDEX_PATH, FAISS_DIRECT_DOCS_PATH
        import json

        if not (FAISS_DIRECT_INDEX_PATH.exists() and FAISS_DIRECT_DOCS_PATH.exists()):
            return {"status": "error", "message": "No index — run rag/index_build.py"}

        with open(FAISS_DIRECT_DOCS_PATH, encoding="utf-8") as fh:
            docs = json.load(fh)
        return {"status": "ok", "message": f"{len(docs)} documents indexed"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def _check_memory() -> dict[str, str]:
    """Issue a trivial SELECT 1 to confirm PostgreSQL is reachable."""
    try:
        from backend.database import get_db_context
        from sqlalchemy import text
        with get_db_context() as db:
            db.execute(text("SELECT 1"))
        return {"status": "ok", "message": "PostgreSQL connected"}
    except Exception as e:
        return {"status": "error", "message": f"DB unreachable: {e}"}


def _check_llm() -> dict[str, str]:
    """
    For OpenRouter: verify key is set and API responds (GET /api/v1/models, no generation).
    For local Qwen:  check whether the model singleton is loaded.
    """
    try:
        from llm.agent import USE_OPENROUTER, OPENROUTER_API_KEY, OPENROUTER_MODEL

        if USE_OPENROUTER:
            if not OPENROUTER_API_KEY:
                return {"status": "error", "message": "OpenRouter: OPENROUTER_API_KEY not set"}

            import requests
            resp = requests.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                timeout=5,
            )
            if resp.status_code == 200:
                return {"status": "ok", "message": f"OpenRouter connected ({OPENROUTER_MODEL})"}
            return {"status": "error", "message": f"OpenRouter HTTP {resp.status_code}"}

        # Local Qwen
        from llm.agent import _model, MODEL_NAME  # type: ignore[attr-defined]
        if _model is not None:
            return {"status": "ok", "message": f"Local Qwen loaded ({MODEL_NAME})"}
        return {"status": "warn", "message": "Local Qwen: not loaded yet (lazy)"}

    except Exception as e:
        return {"status": "error", "message": f"Failed: {e}"}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_all_checks() -> dict[str, dict]:
    """
    Run all six checks, cache the results, and return the raw results dict.
    Called once at startup; results are then served from the /health endpoint.
    """
    global _last_results, _last_checked

    _last_results = {
        "stt":     _check_stt(),
        "tts":     _check_tts(),
        "emotion": _check_emotion(),
        "rag":     _check_rag(),
        "memory":  _check_memory(),
        "llm":     _check_llm(),
    }
    _last_checked = datetime.now(timezone.utc)
    return _last_results


def get_health_status() -> dict[str, Any]:
    """
    Return the cached health status as a dict ready for JSON serialisation.
    If no check has been run yet (e.g. preload_models=False), runs them now.
    """
    if not _last_results:
        run_all_checks()

    overall = (
        "healthy"
        if all(r["status"] == "ok" for r in _last_results.values())
        else "degraded"
    )
    return {
        "status": overall,
        "checked_at": _last_checked.isoformat() if _last_checked else None,
        "checks": _last_results,
    }


def print_health_report(results: dict[str, dict]) -> None:
    """Print the startup health summary to stdout."""
    _ICONS = {"ok": "OK", "warn": "WARN", "error": "FAIL"}
    _SYMBOLS = {"ok": "✅", "warn": "⚠️ ", "error": "❌"}
    _LABELS = {
        "stt":     "STT    ",
        "tts":     "TTS    ",
        "emotion": "Emotion",
        "rag":     "RAG    ",
        "memory":  "Memory ",
        "llm":     "LLM    ",
    }
    print("-" * 60)
    for key, result in results.items():
        sym   = _SYMBOLS.get(result["status"], "?")
        label = _LABELS.get(key, key.upper())
        print(f"[Health] {label}: {sym} {result['message']}")
    print("-" * 60)
