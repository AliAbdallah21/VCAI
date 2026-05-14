"""
Audio storage helpers for session replay.

Saves per-turn audio to data/audio_sessions/<session_id>/ so it can be
played back via the audio endpoint. All write operations are best-effort —
a save failure must never break the live turn pipeline.
"""

from __future__ import annotations

import os
import wave
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Where session audio lives. Sits inside the repo because .gitignore already
# excludes data/audio_sessions/* — files don't pollute git.
AUDIO_ROOT = Path("data/audio_sessions")
TTS_SAMPLE_RATE = 24000  # Chatterbox output rate


def _session_dir(session_id: str) -> Path:
    """Get (and create if needed) the audio folder for a session."""
    d = AUDIO_ROOT / str(session_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_salesperson_audio(session_id: str, turn_number: int, audio_bytes: bytes,
                           ext: str = "webm") -> Optional[str]:
    """
    Save the raw webm/wav bytes received from the client for a salesperson turn.

    Returns the relative path on success (e.g. "<session_id>/turn_3_salesperson.webm"),
    or None on any failure.
    """
    if not audio_bytes:
        return None
    try:
        d = _session_dir(session_id)
        filename = f"turn_{turn_number}_salesperson.{ext}"
        path = d / filename
        path.write_bytes(audio_bytes)
        # Return the relative path (without "data/audio_sessions/" prefix) so
        # the DB column stays portable if the root ever moves.
        return f"{session_id}/{filename}"
    except Exception as e:
        logger.warning("[audio_storage] Failed to save salesperson audio "
                       "(session=%s turn=%s): %s", session_id, turn_number, e)
        return None


def save_customer_audio(session_id: str, turn_number: int,
                        audio: np.ndarray, sample_rate: int = TTS_SAMPLE_RATE) -> Optional[str]:
    """
    Save the combined TTS audio (float32 numpy array) for a customer turn.
    Writes a 16-bit PCM WAV so it plays in any browser without decoding tricks.
    """
    if audio is None or len(audio) == 0:
        return None
    try:
        d = _session_dir(session_id)
        filename = f"turn_{turn_number}_customer.wav"
        path = d / filename

        # Convert float32 [-1, 1] → int16 PCM
        clipped = np.clip(audio.astype(np.float32), -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)

        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16.tobytes())

        return f"{session_id}/{filename}"
    except Exception as e:
        logger.warning("[audio_storage] Failed to save customer audio "
                       "(session=%s turn=%s): %s", session_id, turn_number, e)
        return None


def resolve_audio_path(rel_path: str) -> Optional[Path]:
    """
    Convert a stored audio_path (e.g. "<session>/turn_3_customer.wav") to an
    absolute path on disk, after verifying it sits under AUDIO_ROOT (basic
    path-traversal guard for the serve endpoint).
    """
    if not rel_path:
        return None
    candidate = (AUDIO_ROOT / rel_path).resolve()
    root = AUDIO_ROOT.resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        # Path traversal attempt — reject.
        return None
    if not candidate.is_file():
        return None
    return candidate
