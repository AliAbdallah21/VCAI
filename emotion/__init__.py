"""
Emotion Detection Package
Provides voice emotion detection, text emotion detection, fusion, and emotional context analysis.

Public API
----------
detect_emotion(audio_array, text) -> dict
    Unified entry point: runs voice detection, text detection, fuses at 60/40, and
    returns a flat dict with all keys required by the orchestration pipeline.

detect_text_emotion(text) -> TextEmotionResult
    Text-only emotion detection (AraBERT / keyword fallback).

fuse_emotions(voice_result, text_result, strategy) -> FusedEmotionResult
    Low-level fusion with explicit strategy selection.

fuse(voice_result, text_result) -> FusedEmotionResult
    Standard 60% voice / 40% text fusion.

analyze_emotional_context(current_emotion, history) -> EmotionalContext
    Trend and risk analysis over conversation history.
"""

import logging
import numpy as np

from .voice_emotion import detect_emotion as _voice_detect_emotion, EmotionResult, predict as voice_predict
from .text_emotion import detect_text_emotion, TextEmotionResult, predict as text_predict
from .fusion import fuse_emotions, fuse, FusedEmotionResult, compare_modalities
from .agent import analyze_emotional_context, EmotionalContext, Message
from .config import (
    EMOTION_LABELS,
    ID_TO_LABEL,
    INTENSITY_THRESHOLDS,
    RISK_THRESHOLDS,
    POSITIVE_EMOTIONS,
    NEGATIVE_EMOTIONS,
    AGENT_BEHAVIORS
)

_log = logging.getLogger(__name__)

# Mapping from emotion label → mood score (-100 to +100)
# Used by the orchestration pipeline and evaluation emotion log.
_MOOD_SCORES = {
    "happy":     70,
    "interested": 40,
    "neutral":    0,
    "hesitant":  -30,
    "angry":     -70,
}


def detect_emotion(audio_array: np.ndarray, text: str) -> dict:
    """
    Unified emotion detection: voice → text → 60/40 fusion → context.

    This is the canonical function imported by orchestration/nodes/emotion_node.py.

    Args:
        audio_array: Float32 numpy array at 16 kHz (shape: (n_samples,)).
        text:        Arabic transcription of the audio.

    Returns:
        dict with keys:
            emotion      (str)   – primary emotion label
            confidence   (float) – fused confidence 0-1
            voice_emotion(str)   – emotion from audio model alone
            text_emotion (str)   – emotion from text model alone
            mood_score   (int)   – -100 to +100 (positive = good mood)
            risk_level   (str)   – "low" / "medium" / "high"
            trend        (str)   – "stable" / "improving" / "worsening"
    """
    try:
        # 1. Voice detection
        voice_result = _voice_detect_emotion(text=text, audio=audio_array)

        # 2. Text detection
        text_result = detect_text_emotion(text)

        # 3. Fuse: 60% voice + 40% text
        fused = fuse(voice_result, text_result)

        emotion = fused["primary_emotion"]
        confidence = fused["confidence"]

        # 4. Mood score from label
        mood_score = _MOOD_SCORES.get(emotion, 0)

        # 5. Contextual analysis (no history at single-turn level)
        context = analyze_emotional_context(fused, [])

        return {
            "emotion":       emotion,
            "confidence":    confidence,
            "voice_emotion": fused["voice_emotion"],
            "text_emotion":  fused["text_emotion"],
            "mood_score":    mood_score,
            "risk_level":    context["risk_level"],
            "trend":         context["trend"],
        }

    except Exception as e:
        _log.error(f"[emotion] detect_emotion failed: {e}", exc_info=True)
        return {
            "emotion":       "neutral",
            "confidence":    0.5,
            "voice_emotion": "neutral",
            "text_emotion":  "neutral",
            "mood_score":    0,
            "risk_level":    "low",
            "trend":         "stable",
        }


__all__ = [
    # Unified entry point
    "detect_emotion",

    # Individual modalities
    "detect_text_emotion",
    "voice_predict",
    "text_predict",

    # Fusion
    "fuse",
    "fuse_emotions",
    "compare_modalities",

    # Context analysis
    "analyze_emotional_context",

    # Types
    "EmotionResult",
    "TextEmotionResult",
    "FusedEmotionResult",
    "EmotionalContext",
    "Message",

    # Configuration
    "EMOTION_LABELS",
    "ID_TO_LABEL",
    "INTENSITY_THRESHOLDS",
    "RISK_THRESHOLDS",
    "POSITIVE_EMOTIONS",
    "NEGATIVE_EMOTIONS",
    "AGENT_BEHAVIORS",
]

__version__ = "1.0.0"
