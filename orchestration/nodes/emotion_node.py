# orchestration/nodes/emotion_node.py
"""
Emotion detection node for orchestration.
Detects emotion from text and audio, analyzes emotional context.
Uses FUSION (voice + text) for more accurate results.
"""

import time
import logging
import numpy as np

from orchestration.state import ConversationState
from orchestration.config import OrchestrationConfig
from shared.exceptions import EmotionDetectionError

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level import check — runs once when the orchestration package loads.
# A WARNING here means emotion detection is broken for the entire server run.
# ---------------------------------------------------------------------------
try:
    from emotion import detect_emotion as _emotion_detect_emotion  # noqa: F401
    _emotion_available = True
except ImportError as _emotion_import_error:
    _emotion_available = False
    _logger.warning(
        "\n"
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║  [EMOTION NODE] ⚠️  IMPORT FAILED — emotion module missing   ║\n"
        "║  Every conversation turn will use neutral emotion fallback.  ║\n"
        "╚══════════════════════════════════════════════════════════════╝\n"
        f"  ImportError: {_emotion_import_error}\n"
        "  Fix: ensure emotion/ package is installed and model path exists."
    )


def emotion_node(
    state: ConversationState,
    config: OrchestrationConfig = None
) -> ConversationState:
    start_time = time.time()
    
    if config and config.verbose:
        print("\n[EMOTION NODE] Analyzing emotion...")
    
    if config and not config.enable_emotion:
        if config.verbose:
            print("[EMOTION NODE] Skipped (disabled)")
        return state
    
    try:
        transcription = state.get("transcription")
        audio_input = state.get("audio_input")
        history = state.get("history", [])
        
        if not transcription:
            raise EmotionDetectionError("No transcription available")
        
        if config and config.use_mocks:
            from orchestration.mocks import detect_emotion as _mock_detect_emotion
            from orchestration.mocks import analyze_emotional_context
            emotion = _mock_detect_emotion(
                text=transcription,
                audio=audio_input if audio_input is not None else np.zeros(16000, dtype=np.float32)
            )
        else:
            if not _emotion_available:
                raise EmotionDetectionError(
                    "emotion module failed to import at startup — "
                    "check server logs for the ImportError detail."
                )

            from emotion import detect_emotion as _real_detect_emotion
            from emotion import analyze_emotional_context

            audio = audio_input if audio_input is not None else np.zeros(16000, dtype=np.float32)

            # Unified call: voice → text → 60/40 fusion (returns flat dict)
            raw = _real_detect_emotion(audio_array=audio, text=transcription)

            # Remap to the shape the rest of the pipeline expects
            emotion = {
                "primary_emotion": raw["emotion"],
                "confidence":      raw["confidence"],
                "voice_emotion":   raw["voice_emotion"],
                "text_emotion":    raw["text_emotion"],
                "intensity":       "high" if raw["confidence"] >= 0.7
                                   else "medium" if raw["confidence"] >= 0.4
                                   else "low",
                "scores":          {},
                "mood_score":      raw["mood_score"],
            }

            if config and config.verbose:
                print(f"[EMOTION NODE] Voice: {raw['voice_emotion']}")
                print(f"[EMOTION NODE] Text:  {raw['text_emotion']}")
                print(f"[EMOTION NODE] Fused: {raw['emotion']} ({raw['confidence']:.2f})")
        
        # Analyze emotional context - filter history to only messages with valid emotion
        valid_history = [
            msg for msg in history
            if msg.get("emotion") is not None and isinstance(msg.get("emotion"), dict)
        ]
        
        emotional_context = analyze_emotional_context(
            current_emotion=emotion,
            history=valid_history
        )
        
        state["emotion"] = emotion
        state["emotional_context"] = emotional_context
        
        elapsed = time.time() - start_time
        state["node_timings"]["emotion"] = elapsed
        
        if config and config.verbose:
            print(f"[EMOTION NODE] Final: {emotion['primary_emotion']} ({emotion['confidence']:.2f})")
            print(f"[EMOTION NODE] Risk: {emotional_context['risk_level']}, Trend: {emotional_context['trend']}")
            print(f"[EMOTION NODE] Time: {elapsed:.3f}s")
        
    except Exception as e:
        state["error"] = f"Emotion Error: {str(e)}"
        state["emotion"] = {
            "primary_emotion": "neutral",
            "confidence": 0.5,
            "voice_emotion": "neutral",
            "text_emotion": "neutral",
            "intensity": "low",
            "scores": {}
        }
        state["emotional_context"] = {
            "current": state["emotion"],
            "trend": "stable",
            "recommendation": "normal",
            "risk_level": "low"
        }
        if config and config.verbose:
            print(f"[EMOTION NODE] Error (using defaults): {str(e)}")
            import traceback
            traceback.print_exc()
    
    return state