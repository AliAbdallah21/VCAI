# orchestration/nodes/tts_node.py
"""
TTS node for orchestration.
Supports both full and chunk-based audio generation.
"""

import time
import numpy as np

from orchestration.state import ConversationState
from orchestration.config import OrchestrationConfig
from shared.exceptions import TTSSynthesisError

# Map emotion labels to TTS-compatible presets
EMOTION_TO_TTS = {
    "angry": "frustrated",
    "happy": "happy",
    "hesitant": "hesitant",
    "interested": "interested",
    "neutral": "neutral",
    "frustrated": "frustrated",
    "sad": "neutral",
}


def _get_tts_emotion(state: dict) -> str:
    """Extract TTS emotion from state."""
    detected = state.get("emotion", {})
    label = detected.get("primary_emotion", "neutral")
    return EMOTION_TO_TTS.get(label, "neutral")


def _get_tts_function(config):
    """Get the appropriate TTS function."""
    if config and config.use_mocks:
        from orchestration.mocks import text_to_speech
    else:
        from tts.agent import text_to_speech
    return text_to_speech


def tts_node(
    state: ConversationState,
    config: OrchestrationConfig = None
) -> ConversationState:
    """
    Non-streaming TTS: convert full response to audio.
    """
    start_time = time.time()

    if config and config.verbose:
        print("\n[TTS NODE] Generating audio...")

    try:
        llm_response = state.get("llm_response")
        if not llm_response:
            raise TTSSynthesisError("No LLM response to synthesize")

        text_to_speech = _get_tts_function(config)
        persona = state.get("persona") or {}
        voice_id = persona.get("voice_id", "default")
        tts_emotion = _get_tts_emotion(state)

        audio_output = text_to_speech(
            text=llm_response,
            voice_id=voice_id,
            emotion=tts_emotion
        )

        state["audio_output"] = audio_output
        state["phase"] = "idle"

        elapsed = time.time() - start_time
        state["node_timings"]["tts"] = elapsed

        if config and config.verbose:
            duration = len(audio_output) / 24000 if audio_output is not None else 0
            detected_label = state.get("emotion", {}).get("primary_emotion", "neutral")
            print(f"[TTS NODE] Audio duration: {duration:.2f}s")
            print(f"[TTS NODE] Voice: {voice_id}, Emotion: {tts_emotion} (detected: {detected_label})")
            print(f"[TTS NODE] Time: {elapsed:.3f}s")

    except Exception as e:
        state["error"] = f"TTS Error: {str(e)}"
        state["audio_output"] = None
        if config and config.verbose:
            print(f"[TTS NODE] Error: {str(e)}")

    return state


def tts_chunk(
    text: str,
    state: ConversationState,
    config: OrchestrationConfig = None
) -> np.ndarray:
    """
    Generate audio for a single text chunk (sentence).
    Used by streaming pipeline.
    
    Args:
        text: Single sentence to synthesize
        state: Conversation state (for voice/emotion settings)
        config: Orchestration config
    
    Returns:
        np.ndarray: Audio samples at 24kHz
    """
    start_time = time.time()

    try:
        text_to_speech = _get_tts_function(config)
        persona = state.get("persona") or {}
        voice_id = persona.get("voice_id", "default")
        tts_emotion = _get_tts_emotion(state)

        audio = text_to_speech(
            text=text,
            voice_id=voice_id,
            emotion=tts_emotion
        )

        elapsed = time.time() - start_time

        if config and config.verbose:
            duration = len(audio) / 24000 if audio is not None else 0
            print(f"[TTS CHUNK] '{text[:30]}...' -> {duration:.1f}s audio in {elapsed:.2f}s")

        return audio

    except Exception as e:
        if config and config.verbose:
            print(f"[TTS CHUNK] Error for '{text[:30]}...': {e}")
        return np.zeros(0, dtype=np.float32)