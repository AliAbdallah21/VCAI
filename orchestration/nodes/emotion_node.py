# orchestration/nodes/emotion_node.py
"""
Emotion detection node for orchestration.
Detects emotion from text and audio, analyzes emotional context.
Uses FUSION (voice + text) for more accurate results.
"""

import time
import numpy as np

from orchestration.state import ConversationState
from orchestration.config import OrchestrationConfig
from shared.exceptions import EmotionDetectionError


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
            from orchestration.mocks import detect_emotion, analyze_emotional_context
            emotion = detect_emotion(
                text=transcription,
                audio=audio_input if audio_input is not None else np.zeros(16000, dtype=np.float32)
            )
        else:
            from emotion import detect_emotion, detect_text_emotion, fuse_emotions, analyze_emotional_context
            
            # Voice emotion from audio
            voice_result = detect_emotion(
                text=transcription,
                audio=audio_input if audio_input is not None else np.zeros(16000, dtype=np.float32)
            )
            
            # Text emotion from transcription
            text_result = detect_text_emotion(transcription)
            
            # Fuse both for accuracy (voice model alone is biased)
            emotion = fuse_emotions(voice_result, text_result, "adaptive")
            
            # Add voice/text breakdown for logging
            emotion["voice_emotion"] = voice_result["primary_emotion"]
            emotion["text_emotion"] = text_result["primary_emotion"]
            
            if config and config.verbose:
                print(f"[EMOTION NODE] Voice: {voice_result['primary_emotion']} ({voice_result['confidence']:.2f})")
                print(f"[EMOTION NODE] Text:  {text_result['primary_emotion']} ({text_result['confidence']:.2f})")
                print(f"[EMOTION NODE] Fused: {emotion['primary_emotion']} ({emotion['confidence']:.2f})")
        
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