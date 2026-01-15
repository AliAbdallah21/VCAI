# orchestration/nodes/emotion_node.py
"""
Emotion detection node for orchestration.
Detects emotion from text and audio, analyzes emotional context.
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
    """
    Process emotion detection.
    
    This node:
    1. Detects emotion from transcription and audio
    2. Analyzes emotional context with history
    3. Updates state with emotion results
    
    Args:
        state: Current conversation state
        config: Orchestration configuration
    
    Returns:
        ConversationState: Updated state with emotion
    """
    start_time = time.time()
    
    if config and config.verbose:
        print("\n[EMOTION NODE] Analyzing emotion...")
    
    # Skip if emotion detection is disabled
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
        
        # Get emotion detection functions
        if config and config.use_mocks:
            from orchestration.mocks import detect_emotion, analyze_emotional_context
        else:
            from emotion.agent import detect_emotion, analyze_emotional_context
        
        # Detect emotion
        emotion = detect_emotion(
            text=transcription,
            audio=audio_input if audio_input is not None else np.zeros(16000, dtype=np.float32)
        )
        
        # Analyze emotional context
        emotional_context = analyze_emotional_context(
            current_emotion=emotion,
            history=history
        )
        
        # Update state
        state["emotion"] = emotion
        state["emotional_context"] = emotional_context
        
        # Log timing
        elapsed = time.time() - start_time
        state["node_timings"]["emotion"] = elapsed
        
        if config and config.verbose:
            print(f"[EMOTION NODE] Emotion: {emotion['primary_emotion']} ({emotion['confidence']:.2f})")
            print(f"[EMOTION NODE] Risk: {emotional_context['risk_level']}, Trend: {emotional_context['trend']}")
            print(f"[EMOTION NODE] Time: {elapsed:.3f}s")
        
    except Exception as e:
        # Don't fail the whole pipeline for emotion errors
        state["error"] = f"Emotion Error: {str(e)}"
        # Set default neutral emotion
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
            "recommendation": "be_professional",
            "risk_level": "low"
        }
        if config and config.verbose:
            print(f"[EMOTION NODE] Error (using defaults): {str(e)}")
    
    return state