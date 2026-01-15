# orchestration/nodes/tts_node.py
"""
TTS (Text-to-Speech) node for orchestration.
Converts LLM response to audio output.
"""

import time

from orchestration.state import ConversationState
from orchestration.config import OrchestrationConfig
from shared.exceptions import TTSSynthesisError


def tts_node(
    state: ConversationState,
    config: OrchestrationConfig = None
) -> ConversationState:
    """
    Convert response text to speech audio.
    
    This node:
    1. Gets LLM response text
    2. Determines voice and emotion from persona
    3. Generates audio output
    4. Updates state with audio
    
    Args:
        state: Current conversation state
        config: Orchestration configuration
    
    Returns:
        ConversationState: Updated state with audio output
    """
    start_time = time.time()
    
    if config and config.verbose:
        print("\n[TTS NODE] Generating audio...")
    
    try:
        llm_response = state.get("llm_response")
        
        if not llm_response:
            raise TTSSynthesisError("No LLM response to synthesize")
        
        # Get TTS function
        if config and config.use_mocks:
            from orchestration.mocks import text_to_speech
        else:
            from tts.agent import text_to_speech
        
        # Get voice settings from persona
        persona = state.get("persona") or {}
        voice_id = persona.get("voice_id", "default")
        default_emotion = persona.get("default_emotion", "neutral")
        
        # Could also adjust emotion based on context
        # For now, use persona's default
        emotion = default_emotion
        
        # Generate audio
        audio_output = text_to_speech(
            text=llm_response,
            voice_id=voice_id,
            emotion=emotion
        )
        
        # Update state
        state["audio_output"] = audio_output
        state["phase"] = "idle"  # Ready for next turn
        
        # Log timing
        elapsed = time.time() - start_time
        state["node_timings"]["tts"] = elapsed
        
        if config and config.verbose:
            duration = len(audio_output) / 22050 if audio_output is not None else 0
            print(f"[TTS NODE] Audio duration: {duration:.2f}s")
            print(f"[TTS NODE] Voice: {voice_id}, Emotion: {emotion}")
            print(f"[TTS NODE] Time: {elapsed:.3f}s")
        
    except Exception as e:
        state["error"] = f"TTS Error: {str(e)}"
        state["audio_output"] = None
        if config and config.verbose:
            print(f"[TTS NODE] Error: {str(e)}")
    
    return state