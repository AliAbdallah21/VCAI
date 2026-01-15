# orchestration/nodes/stt_node.py
"""
STT (Speech-to-Text) node for orchestration.
Converts audio input to text transcription.
"""

import time
from datetime import datetime
import numpy as np

from orchestration.state import ConversationState
from orchestration.config import OrchestrationConfig
from shared.exceptions import STTTranscriptionError, AudioTooShortError
from shared.constants import STT_MIN_DURATION, STT_SAMPLE_RATE


def stt_node(
    state: ConversationState,
    config: OrchestrationConfig = None
) -> ConversationState:
    """
    Process audio input through STT.
    
    This node:
    1. Validates audio input
    2. Transcribes audio to text
    3. Updates state with transcription
    
    Args:
        state: Current conversation state
        config: Orchestration configuration
    
    Returns:
        ConversationState: Updated state with transcription
    """
    start_time = time.time()
    
    if config and config.verbose:
        print("\n[STT NODE] Processing audio...")
    
    try:
        # Validate audio input
        audio_input = state.get("audio_input")
        
        if audio_input is None:
            raise STTTranscriptionError("No audio input provided")
        
        if not isinstance(audio_input, np.ndarray):
            raise STTTranscriptionError(f"Invalid audio type: {type(audio_input)}")
        
        # Check audio duration
        duration = len(audio_input) / STT_SAMPLE_RATE
        if duration < STT_MIN_DURATION:
            raise AudioTooShortError(duration, STT_MIN_DURATION)
        
        # Transcribe audio
        # TODO: Replace with real STT when ready
        # from stt.realtime_stt import transcribe_audio
        # transcription = transcribe_audio(audio_input)
        
        # For now, use a placeholder that simulates STT
        # In real implementation, this would call your completed STT
        transcription = _transcribe_audio(audio_input, config)
        
        # Update state
        state["transcription"] = transcription
        state["phase"] = "processing"
        
        # Log timing
        elapsed = time.time() - start_time
        state["node_timings"]["stt"] = elapsed
        
        if config and config.verbose:
            print(f"[STT NODE] Transcription: '{transcription}'")
            print(f"[STT NODE] Time: {elapsed:.3f}s")
        
    except Exception as e:
        state["error"] = f"STT Error: {str(e)}"
        if config and config.verbose:
            print(f"[STT NODE] Error: {str(e)}")
    
    return state


def _transcribe_audio(audio: np.ndarray, config: OrchestrationConfig = None) -> str:
    """
    Internal transcription function.
    Replace this with your actual STT implementation.
    """
    # Check if we should use mock or real
    if config and config.use_mocks:
        # Return a test transcription for development
        # In real usage, the actual audio would be transcribed
        return "هذا نص تجريبي من نظام التعرف على الكلام"
    else:
        # Use real STT
        try:
            from stt.realtime_stt import transcribe_audio
            return transcribe_audio(audio)
        except ImportError:
            # Fallback if STT module not ready
            return "نص تجريبي - STT module not found"


def validate_audio_input(audio: np.ndarray) -> tuple[bool, str]:
    """
    Validate audio input before processing.
    
    Args:
        audio: Audio numpy array
    
    Returns:
        tuple[bool, str]: (is_valid, error_message)
    """
    if audio is None:
        return False, "Audio is None"
    
    if not isinstance(audio, np.ndarray):
        return False, f"Expected numpy array, got {type(audio)}"
    
    if audio.dtype != np.float32:
        return False, f"Expected float32, got {audio.dtype}"
    
    if len(audio.shape) != 1:
        return False, f"Expected 1D array, got shape {audio.shape}"
    
    duration = len(audio) / STT_SAMPLE_RATE
    if duration < STT_MIN_DURATION:
        return False, f"Audio too short: {duration:.2f}s (min: {STT_MIN_DURATION}s)"
    
    return True, ""