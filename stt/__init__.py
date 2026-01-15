# stt/__init__.py
"""
Speech-to-Text module using Faster-Whisper.

USAGE:
    from stt import transcribe_audio
    
    # Transcribe numpy array
    text = transcribe_audio(audio_data)
    
    # Or transcribe file
    from stt import transcribe_file
    text = transcribe_file("path/to/audio.wav")
"""

from stt.realtime_stt import (
    transcribe_audio,
    transcribe_audio_detailed,
    transcribe_file,
    load_model,
    get_model
)

__all__ = [
    "transcribe_audio",         # Main interface function
    "transcribe_audio_detailed", # With extra info
    "transcribe_file",          # From file path
    "load_model",               # Pre-load model
    "get_model"                 # Get loaded model
]