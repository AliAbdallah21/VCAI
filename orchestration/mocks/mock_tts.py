# orchestration/mocks/mock_tts.py
"""
Mock TTS (Text-to-Speech) functions.
Replace with real imports from tts.agent when Person B completes their work.

USAGE:
    # Now (development):
    from orchestration.mocks.mock_tts import text_to_speech
    
    # Later (integration):
    from tts.agent import text_to_speech
"""

import numpy as np
from typing import Optional

from shared.constants import TTS_SAMPLE_RATE


def text_to_speech(
    text: str,
    voice_id: str = "default",
    emotion: str = "neutral",
    language_id: str = "ar"
) -> np.ndarray:
    """
    Mock TTS - generates silent audio with realistic duration.
    
    INPUT:
        text: str - Arabic text to speak
        voice_id: str - Voice ID (ignored in mock)
        emotion: str - Emotion (ignored in mock)
        language_id: str - Language (Arabic by default)
    
    OUTPUT:
        np.ndarray - Silent audio, 24000 Hz, float32
    """
    # Estimate duration: ~0.1 seconds per character for Arabic
    # Average speaking rate
    char_count = len(text)
    duration_seconds = max(0.5, char_count * 0.08)  # Minimum 0.5 seconds
    
    # Generate silent audio (or slight noise to simulate audio)
    num_samples = int(TTS_SAMPLE_RATE * duration_seconds)
    
    # Add very slight noise so it's not completely silent
    audio = np.random.randn(num_samples).astype(np.float32) * 0.001
    
    print(f"[MOCK TTS] Generated {duration_seconds:.2f}s audio for: '{text[:50]}...'")
    print(f"[MOCK TTS] Voice: {voice_id}, Emotion: {emotion}")
    
    return audio


def get_available_voices() -> list[dict]:
    """
    Mock function to list available voices.
    
    OUTPUT:
        list[dict] - Available voice configurations
    """
    return [
        {
            "id": "default",
            "name": "Default Egyptian Male",
            "language": "ar-EG",
            "gender": "male"
        },
        {
            "id": "egyptian_male_01",
            "name": "Egyptian Male 1",
            "language": "ar-EG",
            "gender": "male"
        },
        {
            "id": "egyptian_female_01",
            "name": "Egyptian Female 1",
            "language": "ar-EG",
            "gender": "female"
        }
    ]


# For testing
if __name__ == "__main__":
    test_text = "مرحبا بيك في شركتنا، إزي أقدر أساعدك النهاردة؟"
    audio = text_to_speech(test_text, voice_id="egyptian_male_01", emotion="friendly")
    print(f"Generated audio shape: {audio.shape}")
    print(f"Duration: {len(audio) / TTS_SAMPLE_RATE:.2f} seconds")