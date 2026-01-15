# stt/realtime_stt.py
"""
Real-time Speech-to-Text using Faster-Whisper.
Component: Person A (Ali)

INTERFACE (from shared/interfaces.py):
    def transcribe_audio(audio_data: np.ndarray) -> str

USAGE:
    from stt.realtime_stt import transcribe_audio
    
    # audio_data: numpy array, float32, 16kHz
    text = transcribe_audio(audio_data)
"""

import numpy as np
import time
from typing import Optional
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL MODEL (Lazy Loading)
# ══════════════════════════════════════════════════════════════════════════════

_whisper_model = None
_model_loaded = False


def _get_device_and_compute_type() -> tuple[str, str]:
    """
    Detect best device (CUDA or CPU) and compute type.
    
    Returns:
        tuple[str, str]: (device, compute_type)
    """
    try:
        import torch
        if torch.cuda.is_available():
            print("[STT] CUDA available - using GPU")
            return "cuda", "float16"
        else:
            print("[STT] CUDA not available - using CPU")
            return "cpu", "int8"
    except ImportError:
        print("[STT] PyTorch not found - using CPU")
        return "cpu", "int8"


def load_model(force_cpu: bool = False):
    """
    Load the Whisper model.
    
    Args:
        force_cpu: If True, use CPU even if CUDA available
    """
    global _whisper_model, _model_loaded
    
    if _model_loaded:
        return _whisper_model
    
    print("[STT] Loading Faster-Whisper model...")
    start_time = time.time()
    
    from faster_whisper import WhisperModel
    
    # Get device
    if force_cpu:
        device, compute_type = "cpu", "int8"
    else:
        device, compute_type = _get_device_and_compute_type()
    
    # Load model
    _whisper_model = WhisperModel(
        "large-v3-turbo",
        device=device,
        compute_type=compute_type
    )
    
    _model_loaded = True
    elapsed = time.time() - start_time
    print(f"[STT] ✅ Model loaded in {elapsed:.1f}s (device={device})")
    
    return _whisper_model


def get_model():
    """Get the loaded model, loading it if necessary."""
    global _whisper_model, _model_loaded
    
    if not _model_loaded:
        load_model()
    
    return _whisper_model


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION (Matches Interface)
# ══════════════════════════════════════════════════════════════════════════════

def transcribe_audio(audio_data: np.ndarray) -> str:
    """
    Transcribe audio data to Arabic text.
    
    ⚠️ THIS IS THE INTERFACE FUNCTION - signature must match shared/interfaces.py
    
    Args:
        audio_data: numpy array of audio samples
                   - dtype: float32
                   - sample rate: 16000 Hz
                   - shape: (num_samples,) - 1D array
    
    Returns:
        str: Transcribed Arabic text
    
    Example:
        >>> import numpy as np
        >>> audio = np.random.randn(16000 * 5).astype(np.float32)  # 5 seconds
        >>> text = transcribe_audio(audio)
        >>> print(text)
        "مرحبا، أنا عايز أشوف شقة"
    """
    # ──────────────────────────────────────────────────────────────────────────
    # Input Validation
    # ──────────────────────────────────────────────────────────────────────────
    
    if audio_data is None:
        raise ValueError("[STT] audio_data cannot be None")
    
    if not isinstance(audio_data, np.ndarray):
        raise ValueError(f"[STT] Expected numpy array, got {type(audio_data)}")
    
    # Ensure float32 (Whisper expects this)
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    
    # Ensure 1D array
    if len(audio_data.shape) > 1:
        audio_data = audio_data.flatten()
    
    # Check duration (warn if very short)
    duration_seconds = len(audio_data) / 16000
    if duration_seconds < 0.5:
        print(f"[STT] Warning: Audio very short ({duration_seconds:.2f}s)")
    
    # ──────────────────────────────────────────────────────────────────────────
    # Transcription
    # ──────────────────────────────────────────────────────────────────────────
    
    model = get_model()
    
    start_time = time.time()
    
    # Transcribe (audio_data is numpy array, not file path)
    segments, info = model.transcribe(
        audio_data,              # numpy array directly
        language="ar",           # Arabic
        beam_size=5,
        vad_filter=True,         # Reduces hallucination
        vad_parameters=dict(
            min_silence_duration_ms=500,
            speech_pad_ms=200
        )
    )
    
    # Combine all segments
    full_text = " ".join([segment.text for segment in segments])
    
    elapsed = time.time() - start_time
    print(f"[STT] Transcribed in {elapsed:.3f}s: '{full_text[:50]}...'")
    
    return full_text.strip()


# ══════════════════════════════════════════════════════════════════════════════
# ADDITIONAL UTILITIES (Optional, for your convenience)
# ══════════════════════════════════════════════════════════════════════════════

def transcribe_audio_detailed(audio_data: np.ndarray) -> dict:
    """
    Transcribe with detailed output (language, duration, segments).
    
    Use this when you need more info than just the text.
    
    Args:
        audio_data: numpy array (float32, 16kHz)
    
    Returns:
        dict: {
            'text': str,
            'language': str,
            'duration': float,
            'segments': list
        }
    """
    if audio_data is None:
        raise ValueError("audio_data cannot be None")
    
    if not isinstance(audio_data, np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(audio_data)}")
    
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    
    if len(audio_data.shape) > 1:
        audio_data = audio_data.flatten()
    
    model = get_model()
    
    segments, info = model.transcribe(
        audio_data,
        language="ar",
        beam_size=5,
        vad_filter=True
    )
    
    segments_list = []
    texts = []
    
    for segment in segments:
        segments_list.append({
            'start': segment.start,
            'end': segment.end,
            'text': segment.text
        })
        texts.append(segment.text)
    
    return {
        'text': " ".join(texts).strip(),
        'language': info.language,
        'duration': info.duration,
        'segments': segments_list
    }


def transcribe_file(audio_path: str, language: str = "ar") -> str:
    """
    Transcribe from file path (your original function style).
    
    Args:
        audio_path: Path to audio file
        language: Language code (default: Arabic)
    
    Returns:
        str: Transcribed text
    """
    model = get_model()
    
    segments, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        vad_filter=True
    )
    
    full_text = " ".join([segment.text for segment in segments])
    
    return full_text.strip()


# ══════════════════════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*60)
    print("STT Module Test")
    print("="*60)
    
    # Test 1: Load model
    print("\n[Test 1] Loading model...")
    load_model()
    
    # Test 2: Transcribe silent audio (just to test the function works)
    print("\n[Test 2] Testing with silent audio...")
    silent_audio = np.zeros(16000 * 2, dtype=np.float32)  # 2 seconds silence
    result = transcribe_audio(silent_audio)
    print(f"Result (silent audio): '{result}'")
    
    # Test 3: Transcribe random noise (won't make sense, just testing)
    print("\n[Test 3] Testing with noise...")
    noise_audio = np.random.randn(16000 * 3).astype(np.float32) * 0.1
    result = transcribe_audio(noise_audio)
    print(f"Result (noise): '{result}'")
    # Test 4:
    print("\n[Test 4] Testing with actual file...")
    text = transcribe_file('C:/VCAI/WhatsApp Audio 2026-01-13 at 10.51.22 PM.mpeg')
    text2 = transcribe_file('C:/VCAI/audio.wav')

    print(f"Result (file): '{text}'")
    print(f"Result (file2): '{text2}'")

    print("\n" + "="*60)
    print("✅ STT Module tests completed!")
    print("="*60)
    print("\nTo test with real audio, use:")
    print("  text = transcribe_file('path/to/audio.wav')")