# scripts/test_stt.py
"""
Test STT with real audio file.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import wave

def test_with_file(audio_path):
    """Test STT with a real audio file."""
    print("=" * 50)
    print("STT Test with Real Audio")
    print("=" * 50)
    
    if not os.path.exists(audio_path):
        print(f"ERROR: File not found: {audio_path}")
        return
    
    print(f"\n[1] Loading audio: {audio_path}")
    
    file_size = os.path.getsize(audio_path)
    print(f"    File size: {file_size} bytes")
    
    # Load WAV
    with wave.open(audio_path, 'rb') as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        n_frames = wav.getnframes()
        duration = n_frames / sample_rate
        
        print(f"    Channels: {channels}")
        print(f"    Sample width: {sample_width} bytes")
        print(f"    Sample rate: {sample_rate} Hz")
        print(f"    Duration: {duration:.2f} seconds")
        
        frames = wav.readframes(n_frames)
    
    # Convert to float32
    if sample_width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        print(f"    ERROR: Unsupported sample width")
        return
    
    # Stereo to mono
    if channels == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)
        print("    Converted stereo to mono")
    
    print(f"    Audio shape: {audio.shape}")
    print(f"    Audio RMS: {np.sqrt(np.mean(audio**2)):.4f}")
    
    # Test STT with array
    print("\n[2] Testing transcribe_audio (numpy array)...")
    from stt.realtime_stt import transcribe_audio
    
    result = transcribe_audio(audio)
    
    print(f"\n    RESULT: '{result}'")
    
    # Test STT with file path
    print("\n[3] Testing transcribe_file (file path)...")
    from stt.realtime_stt import transcribe_file
    
    result2 = transcribe_file(audio_path)
    
    print(f"\n    RESULT: '{result2}'")
    
    print("\n" + "=" * 50)
    if result or result2:
        print("SUCCESS! STT is working.")
    else:
        print("WARNING: Empty transcription.")
    print("=" * 50)


if __name__ == "__main__":
    audio_file = r"C:\VCAI\audio.wav"
    test_with_file(audio_file)
