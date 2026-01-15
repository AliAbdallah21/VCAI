# scripts/test_mic_quality.py
"""Test microphone quality and sample rate."""

import numpy as np
import sounddevice as sd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_microphone():
    print("=" * 50)
    print("Microphone Quality Test")
    print("=" * 50)
    
    # List audio devices
    print("\n[1] Available audio devices:")
    print(sd.query_devices())
    
    # Get default input device
    default_input = sd.query_devices(kind='input')
    print(f"\n[2] Default input device:")
    print(f"    Name: {default_input['name']}")
    print(f"    Sample rate: {default_input['default_samplerate']}")
    print(f"    Channels: {default_input['max_input_channels']}")
    
    # Record test audio
    print("\n[3] Recording 5 seconds... SPEAK NOW!")
    sample_rate = 16000
    duration = 5
    
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    audio = audio.flatten()
    
    print(f"\n[4] Recording stats:")
    print(f"    Samples: {len(audio)}")
    print(f"    Duration: {len(audio)/sample_rate:.2f}s")
    print(f"    Min/Max: {audio.min():.4f} / {audio.max():.4f}")
    print(f"    RMS Level: {np.sqrt(np.mean(audio**2)):.4f}")
    
    # Check if audio is good
    rms = np.sqrt(np.mean(audio**2))
    if rms < 0.01:
        print("    ⚠️ WARNING: Audio level very low!")
    elif rms > 0.5:
        print("    ⚠️ WARNING: Audio level very high (clipping?)")
    else:
        print("    ✅ Audio level looks good")
    
    # Test STT
    print("\n[5] Testing STT...")
    from stt.realtime_stt import transcribe_audio
    
    result = transcribe_audio(audio)
    print(f"\n[6] TRANSCRIPTION:")
    print(f"    '{result}'")
    
    # Save audio for inspection
    import wave
    wav_path = "C:/VCAI/test_mic_recording.wav"
    audio_int16 = (audio * 32767).astype(np.int16)
    with wave.open(wav_path, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio_int16.tobytes())
    print(f"\n[7] Saved recording to: {wav_path}")
    
    return result

if __name__ == "__main__":
    # Install sounddevice if needed
    try:
        import sounddevice
    except ImportError:
        print("Installing sounddevice...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "sounddevice"])
        import sounddevice
    
    test_microphone()
