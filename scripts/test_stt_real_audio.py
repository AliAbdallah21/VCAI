# scripts/test_stt_real_audio.py
"""
Test STT with real audio files.
Place your audio files in C:\VCAI\ and run this script.

Usage:
    python scripts/test_stt_real_audio.py
"""

# Fix OpenMP error (must be first!)
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stt.realtime_stt import transcribe_file, load_model


def main():
    print("=" * 60)
    print("STT Real Audio Test")
    print("=" * 60)
    
    # Load model first
    print("\n[1] Loading model...")
    load_model()
    
    # Define test files - ADD YOUR FILES HERE
    test_files = [
        r'C:\VCAI\WhatsApp Audio 2026-01-13 at 10.51.22 PM.mpeg',
        r'C:\VCAI\audio.wav',
        # Add more files here if needed
    ]
    
    # Test each file
    print("\n[2] Testing audio files...")
    
    for i, audio_path in enumerate(test_files, 1):
        print(f"\n--- File {i}: {os.path.basename(audio_path)} ---")
        
        if not os.path.exists(audio_path):
            print(f"  ❌ File not found: {audio_path}")
            continue
        
        try:
            text = transcribe_file(audio_path)
            print(f"  ✅ Transcription:")
            print(f"     '{text}'")
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()