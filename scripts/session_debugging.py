# scripts/session_debugging.py
"""
Debug script for testing the full session pipeline.
"""

import asyncio
import websockets
import json
import base64
import numpy as np
import wave
import tempfile
import os
import requests

API_URL = "http://localhost:8000/api"
WS_URL = "ws://localhost:8000"

TEST_EMAIL = "debug@test.com"
TEST_PASSWORD = "debug123"
TEST_NAME = "Debug User"


def create_test_audio():
    """Create test audio files."""
    print("\n[1] Creating test audio...")
    
    sample_rate = 16000
    duration = 3
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.zeros_like(t)
    for freq in [150, 250, 400, 600, 800]:
        audio += 0.2 * np.sin(2 * np.pi * freq * t)
    
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)
    audio = audio * envelope
    audio = audio / np.max(np.abs(audio)) * 0.8
    audio_int16 = (audio * 32767).astype(np.int16)
    
    wav_path = tempfile.mktemp(suffix='.wav')
    with wave.open(wav_path, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio_int16.tobytes())
    
    print(f"   WAV created: {wav_path}")
    
    # Convert to WebM
    import subprocess
    webm_path = wav_path.replace('.wav', '.webm')
    subprocess.run(['ffmpeg', '-y', '-i', wav_path, '-c:a', 'libopus', webm_path], 
                   capture_output=True)
    
    print(f"   WebM created: {webm_path}")
    return wav_path, webm_path


def get_token():
    """Get auth token."""
    print("\n[2] Authenticating...")
    
    # Try login
    r = requests.post(f"{API_URL}/auth/login",
                      data={"username": TEST_EMAIL, "password": TEST_PASSWORD},
                      headers={"Content-Type": "application/x-www-form-urlencoded"})
    
    if r.status_code == 200:
        print("   Logged in!")
        return r.json()["access_token"]
    
    # Register
    r = requests.post(f"{API_URL}/auth/register",
                      json={"email": TEST_EMAIL, "password": TEST_PASSWORD, "full_name": TEST_NAME})
    
    if r.status_code in [200, 201]:
        print("   Registered and logged in!")
        return r.json()["access_token"]
    
    print(f"   Error: {r.status_code} - {r.text}")
    return None


def create_session(token):
    """Create a session."""
    print("\n[3] Creating session...")
    
    # Get persona
    r = requests.get(f"{API_URL}/personas", headers={"Authorization": f"Bearer {token}"})
    personas = r.json()["personas"]
    persona_id = personas[0]["id"]
    print(f"   Using persona: {persona_id}")
    
    # Create session
    r = requests.post(f"{API_URL}/sessions",
                      json={"persona_id": persona_id, "difficulty": "easy"},
                      headers={"Authorization": f"Bearer {token}"})
    
    if r.status_code == 201:
        session_id = r.json()["id"]
        print(f"   Session: {session_id}")
        return session_id
    
    print(f"   Error: {r.status_code}")
    return None


async def test_raw_audio(session_id, token, wav_path):
    """Test with raw Float32 chunks."""
    print("\n[4] Testing RAW audio (Float32 chunks)...")
    
    with wave.open(wav_path, 'rb') as wav:
        frames = wav.readframes(wav.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    
    url = f"{WS_URL}/ws/{session_id}?token={token}"
    
    async with websockets.connect(url) as ws:
        await ws.recv()  # welcome
        print("   Connected!")
        
        # Send chunks
        chunk_size = 4096
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]
            b64 = base64.b64encode(chunk.tobytes()).decode()
            await ws.send(json.dumps({"type": "audio", "data": {"audio_base64": b64}}))
        
        print(f"   Sent {len(audio)} samples in chunks")
        
        # End speaking
        await ws.send(json.dumps({"type": "end_speaking"}))
        print("   Sent end_speaking")
        
        # Get responses
        print("   Responses:")
        for _ in range(10):
            try:
                r = await asyncio.wait_for(ws.recv(), timeout=10)
                d = json.loads(r)
                t = d["type"]
                if t == "transcription":
                    print(f"      TRANSCRIPTION: '{d['data']['text']}'")
                elif t == "response":
                    print(f"      RESPONSE: '{d['data']['text'][:50]}...'")
                elif t == "error":
                    print(f"      ERROR: {d['data']['message']}")
                else:
                    print(f"      {t}")
                if t == "processing" and d["data"].get("status") == "completed":
                    break
            except asyncio.TimeoutError:
                print("      (timeout)")
                break


async def test_webm_audio(session_id, token, webm_path):
    """Test with WebM file."""
    print("\n[5] Testing WEBM audio (audio_complete)...")
    
    with open(webm_path, 'rb') as f:
        webm_data = f.read()
    
    b64 = base64.b64encode(webm_data).decode()
    print(f"   WebM: {len(webm_data)} bytes")
    
    url = f"{WS_URL}/ws/{session_id}?token={token}"
    
    async with websockets.connect(url) as ws:
        await ws.recv()  # welcome
        print("   Connected!")
        
        # Send complete audio
        await ws.send(json.dumps({
            "type": "audio_complete",
            "data": {"audio_base64": b64, "format": "webm"}
        }))
        print("   Sent audio_complete")
        
        # Get responses
        print("   Responses:")
        for _ in range(10):
            try:
                r = await asyncio.wait_for(ws.recv(), timeout=15)
                d = json.loads(r)
                t = d["type"]
                if t == "transcription":
                    print(f"      TRANSCRIPTION: '{d['data']['text']}'")
                elif t == "response":
                    print(f"      RESPONSE: '{d['data']['text'][:50]}...'")
                elif t == "error":
                    print(f"      ERROR: {d['data']['message']}")
                else:
                    print(f"      {t}")
                if t == "processing" and d["data"].get("status") == "completed":
                    break
            except asyncio.TimeoutError:
                print("      (timeout)")
                break


async def main():
    print("=" * 50)
    print("VCAI Debug Script")
    print("=" * 50)
    
    wav_path, webm_path = create_test_audio()
    
    token = get_token()
    if not token:
        return
    
    # Test 1: Raw audio
    session1 = create_session(token)
    if session1:
        await test_raw_audio(session1, token, wav_path)
    
    # Test 2: WebM audio
    session2 = create_session(token)
    if session2:
        await test_webm_audio(session2, token, webm_path)
    
    # Cleanup
    for f in [wav_path, webm_path]:
        if os.path.exists(f):
            os.unlink(f)
    
    print("\n" + "=" * 50)
    print("DONE")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
