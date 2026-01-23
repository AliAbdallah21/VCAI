# scripts/test_websocket.py
"""
Test WebSocket connection for real-time conversation.

Usage:
    1. Start the backend: python -m backend.main
    2. Run this test: python scripts/test_websocket.py
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import asyncio
import json
import base64
import numpy as np
import websockets
import requests

# Configuration
BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000"


def register_and_login():
    """Register a test user and get token."""
    print("[1] Registering/logging in user...")
    
    # Try to register (might already exist)
    try:
        response = requests.post(
            f"{BASE_URL}/api/auth/register",
            json={
                "email": "wstest@test.com",
                "password": "123456",
                "full_name": "WebSocket Tester"
            }
        )
        if response.status_code == 201:
            data = response.json()
            print(f"    ✅ Registered new user")
            return data["access_token"]
    except:
        pass
    
    # Login if already exists
    response = requests.post(
        f"{BASE_URL}/api/auth/login",
        data={
            "username": "wstest@test.com",
            "password": "123456"
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"    ✅ Logged in")
        return data["access_token"]
    else:
        print(f"    ❌ Login failed: {response.text}")
        return None


def create_session(token: str):
    """Create a training session."""
    print("[2] Creating training session...")
    
    response = requests.post(
        f"{BASE_URL}/api/sessions",
        json={
            "persona_id": "friendly_customer",
            "difficulty": "easy"
        },
        headers={"Authorization": f"Bearer {token}"}
    )
    
    if response.status_code == 201:
        data = response.json()
        print(f"    ✅ Session created: {data['id']}")
        return data["id"]
    else:
        print(f"    ❌ Failed: {response.text}")
        return None


def generate_test_audio():
    """Generate a simple test audio (sine wave)."""
    # Generate 2 seconds of audio at 16kHz
    sample_rate = 16000
    duration = 2.0
    frequency = 440  # Hz
    
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio = 0.3 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    return audio


async def test_websocket(token: str, session_id: str):
    """Test WebSocket connection."""
    print("[3] Connecting to WebSocket...")
    
    uri = f"{WS_URL}/ws/{session_id}?token={token}"
    
    try:
        async with websockets.connect(uri) as websocket:
            # Receive welcome message
            response = await websocket.recv()
            data = json.loads(response)
            print(f"    ✅ Connected: {data['data']['message']}")
            
            # Generate test audio
            print("[4] Sending test audio...")
            audio = generate_test_audio()
            
            # Send audio in chunks
            chunk_size = 4096
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i+chunk_size]
                audio_base64 = base64.b64encode(chunk.tobytes()).decode('utf-8')
                
                await websocket.send(json.dumps({
                    "type": "audio",
                    "data": {"audio_base64": audio_base64}
                }))
            
            print(f"    ✅ Sent {len(audio)} samples")
            
            # Signal end of speaking
            print("[5] Processing turn...")
            await websocket.send(json.dumps({"type": "end_speaking"}))
            
            # Receive responses
            responses_received = []
            while True:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=120.0)
                    data = json.loads(response)
                    responses_received.append(data["type"])
                    
                    print(f"    📩 Received: {data['type']}")
                    
                    if data["type"] == "transcription":
                        print(f"       Text: {data['data']['text'][:50]}...")
                    
                    elif data["type"] == "emotion":
                        print(f"       Emotion: {data['data']['emotion']}, Mood: {data['data']['mood_score']}")
                        if data['data'].get('tip'):
                            print(f"       Tip: {data['data']['tip']}")
                    
                    elif data["type"] == "evaluation":
                        print(f"       Quality: {data['data']['quality']}")
                    
                    elif data["type"] == "response":
                        print(f"       Response: {data['data']['text'][:50]}...")
                    
                    elif data["type"] == "audio":
                        audio_len = len(data['data']['audio_base64'])
                        print(f"       Audio: {audio_len} bytes (base64)")
                    
                    elif data["type"] == "processing":
                        if data['data']['status'] == "completed":
                            break
                    
                except asyncio.TimeoutError:
                    print("    ⚠️ Timeout waiting for response")
                    break
            
            # End session
            print("[6] Ending session...")
            await websocket.send(json.dumps({"type": "end_session"}))
            
            response = await websocket.recv()
            data = json.loads(response)
            
            if data["type"] == "session_ended":
                turns = data['data'].get('total_turns', 0)
                print(f"    ✅ Session ended: {turns} turns")
            
            # Summary
            print("\n" + "=" * 50)
            print("WebSocket Test Summary")
            print("=" * 50)
            print(f"Messages received: {responses_received}")
            
            expected = ["processing", "transcription", "emotion", "evaluation", "response", "audio", "processing"]
            if all(msg in responses_received for msg in ["transcription", "response", "audio"]):
                print("✅ All critical messages received!")
            else:
                print("⚠️ Some messages missing")
            
    except Exception as e:
        print(f"    ❌ WebSocket error: {e}")
        import traceback
        traceback.print_exc()


async def main():
    print("=" * 50)
    print("VCAI WebSocket Test")
    print("=" * 50)
    
    # Step 1: Login
    token = register_and_login()
    if not token:
        print("❌ Cannot continue without token")
        return
    
    # Step 2: Create session
    session_id = create_session(token)
    if not session_id:
        print("❌ Cannot continue without session")
        return
    
    # Step 3: Test WebSocket
    await test_websocket(token, session_id)
    
    print("\n🎉 WebSocket test completed!")


if __name__ == "__main__":
    asyncio.run(main())