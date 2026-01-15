# VCAI - Virtual Customer AI Training System

<div align="center">

![VCAI Logo](https://img.shields.io/badge/VCAI-Virtual%20Customer%20AI-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.12+-green?style=flat-square&logo=python)
![React](https://img.shields.io/badge/React-19-blue?style=flat-square&logo=react)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-teal?style=flat-square&logo=fastapi)
![CUDA](https://img.shields.io/badge/CUDA-Optional-orange?style=flat-square&logo=nvidia)

**AI-powered sales training platform with real-time voice conversation**

[Features](#-features) • [Quick Start](#-quick-start) • [Setup Guide](#-detailed-setup-guide) • [Architecture](#-architecture) • [For Teammates](#-for-teammates)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [System Requirements](#-system-requirements)
- [Quick Start](#-quick-start)
- [Detailed Setup Guide](#-detailed-setup-guide)
- [Project Structure](#-project-structure)
- [Architecture](#-architecture)
- [For Teammates](#-for-teammates)
- [How to Replace Mock Functions](#-how-to-replace-mock-functions)
- [API Documentation](#-api-documentation)
- [Troubleshooting](#-troubleshooting)
- [GPU vs CPU Mode](#-gpu-vs-cpu-mode)

---

## 🎯 Overview

VCAI (Virtual Customer AI) is a training platform for real estate salespeople. It simulates realistic customer conversations in **Egyptian Arabic** using:

- **Speech-to-Text (STT)**: Converts salesperson speech to text using Whisper
- **Emotion Detection**: Analyzes customer emotional state
- **LLM Response Generation**: Creates realistic customer responses
- **Text-to-Speech (TTS)**: Converts responses to natural Egyptian Arabic speech
- **Real-time Evaluation**: Provides feedback on salesperson performance

### Use Case

A real estate salesperson practices handling difficult customers:
1. Salesperson speaks into microphone
2. System transcribes speech (STT)
3. AI customer responds based on persona and emotion
4. Salesperson gets real-time feedback and tips
5. Session ends with performance evaluation

---

## ✨ Features

| Feature | Status | Description |
|---------|--------|-------------|
| 🎤 Real-time STT | ✅ Working | Whisper large-v3-turbo model |
| 🗣️ TTS | 🔶 Mock | Egyptian Arabic voice synthesis |
| 😤 Emotion Detection | 🔶 Mock | Text + voice emotion analysis |
| 🤖 LLM Responses | 🔶 Mock | GPT-powered customer simulation |
| 📚 RAG | 🔶 Mock | Property/company knowledge retrieval |
| 🧠 Memory | 🔶 Mock | Conversation history & checkpoints |
| 👥 Personas | ✅ Working | 5 different customer personalities |
| 📊 Evaluation | 🔶 Mock | Session scoring & feedback |
| 🌐 Web Interface | ✅ Working | React-based training UI |
| 🔌 WebSocket | ✅ Working | Real-time audio streaming |

---

## 💻 System Requirements

### Minimum Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10/11, Ubuntu 20.04+ | Windows 11, Ubuntu 22.04 |
| CPU | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 |
| RAM | 8 GB | 16 GB |
| Storage | 10 GB free | 20 GB free |
| Python | 3.12.x | 3.12.x |
| Node.js | 18.x | 20.x or 22.x |

### For GPU Acceleration (Recommended)

| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA GTX 1060+ (6GB VRAM) |
| CUDA | 11.8 or 12.x |
| cuDNN | 8.x |

> ⚠️ **No GPU?** The system will automatically fall back to CPU mode. STT will be slower (~3-5x) but still functional.

---

## 🚀 Quick Start

For experienced developers who want to get running quickly:

```bash
# Clone repository
git clone https://github.com/your-org/VCAI.git
cd VCAI

# Backend setup
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Frontend setup
cd frontend
npm install
cd ..

# Start backend (Terminal 1)
python -m backend.main

# Start frontend (Terminal 2)
cd frontend
npm run dev

# Open browser
# http://localhost:5173
```

---

## 📖 Detailed Setup Guide

### Step 1: Install Python 3.12

<details>
<summary><b>Windows</b></summary>

1. Go to https://www.python.org/downloads/
2. Download **Python 3.12.x** (NOT 3.13)
3. Run the installer
4. ⚠️ **IMPORTANT**: Check ✅ "Add Python to PATH"
5. Click "Install Now"
6. Verify installation:
   ```powershell
   python --version
   # Should show: Python 3.12.x
   ```

</details>

<details>
<summary><b>Ubuntu/Linux</b></summary>

```bash
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.12 python3.12-venv python3.12-dev

# Verify
python3.12 --version
```

</details>

### Step 2: Install Node.js 20+

<details>
<summary><b>Windows</b></summary>

1. Go to https://nodejs.org/
2. Download **LTS version** (20.x or 22.x)
3. Run installer, accept defaults
4. Verify:
   ```powershell
   node --version
   # Should show: v20.x.x or v22.x.x
   
   npm --version
   # Should show: 10.x.x
   ```

</details>

<details>
<summary><b>Ubuntu/Linux</b></summary>

```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# Verify
node --version
npm --version
```

</details>

### Step 3: Install FFmpeg

FFmpeg is required for audio processing.

<details>
<summary><b>Windows</b></summary>

```powershell
# Using winget (recommended)
winget install ffmpeg

# Restart your terminal after installation!
# Verify:
ffmpeg -version
```

**Manual installation:**
1. Go to https://ffmpeg.org/download.html
2. Download Windows build
3. Extract to `C:\ffmpeg`
4. Add `C:\ffmpeg\bin` to your PATH environment variable

</details>

<details>
<summary><b>Ubuntu/Linux</b></summary>

```bash
sudo apt update
sudo apt install ffmpeg

# Verify
ffmpeg -version
```

</details>

### Step 4: Install CUDA (Optional - For GPU Acceleration)

<details>
<summary><b>Windows with NVIDIA GPU</b></summary>

1. Check your GPU:
   ```powershell
   nvidia-smi
   ```
   
2. If command not found, install NVIDIA drivers from https://www.nvidia.com/drivers

3. Install CUDA Toolkit:
   - Go to https://developer.nvidia.com/cuda-downloads
   - Select: Windows → x86_64 → 11 → exe (local)
   - Download and install (choose "Express" installation)

4. Verify:
   ```powershell
   nvcc --version
   # Should show CUDA version
   ```

</details>

<details>
<summary><b>Ubuntu/Linux with NVIDIA GPU</b></summary>

```bash
# Install NVIDIA drivers
sudo apt install nvidia-driver-535

# Install CUDA
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-1

# Add to PATH (add to ~/.bashrc)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Verify
nvcc --version
```

</details>

### Step 5: Clone and Setup Project

```powershell
# Clone the repository
git clone https://github.com/your-org/VCAI.git
cd VCAI

# Create Python virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
# source venv/bin/activate

# You should see (venv) in your terminal prompt
```

### Step 6: Install Python Dependencies

```powershell
# Make sure (venv) is active!
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt

# This will install:
# - fastapi, uvicorn (web server)
# - faster-whisper (STT)
# - torch (PyTorch for ML)
# - numpy, scipy (audio processing)
# - sqlalchemy (database)
# - and more...
```

**Expected output:**
```
Successfully installed fastapi-0.115.x uvicorn-0.32.x ...
```

### Step 7: Install Frontend Dependencies

```powershell
cd frontend
npm install

# This will install React, Tailwind, etc.
# Wait for it to complete...

cd ..
```

### Step 8: Initialize Database

```powershell
# The database will be created automatically on first run
# But you can verify the setup:
python -c "from backend.database import engine; print('Database OK')"
```

### Step 9: Start the Application

**Terminal 1 - Backend:**
```powershell
cd C:\VCAI  # or your project path
venv\Scripts\activate
python -m backend.main
```

Expected output:
```
============================================================
VCAI Backend Starting...
============================================================
[STT] Loading Faster-Whisper model...
[STT] CUDA available - using GPU    # or "using CPU" if no GPU
[STT] ✅ Model loaded in X.Xs
[Startup] Server ready at http://0.0.0.0:8000
============================================================
```

**Terminal 2 - Frontend:**
```powershell
cd C:\VCAI\frontend
npm run dev
```

Expected output:
```
  VITE v7.x.x  ready in XXX ms
  ➜  Local:   http://localhost:5173/
```

### Step 10: Access the Application

1. Open your browser
2. Go to **http://localhost:5173**
3. Register a new account or login
4. Start a training session!

---

## 📁 Project Structure

```
VCAI/
├── backend/                 # FastAPI backend
│   ├── main.py             # Entry point
│   ├── config.py           # Configuration
│   ├── database.py         # Database setup
│   ├── models/             # SQLAlchemy models
│   ├── routers/            # API endpoints
│   │   ├── auth.py         # Authentication
│   │   ├── sessions.py     # Training sessions
│   │   ├── personas.py     # Customer personas
│   │   └── websocket.py    # Real-time audio
│   ├── schemas/            # Pydantic schemas
│   └── services/           # Business logic
│
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/     # Reusable components
│   │   ├── context/        # React context (Auth)
│   │   ├── pages/          # Page components
│   │   │   ├── Login.jsx
│   │   │   ├── Dashboard.jsx
│   │   │   └── TrainingSession.jsx
│   │   └── services/       # API client
│   └── package.json
│
├── stt/                    # Speech-to-Text (Person A) ✅
│   └── realtime_stt.py     # Whisper implementation
│
├── tts/                    # Text-to-Speech (Person B) 🔶
│   └── (pending)
│
├── emotion/                # Emotion Detection (Person C) 🔶
│   └── (pending)
│
├── llm/                    # LLM Agent (Person D) 🔶
│   └── (pending)
│
├── rag/                    # RAG System (Person D) 🔶
│   └── (pending)
│
├── memory/                 # Memory Agent (Person D) 🔶
│   └── (pending)
│
├── persona/                # Persona Agent (Person B) 🔶
│   └── (pending)
│
├── orchestration/          # Main orchestration
│   ├── mocks/              # Mock implementations
│   │   ├── mock_tts.py
│   │   ├── mock_emotion.py
│   │   ├── mock_llm.py
│   │   ├── mock_rag.py
│   │   ├── mock_memory.py
│   │   └── mock_persona.py
│   └── graphs/             # LangGraph workflows
│
├── shared/                 # Shared types & interfaces
│   ├── interfaces.py       # Function signatures
│   └── types.py            # Type definitions
│
├── scripts/                # Utility scripts
│   ├── test_mocks.py       # Test all mocks
│   ├── test_stt.py         # Test STT
│   └── session_debugging.py
│
├── data/                   # Data files
│   ├── documents/          # RAG documents
│   ├── personas/           # Persona configs
│   └── voices/             # Voice samples
│
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

---

## 🏗️ Architecture

### System Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (React)                          │
│  ┌─────────┐  ┌─────────────┐  ┌──────────────┐                │
│  │  Login  │→ │  Dashboard  │→ │   Training   │                │
│  │  Page   │  │   Page      │  │   Session    │                │
│  └─────────┘  └─────────────┘  └──────┬───────┘                │
│                                        │ WebSocket               │
└────────────────────────────────────────┼────────────────────────┘
                                         │
┌────────────────────────────────────────┼────────────────────────┐
│                        BACKEND (FastAPI)                         │
│                                        │                         │
│  ┌─────────────────────────────────────▼───────────────────┐   │
│  │                   WebSocket Handler                      │   │
│  │  ┌──────┐  ┌─────────┐  ┌─────┐  ┌─────┐  ┌─────┐     │   │
│  │  │ STT  │→ │ Emotion │→ │ RAG │→ │ LLM │→ │ TTS │     │   │
│  │  └──────┘  └─────────┘  └─────┘  └─────┘  └─────┘     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐               │
│  │   Auth     │  │  Sessions  │  │  Personas  │               │
│  │   Router   │  │   Router   │  │   Router   │               │
│  └────────────┘  └────────────┘  └────────────┘               │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    SQLite Database                       │   │
│  └─────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

### Conversation Turn Flow

```
Salesperson Speaks
        │
        ▼
┌───────────────┐
│  1. STT       │  Whisper transcribes audio to Arabic text
│  (Person A)   │  transcribe_audio(audio) → text
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  2. Emotion   │  Detect emotion from text + audio
│  (Person C)   │  detect_emotion(text, audio) → emotion
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  3. RAG       │  Retrieve relevant property/company info
│  (Person D)   │  retrieve_context(text) → documents
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  4. Memory    │  Get conversation history
│  (Person D)   │  get_session_memory(session_id) → memory
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  5. LLM       │  Generate customer response
│  (Person D)   │  generate_response(...) → response_text
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  6. TTS       │  Convert response to speech
│  (Person B)   │  text_to_speech(text) → audio
└───────┬───────┘
        │
        ▼
Customer Responds (Audio plays)
```

---

## 👥 For Teammates

### Team Assignments

| Person | Components | Status |
|--------|----------|--------|
| **Person A (Ali)** | STT, Orchestration, Backend | ✅ Working |
| **Person B** | TTS, Persona Agent | 🔶 Pending |
| **Person C** | Emotion Detection, Emotional Agent | 🔶 Pending |
| **Person D** | RAG, Memory, LLM Agent | 🔶 Pending |

### Your Interface Functions

Each teammate must implement specific functions. See `shared/interfaces.py` for full documentation.

<details>
<summary><b>Person B Functions</b></summary>

```python
# Location: tts/tts_agent.py

def text_to_speech(text: str, voice_id: str = "default", emotion: str = "neutral") -> np.ndarray:
    """
    Convert Arabic text to speech audio.
    
    Args:
        text: Arabic text to speak (max 500 chars)
        voice_id: "default", "egyptian_male_01", "egyptian_female_01"
        emotion: "neutral", "happy", "frustrated", "interested", "hesitant"
    
    Returns:
        np.ndarray: Audio (float32, 22050 Hz sample rate)
    """
    pass

# Location: persona/persona_agent.py

def get_persona(persona_id: str) -> Persona:
    """Get full persona configuration."""
    pass

def list_personas() -> list[PersonaSummary]:
    """List all available personas."""
    pass
```

</details>

<details>
<summary><b>Person C Functions</b></summary>

```python
# Location: emotion/emotion_agent.py

def detect_emotion(text: str, audio: np.ndarray) -> EmotionResult:
    """
    Detect emotion from text and audio.
    
    Args:
        text: Arabic transcription
        audio: Raw audio (float32, 16000 Hz)
    
    Returns:
        EmotionResult: {
            "primary_emotion": str,  # "happy", "angry", "neutral", etc.
            "confidence": float,     # 0.0 to 1.0
            "voice_emotion": str,
            "text_emotion": str,
            "intensity": str,        # "low", "medium", "high"
            "scores": dict           # Emotion probabilities
        }
    """
    pass

def analyze_emotional_context(current_emotion: EmotionResult, history: list[Message]) -> EmotionalContext:
    """
    Analyze emotional trends over conversation.
    
    Returns:
        EmotionalContext: {
            "current": EmotionResult,
            "trend": str,            # "improving", "worsening", "stable"
            "recommendation": str,   # "be_gentle", "be_firm", "show_empathy"
            "risk_level": str        # "low", "medium", "high"
        }
    """
    pass
```

</details>

<details>
<summary><b>Person D Functions</b></summary>

```python
# Location: rag/rag_agent.py

def retrieve_context(query: str, top_k: int = 3) -> RAGContext:
    """
    Retrieve relevant documents for query.
    
    Returns:
        RAGContext: {
            "query": str,
            "documents": [{"content": str, "source": str, "score": float}, ...],
            "total_found": int
        }
    """
    pass

# Location: memory/memory_agent.py

def store_message(session_id: str, message: Message) -> bool:
    """Store a conversation message."""
    pass

def get_recent_messages(session_id: str, last_n: int = 10) -> list[Message]:
    """Get recent messages from session."""
    pass

def get_session_memory(session_id: str) -> SessionMemory:
    """Get full session memory (checkpoints + recent messages)."""
    pass

# Location: llm/llm_agent.py

def generate_response(
    customer_text: str,
    emotion: EmotionResult,
    emotional_context: EmotionalContext,
    persona: Persona,
    memory: SessionMemory,
    rag_context: RAGContext
) -> str:
    """
    Generate customer response using LLM.
    
    Returns:
        str: Response in Egyptian Arabic
    """
    pass
```

</details>

---

## 🔄 How to Replace Mock Functions

### Step-by-Step Guide

When your component is ready, follow these steps to replace the mock:

#### Step 1: Verify Your Function Matches the Interface

Open `shared/interfaces.py` and ensure your function:
- Has the **exact same name**
- Has the **exact same parameters**
- Returns the **exact same type**

```python
# Example: Your function should match this exactly
def detect_emotion(text: str, audio: np.ndarray) -> EmotionResult:
```

#### Step 2: Test Your Function Independently

Create a test script:

```python
# test_my_function.py
from your_module.your_file import your_function

# Test with sample data
result = your_function(sample_input)
print(f"Result: {result}")
assert isinstance(result, ExpectedType)
print("✅ Test passed!")
```

#### Step 3: Find Where the Mock is Called

Search for the mock import in `backend/routers/websocket.py`:

```powershell
Select-String -Path "C:\VCAI\backend\routers\websocket.py" -Pattern "from orchestration.mocks"
```

#### Step 4: Replace the Import

**Before (Mock):**
```python
from orchestration.mocks import detect_emotion
```

**After (Real):**
```python
from emotion.emotion_agent import detect_emotion
```

#### Step 5: Test the Integration

1. Start the backend:
   ```powershell
   python -m backend.main
   ```

2. Start the frontend:
   ```powershell
   cd frontend
   npm run dev
   ```

3. Test a full conversation session

4. Check the backend logs for errors

#### Step 6: Handle Errors

If something breaks:

1. Check the backend terminal for error messages
2. Verify your function signature matches the interface
3. Verify your return type matches expected type
4. Add debug prints:
   ```python
   print(f"[YOUR_MODULE] Input: {input_data}")
   print(f"[YOUR_MODULE] Output: {output_data}")
   ```

### Quick Reference: Where to Replace

| Component | Mock Location | Replace With |
|-----------|--------------|--------------|
| TTS | `from orchestration.mocks import text_to_speech` | `from tts.tts_agent import text_to_speech` |
| Emotion | `from orchestration.mocks import detect_emotion` | `from emotion.emotion_agent import detect_emotion` |
| RAG | `from orchestration.mocks import retrieve_context` | `from rag.rag_agent import retrieve_context` |
| Memory | `from orchestration.mocks import get_session_memory` | `from memory.memory_agent import get_session_memory` |
| LLM | `from orchestration.mocks import generate_response` | `from llm.llm_agent import generate_response` |

### Example: Replacing Emotion Detection

```python
# File: backend/routers/websocket.py
# Line ~250

# BEFORE:
if settings.use_mocks:
    from orchestration.mocks import detect_emotion
    emotion_result = detect_emotion(results["transcription"], audio)
else:
    from orchestration.mocks import detect_emotion  # Still mock!
    emotion_result = detect_emotion(results["transcription"], audio)

# AFTER:
if settings.use_mocks:
    from orchestration.mocks import detect_emotion
    emotion_result = detect_emotion(results["transcription"], audio)
else:
    from emotion.emotion_agent import detect_emotion  # Real implementation!
    emotion_result = detect_emotion(results["transcription"], audio)
```

---

## 📚 API Documentation

When backend is running, visit: **http://localhost:8000/docs**

### Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/register` | Register new user |
| POST | `/api/auth/login` | Login user |
| GET | `/api/auth/me` | Get current user |
| GET | `/api/personas` | List all personas |
| POST | `/api/sessions` | Create training session |
| GET | `/api/sessions` | List user sessions |
| WS | `/ws/{session_id}` | WebSocket for real-time audio |

### WebSocket Message Types

**Client → Server:**
```json
{"type": "audio_complete", "data": {"audio_base64": "...", "format": "webm"}}
{"type": "end_session"}
{"type": "ping"}
```

**Server → Client:**
```json
{"type": "connected", "data": {"session_id": "...", "persona": {...}}}
{"type": "transcription", "data": {"text": "..."}}
{"type": "emotion", "data": {"emotion": "...", "mood_score": 50, "risk_level": "low"}}
{"type": "response", "data": {"text": "..."}}
{"type": "audio", "data": {"audio_base64": "...", "sample_rate": 22050}}
{"type": "processing", "data": {"status": "started|completed"}}
{"type": "error", "data": {"message": "..."}}
```

---

## 🔧 Troubleshooting

### Common Issues

<details>
<summary><b>Backend won't start: "Module not found"</b></summary>

**Problem:** Python can't find modules.

**Solution:**
```powershell
# Make sure you're in the project root
cd C:\VCAI

# Make sure venv is activated
venv\Scripts\activate

# Reinstall dependencies
pip install -r requirements.txt
```

</details>

<details>
<summary><b>Frontend stuck on "Loading..."</b></summary>

**Problem:** Auth token is corrupted.

**Solution:**
1. Open browser DevTools (F12)
2. Go to Application → Local Storage
3. Clear all items for localhost:5173
4. Refresh the page

</details>

<details>
<summary><b>STT returns empty transcription</b></summary>

**Problem:** Microphone not working or volume too low.

**Solution:**
1. Check microphone in Windows Sound Settings
2. Increase microphone volume to 100%
3. Enable "Microphone Boost" (+20dB)
4. Test with: `python scripts/test_mic_quality.py`

</details>

<details>
<summary><b>CUDA error / GPU not detected</b></summary>

**Problem:** CUDA not properly installed.

**Solution:**
```powershell
# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# If False, the system will use CPU (slower but works)
# To fix GPU:
# 1. Update NVIDIA drivers
# 2. Reinstall CUDA toolkit
# 3. Reinstall PyTorch:
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

</details>

<details>
<summary><b>WebSocket connection fails (403)</b></summary>

**Problem:** Token expired or invalid.

**Solution:**
1. Log out and log back in
2. Clear browser local storage
3. Check that backend is running

</details>

<details>
<summary><b>"Cannot subtract offset-naive and offset-aware datetimes"</b></summary>

**Problem:** Timezone handling bug.

**Solution:** This is a known issue in session_service.py. The session still works, just ignore the error for now.

</details>

### Debug Commands

```powershell
# Test STT with file
python scripts/test_stt.py

# Test microphone quality
python scripts/test_mic_quality.py

# Test all mocks
python scripts/test_mocks.py

# Test WebSocket session
python scripts/session_debugging.py

# Check CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 🖥️ GPU vs CPU Mode

### Automatic Detection

The system automatically detects if CUDA is available:

```python
# stt/realtime_stt.py
if torch.cuda.is_available():
    device = "cuda"      # Use GPU
    compute_type = "float16"
else:
    device = "cpu"       # Fallback to CPU
    compute_type = "int8"
```

### Performance Comparison

| Mode | STT Speed | Model Load | VRAM Usage |
|------|-----------|------------|------------|
| GPU (CUDA) | ~0.3-1s per turn | ~5s | ~4GB |
| CPU | ~2-5s per turn | ~15s | 0 |

### Force CPU Mode

If you have GPU issues, you can force CPU mode:

```python
# In stt/realtime_stt.py, change:
load_model(force_cpu=True)
```

Or set environment variable:
```powershell
$env:CUDA_VISIBLE_DEVICES = ""
python -m backend.main
```

---

## 📝 Environment Variables

Create a `.env` file in the project root (optional):

```env
# Database
DATABASE_URL=sqlite:///./vcai.db

# JWT Secret (change in production!)
SECRET_KEY=your-secret-key-change-in-production

# Use mock implementations
USE_MOCKS=true

# Debug mode
DEBUG=true
```

---

## 🧪 Testing

### Run All Tests

```powershell
# Test mocks
python scripts/test_mocks.py

# Test STT
python scripts/test_stt.py

# Test full session flow
python scripts/session_debugging.py
```

### Manual Testing

1. Start backend and frontend
2. Register/login
3. Select a persona
4. Start session
5. Speak into microphone
6. Verify:
   - Transcription appears
   - Customer responds
   - Emotion indicator updates

---

## 📦 Dependencies

### Python (requirements.txt)

```
fastapi==0.115.5
uvicorn==0.32.1
sqlalchemy==2.0.36
pydantic==2.10.2
python-jose==3.3.0
passlib==1.7.4
bcrypt==4.2.1
python-multipart==0.0.17
faster-whisper==1.1.0
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
websockets>=12.0
```

### Frontend (package.json)

```json
{
  "dependencies": {
    "react": "^19.0.0",
    "react-dom": "^19.0.0",
    "react-router-dom": "^7.1.1",
    "axios": "^1.7.9"
  },
  "devDependencies": {
    "vite": "^7.0.0",
    "tailwindcss": "^4.0.0",
    "@vitejs/plugin-react": "^4.3.4"
  }
}
```

---

## 📄 License

This project is for educational purposes as part of [Your University/Course Name].

---

## 🤝 Contributing

1. Pull latest changes: `git pull origin main`
2. Create feature branch: `git checkout -b feature/your-feature`
3. Make changes
4. Test thoroughly
5. Commit: `git commit -m "Add your feature"`
6. Push: `git push origin feature/your-feature`
7. Create Pull Request

---

## 📞 Support

- **Ali (Person A)**: STT, Backend, Integration issues
- **Person B**: TTS, Persona issues
- **Person C**: Emotion detection issues
- **Person D**: RAG, Memory, LLM issues

---

<div align="center">

**Built with ❤️ for sales training excellence**

</div>