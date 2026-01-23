# VCAI - Virtual Customer AI Training System

<div align="center">

![VCAI](https://img.shields.io/badge/VCAI-Virtual%20Customer%20AI-0066CC?style=for-the-badge&labelColor=000000)
![Version](https://img.shields.io/badge/version-1.0.0-green?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)

**AI-powered sales training platform with real-time voice conversations in Egyptian Arabic**

[Overview](#-overview) · [Features](#-features) · [Demo](#-demo) · [Installation](#-installation) · [Architecture](#-architecture) · [API](#-api-documentation)

---

</div>

## 🎯 Overview

VCAI (Virtual Customer AI) is an intelligent training platform designed for real estate sales professionals. It simulates realistic customer interactions in **Egyptian Arabic**, providing a safe environment to practice handling various customer personalities and scenarios.

### The Problem

Traditional sales training relies on role-playing with colleagues or managers, which is:
- **Inconsistent** - Different trainers provide different experiences
- **Limited** - Can't practice 24/7
- **Biased** - Colleagues may not act like real difficult customers
- **Expensive** - Requires dedicated training time from senior staff

### The Solution

VCAI provides an AI-powered virtual customer that:
- **Responds naturally** in Egyptian Arabic dialect
- **Adapts emotionally** based on the conversation flow
- **Simulates different personalities** from friendly to difficult customers
- **Provides instant feedback** on sales techniques
- **Available 24/7** for unlimited practice sessions

---

## ✨ Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| 🎤 **Real-time Speech Recognition** | Powered by Whisper large-v3-turbo with GPU acceleration |
| 🗣️ **Natural Voice Responses** | Egyptian Arabic text-to-speech synthesis |
| 😤 **Emotion Detection** | Analyzes voice tone and text sentiment |
| 🤖 **Intelligent Responses** | Context-aware customer simulation using fine-tuned LLM |
| 📚 **Knowledge Retrieval** | RAG system with property and company information |
| 🧠 **Conversation Memory** | Maintains context with automatic checkpointing |
| 👥 **Multiple Personas** | 5 distinct customer personalities |
| 📊 **Performance Analytics** | Session scoring and improvement tracking |

### Customer Personas

| Persona | Personality | Challenge Level |
|---------|-------------|-----------------|
| 🧐 **Price-Focused Customer** | Primarily concerned with getting the best deal | Medium |
| 😤 **Difficult Customer** | Skeptical, hard to please, raises objections | Hard |
| 😊 **Friendly Customer** | Open and cooperative, easy to work with | Easy |
| ⏰ **Rushed Customer** | Limited time, wants quick answers | Medium |
| 🔬 **Detail-Oriented Customer** | Asks many technical questions | Hard |

---

## 🎬 Demo

### Training Session Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     Training Session                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Salesperson: "السلام عليكم، معاك أحمد من شركة العقارات"      │
│               (Hello, this is Ahmed from the real estate     │
│                company)                                      │
│                                                              │
│  ─────────────────────────────────────────────────────────  │
│                                                              │
│  🎭 Customer (Price-Focused):                                │
│     "وعليكم السلام، أنا عايز أعرف أسعار الشقق عندكم،         │
│      بس مش عايز حاجة غالية"                                  │
│     (Hello, I want to know your apartment prices,           │
│      but I don't want anything expensive)                   │
│                                                              │
│  📊 Emotion: Interested │ Mood: 65% │ Risk: Low             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 💻 System Requirements

### Minimum Specifications

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Windows 10/11, Ubuntu 20.04+ | Windows 11, Ubuntu 22.04 |
| **CPU** | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 |
| **RAM** | 8 GB | 16 GB |
| **Storage** | 10 GB | 20 GB SSD |
| **Python** | 3.12.x | 3.12.x |
| **Node.js** | 18.x | 20.x+ |

### GPU Acceleration (Recommended)

| Component | Requirement |
|-----------|-------------|
| **GPU** | NVIDIA GTX 1060+ (6GB VRAM) |
| **CUDA** | 11.8 or 12.x |
| **cuDNN** | 8.x |

> 💡 **No GPU?** VCAI automatically falls back to CPU mode. Speech recognition will be slower (~3-5x) but fully functional.

---

## 🚀 Installation

### Prerequisites

Ensure you have installed:
- [Python 3.12](https://www.python.org/downloads/)
- [Node.js 20+](https://nodejs.org/)
- [FFmpeg](https://ffmpeg.org/download.html)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (optional, for GPU)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/your-org/VCAI.git
cd VCAI

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install backend dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend && npm install && cd ..

# Start the application
# Terminal 1 - Backend
python -m backend.main

# Terminal 2 - Frontend
cd frontend && npm run dev
```

### Access the Application

Open your browser and navigate to: **http://localhost:5173**

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Client Layer                                 │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                    React Frontend (Vite)                       │ │
│  │   Dashboard │ Training Session │ Analytics │ Settings          │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ WebSocket / REST API
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         API Layer                                    │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                   FastAPI Backend                              │ │
│  │   Authentication │ Sessions │ Personas │ WebSocket Handler     │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Orchestration Layer                             │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                  LangGraph Pipeline                            │ │
│  │                                                                 │ │
│  │   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐   │ │
│  │   │ Memory  │───▶│   STT   │───▶│ Emotion │───▶│   RAG   │   │ │
│  │   │  Load   │    │         │    │         │    │         │   │ │
│  │   └─────────┘    └─────────┘    └─────────┘    └─────────┘   │ │
│  │                                                      │         │ │
│  │   ┌─────────┐    ┌─────────┐    ┌─────────┐         │         │ │
│  │   │ Memory  │◀───│   TTS   │◀───│   LLM   │◀────────┘         │ │
│  │   │  Save   │    │         │    │         │                   │ │
│  │   └─────────┘    └─────────┘    └─────────┘                   │ │
│  │                                                                 │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         AI Services                                  │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │     STT      │  │   Emotion    │  │     LLM      │              │
│  │   Whisper    │  │  Detection   │  │  Fine-tuned  │              │
│  │ large-v3-tb  │  │    Model     │  │    Arabic    │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │     TTS      │  │     RAG      │  │    Memory    │              │
│  │   Egyptian   │  │   ChromaDB   │  │   PostgreSQL │              │
│  │    Arabic    │  │   Embeddings │  │   /SQLite    │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Conversation Turn Pipeline

| Step | Component | Description | Latency |
|------|-----------|-------------|---------|
| 1 | **Memory Load** | Retrieve conversation history and checkpoints | ~5ms |
| 2 | **STT** | Transcribe Arabic speech using Whisper | ~300-500ms |
| 3 | **Emotion** | Analyze emotional state from voice + text | ~50ms |
| 4 | **RAG** | Retrieve relevant property information | ~100ms |
| 5 | **LLM** | Generate contextual customer response | ~500-800ms |
| 6 | **TTS** | Synthesize Egyptian Arabic speech | ~200ms |
| 7 | **Memory Save** | Store messages and create checkpoints | ~10ms |

**Total Turn Latency:** ~1-2 seconds (GPU) / ~3-5 seconds (CPU)

---

## 📁 Project Structure

```
VCAI/
├── backend/                    # FastAPI Backend
│   ├── main.py                # Application entry point
│   ├── config.py              # Configuration management
│   ├── database.py            # Database connection
│   ├── models/                # SQLAlchemy ORM models
│   ├── routers/               # API route handlers
│   ├── schemas/               # Pydantic validation schemas
│   └── services/              # Business logic services
│
├── frontend/                   # React Frontend
│   ├── src/
│   │   ├── components/        # Reusable UI components
│   │   ├── pages/             # Page components
│   │   ├── context/           # React context providers
│   │   └── services/          # API client services
│   └── package.json
│
├── orchestration/              # LangGraph Orchestration
│   ├── agent.py               # Main orchestration agent
│   ├── state.py               # Conversation state management
│   ├── config.py              # Pipeline configuration
│   ├── graphs/                # LangGraph workflow definitions
│   ├── nodes/                 # Individual pipeline nodes
│   └── mocks/                 # Mock implementations for testing
│
├── stt/                        # Speech-to-Text Module
│   └── realtime_stt.py        # Whisper implementation
│
├── tts/                        # Text-to-Speech Module
│   └── agent.py               # TTS implementation
│
├── emotion/                    # Emotion Detection Module
│   └── agent.py               # Emotion classifier
│
├── llm/                        # Language Model Module
│   ├── agent.py               # Response generation
│   └── prompts.py             # Prompt templates
│
├── rag/                        # Retrieval-Augmented Generation
│   └── agent.py               # Document retrieval
│
├── memory/                     # Conversation Memory
│   ├── agent.py               # Memory interface
│   └── store.py               # Database operations
│
├── persona/                    # Customer Personas
│   └── agent.py               # Persona management
│
├── shared/                     # Shared Utilities
│   ├── types.py               # TypedDict definitions
│   ├── constants.py           # Application constants
│   └── interfaces.py          # Function signatures
│
├── scripts/                    # Utility Scripts
│   └── tests/                 # Test scripts
│
├── data/                       # Data Files
│   ├── documents/             # RAG knowledge base
│   ├── personas/              # Persona configurations
│   └── models/                # Trained model weights
│
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 📚 API Documentation

### Interactive Documentation

When the backend is running, access the interactive API docs at:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/auth/register` | Register new user |
| `POST` | `/api/auth/login` | Authenticate user |
| `GET` | `/api/auth/me` | Get current user profile |
| `GET` | `/api/personas` | List available personas |
| `GET` | `/api/personas/{id}` | Get persona details |
| `POST` | `/api/sessions` | Create training session |
| `GET` | `/api/sessions` | List user sessions |
| `GET` | `/api/sessions/{id}` | Get session details |

### WebSocket Protocol

**Endpoint:** `ws://localhost:8000/ws/{session_id}?token={jwt_token}`

#### Client → Server Messages

```json
{
  "type": "audio_complete",
  "data": {
    "audio_base64": "...",
    "format": "webm"
  }
}
```

```json
{
  "type": "end_session"
}
```

#### Server → Client Messages

```json
{
  "type": "transcription",
  "data": {
    "text": "السلام عليكم"
  }
}
```

```json
{
  "type": "response",
  "data": {
    "text": "وعليكم السلام، أقدر أساعدك في إيه؟"
  }
}
```

```json
{
  "type": "emotion",
  "data": {
    "emotion": "interested",
    "mood_score": 65,
    "risk_level": "low"
  }
}
```

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Database
DATABASE_URL=sqlite:///./vcai.db

# Security
SECRET_KEY=your-secure-secret-key-here

# Feature Flags
USE_MOCKS=false
DEBUG=false

# Model Settings
STT_MODEL=large-v3-turbo
LLM_MODEL=your-model-path
```

### Runtime Configuration

Key settings in `backend/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `use_mocks` | `false` | Use mock implementations |
| `checkpoint_interval` | `5` | Turns between checkpoints |
| `recent_messages_count` | `10` | Messages to include in context |
| `rag_top_k` | `3` | Number of documents to retrieve |

---

## 🔧 Troubleshooting

### Common Issues

<details>
<summary><b>🔴 Backend fails to start</b></summary>

**Symptoms:** Module not found errors

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```
</details>

<details>
<summary><b>🔴 CUDA not detected</b></summary>

**Symptoms:** Running on CPU despite having NVIDIA GPU

**Solution:**
```bash
# Verify CUDA installation
nvidia-smi
nvcc --version

# Reinstall PyTorch with CUDA support
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```
</details>

<details>
<summary><b>🔴 Microphone not working</b></summary>

**Symptoms:** Empty transcriptions or "No audio detected"

**Solution:**
1. Check microphone permissions in browser
2. Verify microphone in system settings
3. Increase microphone volume and enable boost
4. Test with: `python scripts/test_mic.py`
</details>

<details>
<summary><b>🔴 WebSocket connection fails</b></summary>

**Symptoms:** 403 Forbidden or connection refused

**Solution:**
1. Ensure backend is running on port 8000
2. Check JWT token validity (re-login if needed)
3. Clear browser local storage and refresh
</details>

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Test specific component
python scripts/test_stt.py
python scripts/test_emotion.py
python scripts/test_memory.py

# Test full conversation flow
python scripts/test_full_pipeline.py
```

---

## 📈 Performance Metrics

### Benchmarks (NVIDIA RTX 3060)

| Metric | Value |
|--------|-------|
| STT Latency | 300-500ms |
| LLM Response | 500-800ms |
| Total Turn Time | 1.2-1.5s |
| Concurrent Sessions | 5-10 |
| Memory Usage | ~4GB VRAM |

### Optimization Tips

1. **Enable GPU acceleration** for 3-5x faster inference
2. **Use SSD storage** for faster model loading
3. **Increase checkpoint interval** to reduce database writes
4. **Limit conversation history** to most recent messages

---

## 🛣️ Roadmap

- [x] Core conversation pipeline
- [x] Real-time speech recognition
- [x] LangGraph orchestration
- [x] Conversation memory with checkpoints
- [ ] Enhanced emotion detection model
- [ ] Egyptian Arabic TTS integration
- [ ] RAG with property database
- [ ] Performance analytics dashboard
- [ ] Multi-language support
- [ ] Mobile application

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) for speech recognition
- [LangGraph](https://github.com/langchain-ai/langgraph) for orchestration
- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework
- [React](https://react.dev/) for the frontend framework

---

<div align="center">

**Built with ❤️ for sales excellence**

[Report Bug](https://github.com/your-org/VCAI/issues) · [Request Feature](https://github.com/your-org/VCAI/issues)

</div>
