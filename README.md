# VCAI - Virtual Customer AI Training System

<div align="center">

![VCAI](https://img.shields.io/badge/VCAI-Virtual%20Customer%20AI-0066CC?style=for-the-badge&labelColor=000000)
![Version](https://img.shields.io/badge/version-1.0.0-green?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)

*AI-powered sales training platform with real-time voice conversations in Egyptian Arabic*

[Overview](#-overview) · [Features](#-features) · [Demo](#-demo) · [Installation](#-installation) · [Architecture](#-architecture) · [API](#-api-documentation)

---

</div>

## 🎯 Overview

VCAI (Virtual Customer AI) is an intelligent training platform designed for real estate sales professionals. It simulates realistic customer interactions in *Egyptian Arabic*, providing a safe environment to practice handling various customer personalities and scenarios.

### The Problem

Traditional sales training relies on role-playing with colleagues or managers, which is:
- *Inconsistent* - Different trainers provide different experiences
- *Limited* - Can't practice 24/7
- *Biased* - Colleagues may not act like real difficult customers
- *Expensive* - Requires dedicated training time from senior staff

### The Solution

VCAI provides an AI-powered virtual customer that:
- *Responds naturally* in Egyptian Arabic dialect
- *Adapts emotionally* based on the conversation flow
- *Simulates different personalities* from friendly to difficult customers
- *Provides instant feedback* on sales techniques
- *Available 24/7* for unlimited practice sessions

---

## ✨ Features

### Core Capabilities

| Feature | Technology | Status |
|---------|------------|--------|
| 🎤 *Real-time Speech Recognition* | Faster-Whisper large-v3-turbo (GPU) | ✅ Working |
| 🗣️ *Egyptian Arabic TTS* | Chatterbox Multilingual, fine-tuned on Egyptian data | ✅ Working |
| 😤 *Emotion Detection* | Custom-trained emotion2vec + AraBERT fusion (96.8% accuracy) | ✅ Working |
| 🤖 *Intelligent Responses* | Qwen 2.5-7B-Instruct, 4-bit quantized (BitsAndBytes NF4) | ✅ Working |
| 📚 *Knowledge Retrieval* | ChromaDB + Sentence-Transformers RAG | 🟡 In Progress |
| 🧠 *Conversation Memory* | PostgreSQL with automatic checkpointing every 5 turns | ✅ Working |
| 🔊 *Streaming Audio* | LLM→TTS sentence-level streaming for low perceived latency | ✅ Working |
| 👥 *Multiple Personas* | 5 distinct customer personalities | ✅ Working |

### Customer Personas

| Persona | Personality | Challenge Level |
|---------|-------------|-----------------|
| 🧐 *Price-Focused Customer* | Primarily concerned with getting the best deal | Medium |
| 😤 *Difficult Customer* | Skeptical, hard to please, raises objections | Hard |
| 😊 *Friendly Customer* | Open and cooperative, easy to work with | Easy |
| ⏰ *Rushed Customer* | Limited time, wants quick answers | Medium |
| 🔬 *Detail-Oriented Customer* | Asks many technical questions | Hard |

---

## 🎬 Demo

### Training Session Flow


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


---

## 💻 System Requirements

### Minimum Specifications

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| *OS* | Windows 10/11, Ubuntu 20.04+ | Windows 11, Ubuntu 22.04 |
| *CPU* | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 |
| *RAM* | 16 GB | 32 GB |
| *GPU* | NVIDIA GTX 1060 (6GB VRAM) | NVIDIA RTX 3060+ (12GB VRAM) |
| *Storage* | 20 GB SSD | 40 GB SSD |
| *CUDA* | 12.1 | 12.1+ |
| *Python* | 3.11.x | 3.11.x |
| *Node.js* | 18.x | 20.x+ |

> ⚠️ *GPU is required.* VCAI loads multiple models simultaneously (STT, LLM, TTS, Emotion) which require ~10GB VRAM total.

---

## 🚀 Installation

### Prerequisites

Ensure you have installed:
- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [PostgreSQL](https://www.postgresql.org/download/) (create a database named vcai)
- [Node.js 20+](https://nodejs.org/)
- [FFmpeg](https://ffmpeg.org/download.html)
- [CUDA Toolkit 12.1+](https://developer.nvidia.com/cuda-downloads)
- NVIDIA GPU drivers (latest)

### Step-by-Step Setup

bash
# 1. Clone the repository
git clone https://github.com/your-org/VCAI.git
cd VCAI

# 2. Create conda environment (Python 3.11 required for Chatterbox)
conda create -n vcai python=3.11 -y
conda activate vcai

# 3. Install Chatterbox TTS first (has specific dependency requirements)
pip install chatterbox-tts

# 4. Reinstall PyTorch with CUDA support (chatterbox may install CPU-only)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Install remaining project dependencies
pip install -r requirements.txt

# 6. Install frontend dependencies
cd frontend && npm install && cd ..

# 7. Start the application
# Terminal 1 — Backend
python -m backend.main

# Terminal 2 — Frontend
cd frontend && npm run dev


### Verify Installation

bash
# Check CUDA is available
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"

# Check bitsandbytes
python -c "import bitsandbytes; print('BitsAndBytes OK')"

# Check chatterbox
python -c "from chatterbox.mtl_tts import ChatterboxMultilingualTTS; print('Chatterbox OK')"


### Access the Application

Open your browser and navigate to: *http://localhost:5173*

---

## 🏗️ Architecture

### System Overview


┌─────────────────────────────────────────────────────────────────────┐
│                         Client Layer                                 │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                    React Frontend (Vite)                       │ │
│  │   Dashboard │ Training Session │ Session Setup │ Login        │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ WebSocket (streaming audio chunks)
                                   │ REST API (auth, sessions, personas)
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
│  │              LangGraph Pipeline (Streaming)                    │ │
│  │                                                                 │ │
│  │   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐   │ │
│  │   │ Memory  │───▶│   STT   │───▶│ Emotion │───▶│   RAG   │   │ │
│  │   │  Load   │    │ Whisper │    │ Fusion  │    │ ChromaDB│   │ │
│  │   └─────────┘    └─────────┘    └─────────┘    └─────────┘   │ │
│  │                                                      │         │ │
│  │   ┌─────────┐    ┌─────────┐    ┌─────────┐         │         │ │
│  │   │ Memory  │◀───│   TTS   │◀──▶│   LLM   │◀────────┘         │ │
│  │   │  Save   │    │Chatter- │    │  Qwen   │  (streaming)     │ │
│  │   └─────────┘    │  box    │    │ 2.5-7B  │                   │ │
│  │                   └─────────┘    └─────────┘                   │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘


### Streaming Pipeline

VCAI uses a sentence-level streaming architecture for low perceived latency:


User speaks → STT (0.3s) → Emotion (0.06s) → RAG → LLM starts generating
  → Sentence 1 complete → TTS chunk 1 → Send to browser → PLAY immediately
  → Sentence 2 complete → TTS chunk 2 → Send to browser → PLAY next
  → ...
User hears first audio at ~2.5s instead of ~5.5s (55% faster perceived latency)


### Conversation Turn Pipeline

| Step | Component | Technology | Latency |
|------|-----------|------------|---------|
| 1 | *Memory Load* | PostgreSQL + checkpoints | ~5ms |
| 2 | *STT* | Faster-Whisper large-v3-turbo | ~300-450ms |
| 3 | *Emotion* | emotion2vec + AraBERT fusion | ~55-60ms |
| 4 | *RAG* | ChromaDB + sentence-transformers | ~100ms |
| 5 | *LLM* | Qwen 2.5-7B (4-bit NF4) | ~1-3s (streamed) |
| 6 | *TTS* | Chatterbox Multilingual (Egyptian fine-tuned) | ~1.5-3s per chunk |
| 7 | *Memory Save* | PostgreSQL + LLM summarization | ~5ms (8s on checkpoint) |

---

## 📁 Project Structure


VCAI/
├── backend/                    # FastAPI Backend
│   ├── main.py                # Entry point + ML model preloading
│   ├── config.py              # Configuration management
│   ├── database.py            # Database connection
│   ├── models/                # SQLAlchemy ORM models
│   ├── routers/               # API routes + WebSocket handler
│   ├── schemas/               # Pydantic validation schemas
│   └── services/              # Business logic services
│
├── frontend/                   # React Frontend (Vite)
│   └── src/
│       ├── components/        # Reusable UI components
│       ├── pages/             # Page components
│       │   ├── TrainingSession.jsx  # Main training UI + audio streaming
│       │   ├── SessionSetup.jsx     # Persona selection
│       │   ├── Dashboard.jsx        # Session history
│       │   └── Login.jsx / Register.jsx
│       ├── context/           # Auth context provider
│       └── services/          # API + WebSocket client
│
├── orchestration/              # LangGraph Orchestration
│   ├── agent.py               # Main orchestration agent
│   ├── state.py               # ConversationState (TypedDict)
│   ├── config.py              # Pipeline configuration
│   ├── graphs/                # LangGraph workflow definitions
│   │   └── conversation_graph.py  # Main pipeline graph
│   └── nodes/                 # Individual pipeline nodes
│       ├── stt_node.py
│       ├── emotion_node.py
│       ├── rag_node.py
│       ├── llm_node.py        # + llm_node_streaming() generator
│       ├── tts_node.py        # + tts_chunk() for streaming
│       └── memory_node.py     # load + save
│
├── stt/                        # Speech-to-Text
│   └── realtime_stt.py        # Faster-Whisper implementation
│
├── tts/                        # Text-to-Speech
│   ├── agent.py               # TTS interface + Egyptian checkpoint loading
│   └── chatterbox_model.py    # Chatterbox wrapper class
│
├── emotion/                    # Emotion Detection
│   ├── agent.py               # Emotion analysis orchestrator
│   ├── voice_emotion.py       # emotion2vec voice classifier
│   ├── text_emotion.py        # AraBERT text sentiment
│   └── fusion.py              # Voice + text emotion fusion
│
├── llm/                        # Language Model
│   ├── agent.py               # Qwen 2.5-7B with streaming support
│   └── prompts.py             # System prompt templates
│
├── rag/                        # Retrieval-Augmented Generation
│   ├── agent.py               # RAG interface
│   ├── embeddings.py          # Embedding model
│   ├── vector_store.py        # ChromaDB operations
│   └── document_loader.py     # Document ingestion
│
├── memory/                     # Conversation Memory
│   ├── agent.py               # Memory interface
│   └── store.py               # PostgreSQL CRUD operations
│
├── persona/                    # Customer Personas
│   └── agent.py               # Persona management + prompts
│
├── shared/                     # Shared Utilities
│   ├── types.py               # TypedDict definitions
│   ├── constants.py           # Application constants
│   └── interfaces.py          # Function signatures
│
├── requirements.txt            # Python dependencies
└── README.md


---

## 📚 API Documentation

### Interactive Documentation

When the backend is running, access the interactive API docs at:
- *Swagger UI:* http://localhost:8000/docs
- *ReDoc:* http://localhost:8000/redoc

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/auth/register | Register new user |
| POST | /api/auth/login | Authenticate user |
| GET | /api/auth/me | Get current user profile |
| GET | /api/personas | List available personas |
| GET | /api/personas/{id} | Get persona details |
| POST | /api/sessions | Create training session |
| GET | /api/sessions | List user sessions |
| GET | /api/sessions/{id} | Get session details |

### WebSocket Protocol

*Endpoint:* ws://localhost:8000/ws/{session_id}?token={jwt_token}

#### Client → Server

json
{ "type": "audio_complete", "data": { "audio_base64": "...", "format": "webm" } }

json
{ "type": "end_session" }


#### Server → Client

json
{ "type": "transcription", "data": { "text": "السلام عليكم" } }

json
{ "type": "audio_chunk", "data": { "audio_base64": "...", "sample_rate": 24000, "chunk_index": 1, "text": "وعليكم السلام", "is_final": false } }

json
{ "type": "audio_chunk", "data": { "is_final": true, "total_chunks": 2 } }

json
{ "type": "response", "data": { "text": "Full response text" } }

json
{ "type": "emotion", "data": { "emotion": "interested", "mood_score": 65, "risk_level": "low", "tip": "..." } }


---

## ⚙️ Configuration

### Environment Variables

Create a .env file in the project root:

env
# Database (PostgreSQL required)
DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/vcai

# Security
SECRET_KEY=your-secure-secret-key-here
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# Feature Flags
USE_MOCKS=false
DEBUG=false


> ⚠️ *PostgreSQL is required.* Install [PostgreSQL](https://www.postgresql.org/download/) and create a database named vcai before running the app. The SQL schema is in scripts/setup_db.sql.

### TTS Fine-tuned Checkpoint

The TTS uses an Egyptian Arabic fine-tuned checkpoint. Configure the path in tts/agent.py:

python
EGYPTIAN_CHECKPOINT = r"C:\path\to\checkpoint-2000\model.safetensors"
# Set to None to use base Chatterbox model


---

## 🔧 Troubleshooting

<details>
<summary><b>🔴 Chatterbox install fails (pkuseg error)</b></summary>

This is a known issue with chatterbox-tts >= 0.1.3. Fix:
bash
pip install --upgrade pip setuptools wheel cython
pip install numpy
pip install --no-build-isolation pkuseg
pip install chatterbox-tts

</details>

<details>
<summary><b>🔴 CUDA not detected after install</b></summary>

Chatterbox may install CPU-only PyTorch. Reinstall:
bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

</details>

<details>
<summary><b>🔴 bcrypt / chromadb dependency conflict</b></summary>

Install bcrypt before chromadb:
bash
pip install bcrypt==4.0.1
pip install chromadb

</details>

<details>
<summary><b>🔴 bitsandbytes CUDA errors on Windows</b></summary>

Use the latest version which has native Windows support:
bash
pip install bitsandbytes>=0.45.0

</details>

<details>
<summary><b>🔴 Microphone not transcribing accurately</b></summary>

Increase microphone volume and enable boost in Windows sound settings. The system normalizes quiet audio automatically, but very low input levels may still cause issues.
</details>

---

## 📈 Performance Metrics

### Benchmarks (NVIDIA RTX — tested)

| Metric | First Turn | Subsequent Turns |
|--------|-----------|-----------------|
| STT Latency | ~1.1s (cold start) | 0.25-0.45s |
| Emotion Analysis | ~0.7s (model load) | 0.05-0.06s |
| LLM Response | 2-3s | 1-3s |
| TTS per Chunk | 1.5-3s | 1.2-2.5s |
| Memory Load/Save | 50ms / 5ms | 3-5ms / 5ms |
| *First Audio Heard* | *~4s* | *~2.5s* |
| *Total Turn Time* | ~6.5s | 2.5-5s |

---

## 🛣️ Roadmap

- [x] Core conversation pipeline (LangGraph orchestration)
- [x] Real-time speech recognition (Faster-Whisper GPU)
- [x] Emotion detection (custom-trained emotion2vec + AraBERT fusion)
- [x] LLM integration (Qwen 2.5-7B, 4-bit quantized)
- [x] Egyptian Arabic TTS (Chatterbox fine-tuned)
- [x] Streaming audio pipeline (sentence-level LLM→TTS)
- [x] Conversation memory with checkpoints
- [x] WebSocket real-time communication
- [ ] RAG with property database (ChromaDB)
- [ ] Performance analytics dashboard
- [ ] Post-session evaluation and scoring
- [ ] Mobile application

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) — Speech recognition
- [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) — Text-to-speech
- [Qwen 2.5](https://github.com/QwenLM/Qwen2.5) — Language model
- [LangGraph](https://github.com/langchain-ai/langgraph) — Pipeline orchestration
- [FastAPI](https://fastapi.tiangolo.com/) — Backend framework
- [React](https://react.dev/) — Frontend framework

---

<div align="center">

*Built with ❤️ for sales excellence*

</div>