<div align="center">

<img src="https://img.shields.io/badge/VCAI-Virtual%20Customer%20AI-6B21A8?style=for-the-badge&labelColor=1E1B4B" />
<img src="https://img.shields.io/badge/version-1.0.0-10B981?style=for-the-badge" />
<img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black" />
<img src="https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
<img src="https://img.shields.io/badge/LangGraph-Orchestration-7C3AED?style=for-the-badge" />

<br/><br/>

# VCAI — Virtual Customer AI

### *First Egyptian Arabic Offline Virtual Customer AI for Real Estate Sales Training*

> Practice real estate sales conversations with an AI customer that speaks Egyptian Arabic, adapts emotionally, remembers across sessions, and scores your performance — all running offline on your GPU.

[Overview](#overview) · [Features](#features) · [Architecture](#architecture) · [Installation](#installation) · [API](#api) · [Evaluation System](#evaluation-system) · [Results](#results)

---

</div>

## Overview

VCAI is an AI-powered sales training platform for Egyptian real estate companies. It simulates a virtual customer who speaks Egyptian Arabic dialect, reacts emotionally to the salesperson's approach, maintains memory across the full conversation, and provides automated performance evaluation after every session.

**What makes it unique:**

- **First Egyptian Arabic virtual customer AI** — no prior system targets Egyptian dialect for sales training
- **Fully offline** — the entire pipeline runs on a local GPU; no customer data, scripts, or recordings leave the organization's network
- **Deterministic evaluation** — factual errors are caught via structured RAG, not LLM guessing
- **Cross-session memory** — the virtual customer remembers previous calls via LLM-summarized checkpoints

**Built as a Graduation Project at Misr International University, Computer Science Department, 2026.**

| Student | ID |
|---|---|
| Ali Abdallah | 2022/05974 |
| Ismail Hesham | 2022/00106 |
| Abubakr Hegazy | 2022/02645 |
| Mena Khaled | 2022/03469 |

**Supervised by:** Dr. Ahmed Mansour · T.A. Karim Mohamed

---

## Features

### Core Pipeline

| Component | Technology | Status | Metric |
|---|---|---|---|
| 🎤 Speech-to-Text | Faster-Whisper large-v3-turbo (CUDA) | ✅ Working | WER 15% (Egyptian Arabic) |
| 😤 Emotion Detection | emotion2vec + AraBERT fusion (60/40) | ✅ Working | 96% accuracy |
| 🤖 LLM — Virtual Customer | Qwen 2.5-7B 4-bit NF4 via OpenRouter | ✅ Working | ~1.5s first sentence |
| 🔊 Text-to-Speech | Chatterbox Multilingual (Egyptian fine-tuned) | ✅ Working | ~1.5s per chunk |
| 🧠 Conversation Memory | PostgreSQL + LLM checkpoint summaries | ✅ Working | Every 5 turns |
| 🔍 RAG Fact-Checking | FAISS + Structured JSON (evaluation only) | ✅ Working | 152 chunks indexed |
| 📊 Evaluation Pipeline | LangGraph 2-pass LLM scoring | ✅ Working | 8 skills assessed |
| 🌐 Real-time Streaming | WebSocket sentence-level LLM→TTS | ✅ Working | ~3.5s first audio |

### Customer Personas

| Persona | Personality | Arabic Name | Difficulty |
|---|---|---|---|
| 💰 Price-Focused | Negotiates aggressively, questions every cost | العميل المهتم بالسعر | Medium |
| 😤 Difficult | Skeptical, raises objections, hard to close | العميل الصعب | Hard |
| 😊 Friendly | Open and cooperative, easy to build rapport | العميل الودود | Easy |
| ⏰ Rushed | Limited time, wants quick concise answers | العميل المتسرع | Medium |
| 🔬 Detail-Oriented | Asks technical questions, needs full specs | العميل المهتم بالتفاصيل | Hard |

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────┐
│                   React Frontend                     │
│  Login · Dashboard · Training Session · Report       │
└────────────────────┬────────────────────────────────┘
                     │ WebSocket (audio chunks)
                     │ REST API (auth / sessions / personas)
                     ▼
┌─────────────────────────────────────────────────────┐
│              FastAPI Backend (port 8000)             │
│  Auth · Sessions · Personas · WebSocket Handler      │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│           LangGraph Orchestration Pipeline           │
│                                                      │
│  memory_load → STT → emotion → LLM → TTS → memory_save
│                                                      │
│  (streaming path: llm_node_streaming → tts_chunk)    │
└────────────────────┬────────────────────────────────┘
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
      PostgreSQL   FAISS     OpenRouter
      (memory)    (eval RAG)  (LLM API)
```

### Conversation Turn Flow

Every turn follows this exact pipeline:

```
1. memory_load_node    Load checkpoints + last 10 messages from PostgreSQL
2. stt_node            Faster-Whisper transcribes audio → Arabic text (WER 15%)
3. emotion_node        emotion2vec (voice 60%) + AraBERT (text 40%) → fused emotion
4. llm_node            Build persona prompt + memory context → stream sentences
5. tts_node            Synthesize each sentence → send audio chunk to browser
6. memory_save_node    Save turn to DB; every 5 turns → LLM summarizes checkpoint
```

**First audio reaches the browser in ~3.5 seconds** (vs 9–11s without streaming):

```
STT ~0.5s + Emotion ~0.06s + LLM first sentence ~1.5s + TTS first chunk ~1.5s = ~3.5s
```

### RAG Architecture — Evaluation Only

RAG is **not** used during conversation. The virtual customer responds based on persona only.

RAG is used **post-session** to fact-check the salesperson's claims:

```
Session ends → fact_check_transcript(transcript)
    ↓
Phase 1: Extract claims from all salesperson turns (regex + keywords)
         Price claims: مليون، ألف، جنيه + numbers
         Size claims: متر، م٢ + numbers
         Payment claims: مقدم، قسط، سنة، % + numbers
         Location: matched against properties.json keywords
    ↓
Phase 2: Global hint pre-scan — finds property name across ALL turns
         (fixes cross-turn bug: property named in turn 2, price in turn 5)
    ↓
Phase 3: FAISS semantic search identifies which property is being discussed
         Confidence threshold: > 0.6 cosine similarity
    ↓
Phase 4: Exact field comparison against properties.json
         claimed 1.5M vs actual price_min=2M → ❌ CRITICAL error
    ↓
FactCheckResult → passed to evaluation LLM as pre-computed facts
```

**Why hybrid FAISS + structured JSON instead of pure vector search?**
Semantic search finds which property is being discussed. Structured JSON does the actual comparison. `1,500,000 ≠ 2,000,000` is a deterministic check — not an LLM opinion.

### Memory System

```
Turn 1-5:   All messages stored verbatim in PostgreSQL
Turn 5:     LLM summarizes turns 1–5 → checkpoint saved
Turn 6-10:  New messages + previous checkpoint in context
Turn 10:    LLM summarizes turns 6–10 → second checkpoint saved
...
```

At any point the LLM receives: `[checkpoint summaries] + [last 10 messages]` — full context without token explosion.

---

## Project Structure

```
VCAI/
├── backend/                    # FastAPI backend
│   ├── main.py                 # Startup + model preloading + health checks
│   ├── config.py               # Env-based configuration (no hardcoded secrets)
│   ├── health.py               # 6-module health check system
│   ├── models/                 # SQLAlchemy ORM models
│   ├── routers/                # Auth, sessions, personas, evaluation, websocket
│   ├── schemas/                # Pydantic validation schemas
│   └── services/               # Business logic + evaluation service
│
├── frontend/                   # React 19 + Vite frontend
│   └── src/
│       ├── pages/              # Login, Dashboard, SessionSetup, TrainingSession, EvaluationReport
│       ├── components/         # Shared UI components
│       ├── context/            # Auth context
│       └── services/           # API + WebSocket client
│
├── orchestration/              # LangGraph pipeline
│   ├── agent.py                # OrchestrationAgent (session management)
│   ├── state.py                # ConversationState TypedDict
│   ├── graphs/
│   │   └── conversation_graph.py  # Main pipeline + streaming graph
│   └── nodes/
│       ├── stt_node.py
│       ├── emotion_node.py
│       ├── llm_node.py         # + llm_node_streaming() generator
│       ├── tts_node.py         # + tts_chunk() for streaming
│       └── memory_node.py      # load + save + checkpoint
│
├── stt/
│   └── realtime_stt.py         # Faster-Whisper large-v3-turbo, CUDA
│
├── tts/
│   └── agent.py                # Chatterbox + Egyptian checkpoint + 3-layer fallback
│
├── emotion/
│   ├── voice_emotion.py        # emotion2vec classifier
│   ├── text_emotion.py         # AraBERT sentiment
│   └── fusion.py               # 60/40 weighted fusion
│
├── llm/
│   ├── agent.py                # OpenRouter SSE streaming + Qwen local fallback
│   └── prompts.py              # Persona prompt templates
│
├── rag/
│   ├── agent.py                # retrieve_context() + fact_check_transcript()
│   ├── claim_extractor.py      # Arabic regex claim extraction + global hint
│   ├── fact_checker.py         # FAISS → structured JSON exact comparison
│   ├── structured_store.py     # properties.json + policies.json singleton cache
│   ├── vector_store.py         # FAISS IndexFlatIP operations
│   ├── document_loader.py      # .txt + .json document ingestion
│   └── index_build.py          # Build/rebuild FAISS index (152 chunks)
│
├── memory/
│   ├── agent.py                # Public memory interface
│   └── store.py                # PostgreSQL CRUD + checkpoint management
│
├── persona/
│   └── agent.py                # 5 personas with full Egyptian Arabic prompts
│
├── evaluation/
│   ├── manager.py              # EvaluationManager — orchestrates full pipeline
│   ├── state.py                # EvaluationState TypedDict
│   ├── graphs/
│   │   └── evaluation_graph.py # compute_quick_stats → analyzer → synthesizer
│   ├── pipeline/
│   │   ├── analyzer.py         # 8-skill LLM scoring node
│   │   └── synthesizer.py      # Final report assembly node
│   ├── prompts.py              # Analyzer/synthesizer prompt templates
│   └── results/                # Auto-saved JSON reports (result1.json, result2.json...)
│
├── data/
│   ├── documents/
│   │   ├── properties.json     # 11 Egyptian properties (structured KB)
│   │   ├── policies.json       # 10 company policy sets
│   │   ├── sample_properties.txt
│   │   └── company_policies.txt
│   └── faiss_index/            # FAISS IndexFlatIP (152 chunks, multilingual embeddings)
│
├── shared/
│   ├── types.py                # All TypedDict definitions
│   ├── constants.py            # Application-wide constants
│   ├── interfaces.py           # Function interface specifications
│   └── exceptions.py           # 40+ custom exception classes
│
├── scripts/
│   └── seed_personas.py        # Seeds 5 personas to PostgreSQL
│
├── reports/                    # trigger_eval.py output directory
├── trigger_eval.py             # CLI evaluation trigger (auto-detects latest session)
├── CLAUDE.md                   # Architecture decisions log
└── requirements.txt
```

---

## Installation

### Hardware Requirements

| Component | Minimum | Tested On |
|---|---|---|
| GPU | NVIDIA GTX 1060 (6GB VRAM) | NVIDIA RTX 5060 Ti (16GB VRAM) |
| RAM | 16 GB | 32 GB |
| Storage | 20 GB SSD | NVMe SSD |
| CUDA | 12.1+ | 13.1 |
| OS | Windows 10/11 or Ubuntu 20.04+ | Windows 11 |

> ⚠️ GPU is required. All four models (STT, LLM, TTS, Emotion) load simultaneously and require ~10–14GB VRAM.

### Prerequisites

Install before proceeding:
- [Anaconda or Miniconda](https://www.anaconda.com/download)
- [PostgreSQL](https://www.postgresql.org/download/) — create a database named `vcai`
- [Node.js 20+](https://nodejs.org/)
- [FFmpeg](https://ffmpeg.org/download.html)
- [CUDA Toolkit 12.1+](https://developer.nvidia.com/cuda-downloads)

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-org/VCAI.git
cd VCAI

# 2. Create conda environment (Python 3.11 required for Chatterbox)
conda create -n vcai python=3.11 -y
conda activate vcai

# 3. Install Chatterbox TTS first (has specific dependency order)
pip install chatterbox-tts

# 4. Reinstall PyTorch with CUDA support (Chatterbox may install CPU-only)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Install all remaining dependencies
pip install -r requirements.txt

# 6. Install frontend dependencies
cd frontend && npm install && cd ..

# 7. Configure environment
cp .env.example .env
# Edit .env with your PostgreSQL credentials and API keys

# 8. Initialize database and seed personas
python scripts/setup_db.py
python scripts/seed_personas.py

# 9. Build the FAISS knowledge base index
python rag/index_build.py

# 10. Start the backend
python -m backend.main

# 11. Start the frontend (new terminal)
cd frontend && npm run dev
```

### Environment Variables

Create a `.env` file in the project root:

```env
# Database
DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/vcai

# Security
SECRET_KEY=your-secure-secret-key-change-this
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# LLM — choose one
USE_OPENROUTER=true
OPENROUTER_API_KEY=sk-or-v1-your-key-here
OPENROUTER_MODEL=anthropic/claude-3.7-sonnet

# TTS Egyptian checkpoint path
EGYPTIAN_TTS_CHECKPOINT=C:\path\to\model.safetensors

# Feature flags
USE_MOCKS=false
DEBUG=false
```

### Verify Installation

```bash
# Check health endpoint after startup
curl http://localhost:8000/health

# Expected output:
# {
#   "overall": "healthy",
#   "stt": {"status": "ok", "message": "Working"},
#   "tts": {"status": "ok", "message": "Egyptian dialect"},
#   "emotion": {"status": "ok", "message": "Working (detected: interested)"},
#   "rag": {"status": "ok", "message": "152 documents indexed"},
#   "memory": {"status": "ok", "message": "PostgreSQL connected"},
#   "llm": {"status": "ok", "message": "OpenRouter connected"}
# }
```

Open your browser at **http://localhost:5173**

### Troubleshooting

<details>
<summary><b>Chatterbox install fails (pkuseg error)</b></summary>

```bash
pip install --upgrade pip setuptools wheel cython
pip install numpy
pip install --no-build-isolation pkuseg
pip install chatterbox-tts
```
</details>

<details>
<summary><b>CUDA not detected after install</b></summary>

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
</details>

<details>
<summary><b>bitsandbytes CUDA errors on Windows</b></summary>

```bash
pip install bitsandbytes>=0.45.0
```
</details>

<details>
<summary><b>uvicorn restarts when evaluation saves JSON files</b></summary>

This is already handled. The uvicorn config excludes `evaluation/results/`, `reports/`, and `data/` from the file watcher. If it persists, check that `reload_excludes` is set in `backend/main.py`.
</details>

---

## API

### Interactive Docs

With the backend running:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **Health:** http://localhost:8000/health

### REST Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/auth/register` | Register new user |
| POST | `/api/auth/login` | Authenticate user (returns JWT) |
| GET | `/api/auth/me` | Get current user profile |
| GET | `/api/personas` | List all 5 personas |
| POST | `/api/sessions` | Create training session |
| GET | `/api/sessions` | List user sessions |
| GET | `/api/sessions/{id}` | Get session details |
| POST | `/api/sessions/{id}/evaluate` | Trigger evaluation (background task) |
| GET | `/api/sessions/{id}/eval-status` | Poll evaluation progress |
| GET | `/api/sessions/{id}/report` | Get completed evaluation report |
| GET | `/api/sessions/{id}/quick-stats` | Get session quick stats |

### WebSocket Protocol

**Endpoint:** `ws://localhost:8000/ws/{session_id}?token={jwt_token}`

**Client → Server:**
```json
{ "type": "audio_complete", "data": { "audio_base64": "...", "format": "webm" } }
{ "type": "end_session" }
```

**Server → Client (streaming):**
```json
{ "type": "transcription", "data": { "text": "السلام عليكم" } }
{ "type": "emotion", "data": { "emotion": "interested", "risk_level": "low", "trend": "stable" } }
{ "type": "audio_chunk", "data": { "audio_base64": "...", "chunk_index": 1, "text": "أهلاً يا أستاذ...", "is_final": false } }
{ "type": "audio_chunk", "data": { "is_final": true, "total_chunks": 2 } }
```

---

## Evaluation System

After every session, VCAI automatically evaluates the salesperson's performance using a two-pass LLM pipeline.

### Trigger

```bash
# Trigger via CLI (auto-detects latest session)
python trigger_eval.py

# Or for a specific session
python trigger_eval.py <session_id>

# Report saved automatically to:
# reports/eval_{session_id[:8]}_{YYYY-MM-DD_HH-MM}.json
```

### Evaluation Pipeline

```
POST /api/sessions/{id}/evaluate
    ↓
gather_evaluation_inputs()
  ├── Load transcript from PostgreSQL
  ├── Load emotion log (per-turn)
  ├── Load persona (name + difficulty from DB)
  └── Run fact_check_transcript() → structured errors
    ↓
LangGraph Evaluation Graph:
  compute_quick_stats → analyzer_node → synthesizer_node
    ↓
analyzer_node receives:
  ├── Full transcript
  ├── Emotion timeline
  └── PRE-COMPUTED FACT-CHECK RESULTS (deterministic, not LLM-guessed)
    ↓
Scores 8 skills:
  Rapport Building · Active Listening · Needs Discovery
  Product Knowledge · Objection Handling · Emotional Intelligence
  Closing Skills · Communication Clarity
    ↓
synthesizer_node builds final report:
  ├── Overall score (0–100, pass threshold: 75)
  ├── Per-skill scores with strengths + areas to improve
  ├── Per-turn feedback with suggested alternatives in Arabic
  ├── Checkpoint achievements (6 milestones tracked)
  └── Recommended practice exercises
    ↓
Saved to DB + evaluation/results/result_N.json
```

### Sample Report Structure

```json
{
  "overall_score": 58,
  "pass_threshold": 75,
  "status": "failed",
  "scores": {
    "product_knowledge": { "score": 65, "weight": 0.14 },
    "objection_handling": { "score": 10, "weight": 0.14 },
    "rapport_building": { "score": 40, "weight": 0.12 }
  },
  "checkpoints": [
    { "name": "Needs Identified", "achieved": true },
    { "name": "Rapport Established", "achieved": false }
  ],
  "turn_feedback": [
    {
      "turn_number": 2,
      "assessment": "needs_improvement",
      "what_to_improve": "Misspelled company name",
      "suggested_alternative": "السلام عليكم! معك علي من شركة حسن علام للتسويق العقاري..."
    }
  ]
}
```

---

## Knowledge Base

### Properties (data/documents/properties.json)

11 Egyptian real estate properties covering:
- **New Cairo:** مدينتي, التجمع الخامس, زد إيست
- **6th October:** بالم هيلز
- **Sheikh Zayed:** وستاون, الكارما
- **New Administrative Capital**
- **North Coast** (summer properties)
- **Maadi, Heliopolis** (established areas)

Each property has structured fields: `price_min`, `price_max`, `size_min_sqm`, `down_payment_percent`, `installment_years`, `delivery_year`, `features`, `keywords_ar`.

### Policies (data/documents/policies.json)

10 policy entries covering: reservation deposit, cancellation tiers, installment structure, cash discounts (5–15%), maintenance fees, unit transfer, resale restrictions, bank mortgage options.

### Rebuilding the Index

```bash
python rag/index_build.py
# Expected: [RAG] Indexed 152 chunks successfully
```

---

## Results

### Performance Benchmarks (NVIDIA RTX 5060 Ti, 16GB VRAM)

| Metric | Value |
|---|---|
| STT Word Error Rate (Egyptian Arabic) | **15%** |
| Emotion Detection Accuracy | **96%** |
| First Audio Latency (streaming) | **~3.5 seconds** |
| Non-streaming baseline latency | ~9–11 seconds |
| Streaming improvement | **~65% faster** |
| FAISS index size | 152 chunks |
| Evaluation turnaround | ~30–60 seconds |

### Turn Latency Breakdown

| Stage | Latency |
|---|---|
| STT (Faster-Whisper GPU) | ~0.5s |
| Emotion detection (dual-modal) | ~0.06s |
| LLM first sentence (OpenRouter SSE) | ~1.5s |
| TTS first chunk (Chatterbox) | ~1.5s |
| **Total first audio** | **~3.5s** |

### Test Coverage

44 test cases defined across:
- STT accuracy on Egyptian dialect audio samples
- Emotion detection on 8 emotion classes
- TTS audio quality and fallback behavior
- LLM persona consistency across 10+ turns
- Memory checkpoint creation and retrieval
- RAG claim extraction on 15 Arabic sentence patterns
- Evaluation scoring on sessions with known factual errors
- WebSocket connection handling and streaming

---

## Roadmap

- [x] Core conversation pipeline (LangGraph orchestration)
- [x] Egyptian Arabic STT (Faster-Whisper large-v3-turbo, WER 15%)
- [x] Dual-modal emotion detection (emotion2vec + AraBERT, 96% accuracy)
- [x] LLM virtual customer (Qwen 2.5-7B 4-bit NF4 + OpenRouter)
- [x] Egyptian Arabic TTS (Chatterbox fine-tuned on audiobooks)
- [x] Sentence-level streaming pipeline (~3.5s first audio)
- [x] Cross-session memory with LLM checkpoint summaries
- [x] 5 customer personas with full Egyptian Arabic prompts
- [x] Structured knowledge base (11 properties + 10 policies in JSON)
- [x] Hybrid FAISS + structured RAG for deterministic fact-checking
- [x] Automated post-session evaluation (8 skills + turn-level feedback)
- [x] Health check system (6 modules, `/health` endpoint)
- [ ] GraphRAG knowledge graph for multi-hop property queries
- [ ] Mobile application (edge quantization via GGUF)
- [ ] Multi-dialect support (Saudi, Gulf, Levantine)
- [ ] RL from evaluation feedback — adaptive persona difficulty
- [ ] End-user acceptance testing with real estate sales teams
- [ ] Company-level analytics dashboard

---

## Acknowledgments

- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2-based Whisper for fast Arabic STT
- [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) — Multilingual TTS with Egyptian fine-tuning support
- [Qwen 2.5](https://github.com/QwenLM/Qwen2.5) — Best-in-class Arabic language model
- [LangGraph](https://github.com/langchain-ai/langgraph) — Stateful pipeline orchestration with checkpointing
- [FAISS](https://github.com/facebookresearch/faiss) — Fast similarity search for RAG retrieval
- [FastAPI](https://fastapi.tiangolo.com/) — High-performance async Python backend
- [React](https://react.dev/) — Frontend framework

---

<div align="center">

**VCAI — AI Virtual Customer**
Misr International University · Computer Science Department · GP 2026

*Built for Egyptian real estate sales excellence*

</div>
