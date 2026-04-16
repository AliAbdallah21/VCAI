# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VCAI is an AI-powered sales training platform with real-time voice conversations in Egyptian Arabic. It simulates realistic customer interactions to help real estate salespeople practice handling various customer personas.

## Commands

### Backend
```bash
# Start backend server
python -m backend.main
# Or with hot-reload:
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Run all tests
pytest backend/tests/

# Run a single test file
pytest backend/tests/api/test_auth.py

# Run a specific test
pytest backend/tests/api/test_auth.py::test_login
```

### Frontend
```bash
cd frontend
npm run dev      # Start dev server (http://localhost:5173)
npm run build    # Production build
```

### Environment Setup
```bash
conda create -n vcai python=3.11 -y && conda activate vcai
pip install chatterbox-tts
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
cd frontend && npm install
```

## Architecture

### System Flow
```
Browser (React) ↔ WebSocket/REST ↔ FastAPI (backend/main.py)
                                          ↓
                               LangGraph pipeline (orchestration/)
                                          ↓
        ┌──────────┬──────────┬──────────┬──────────┐
       STT       Emotion      LLM        TTS      Memory
    (Whisper) (emotion2vec) (Qwen/OR) (Chatterbox) (PostgreSQL)

Post-session:
    Evaluation pipeline (evaluation/) ← RAG (FAISS) for fact-checking
```

### RAG Architecture — IMPORTANT

**RAG is used ONLY in the post-session evaluation pipeline. It is NOT part of the real-time conversation.**

| Where | Uses RAG? | Why |
|-------|-----------|-----|
| Conversation pipeline (orchestration/) | ❌ NO | The virtual customer responds based on its persona only. It does not need to "look up" property data. |
| Evaluation pipeline (evaluation/) | ✅ YES | After the session ends, the evaluation analyzer retrieves knowledge-base documents to fact-check what the salesperson said (wrong prices, features, policies). |

**Do NOT re-add RAG to `conversation_graph.py`.** This is a deliberate architectural decision.

Knowledge base documents live in `data/documents/`:
- `sample_properties.txt` — property names, prices, sizes, payment plans, delivery dates
- `company_policies.txt` — reservation, cancellation, installment, transfer policies

### LangGraph Pipeline (orchestration/)

The conversation runs through these nodes in sequence (`orchestration/graphs/conversation_graph.py`):
1. `memory_load` → Load conversation history from PostgreSQL
2. `stt` → Faster-Whisper transcribes audio (~300-450ms)
3. `emotion` → emotion2vec + AraBERT fusion (~55ms)
4. `llm` → Qwen 2.5-7B or OpenRouter API, streamed sentence-by-sentence (~1-3s)
5. `tts` → Chatterbox synthesizes each sentence as it arrives (~1.5-3s/chunk)
6. `memory_save` → PostgreSQL checkpoint every 5 turns

**Streaming:** LLM completes one sentence → TTS processes it → audio chunk sent to browser immediately. This achieves ~2.5s first audio vs ~5.5s without streaming.

### Evaluation Pipeline (evaluation/)

Triggered after session ends. Two-pass LLM pipeline:
1. `compute_quick_stats` → duration, turn count, final emotion (no LLM)
2. `analyzer_node` → deep analysis: skills, fact-checking against RAG knowledge base, checkpoints
3. `synthesizer_node` → final scored report (training mode: coaching tone; testing mode: pass/fail)

Fact-checking in the analyzer: `evaluation/manager.py` calls `rag.agent.retrieve_context()` for each fact topic (prices, sizes, payment plans, locations) **after** the session ends. Results are passed to the analyzer as `rag_context` so it can flag salesperson errors like quoting a wrong price or incorrect payment terms.

### Key Modules

| Module | Location | Responsibility |
|--------|----------|---------------|
| FastAPI app | `backend/main.py` | HTTP/WebSocket server, model preloading |
| Orchestration | `orchestration/agent.py` | Session management, `OrchestrationAgent` |
| Pipeline nodes | `orchestration/nodes/` | STT, emotion, LLM, TTS, memory nodes (no RAG in conversation) |
| Shared state | `orchestration/state.py` | `ConversationState` TypedDict |
| API routes | `backend/routers/` | Auth, sessions, personas, evaluation |
| DB models | `backend/models/` | SQLAlchemy ORM (users, sessions, personas, evaluations) |
| Evaluation | `evaluation/` | Post-session scoring pipeline |
| Shared types | `shared/types.py` | `Persona`, `Message`, `EmotionResult`, `SessionMemory`, `RAGContext` |
| Constants | `shared/constants.py` | `CHECKPOINT_INTERVAL=5`, `RAG_TOP_K=3`, `STT_SAMPLE_RATE=16000` |

### WebSocket Protocol

```
ws://localhost:8000/ws/{session_id}?token={jwt_token}

Client → Server: { "type": "audio_complete", "data": { "audio_base64": "...", "format": "webm" } }
Server → Client: Audio chunks, transcriptions, emotion analysis, AI responses
```

### Configuration & Feature Flags

Key environment variables (`.env`):
- `DATABASE_URL` — PostgreSQL connection string
- `JWT_SECRET` — Auth token secret
- `USE_OPENROUTER` — Use OpenRouter API instead of local Qwen model
- `OPENROUTER_API_KEY` / `OPENROUTER_MODEL` — OpenRouter credentials
- `USE_MOCKS` — Use mock implementations for testing
- `PRELOAD_MODELS` — Load ML models at startup

Orchestration configs (`orchestration/config.py`):
- `get_development_config()` — mocks enabled, verbose
- `get_production_config()` — real agents
- `get_testing_config()` — mocks, streaming disabled

### Interface Contracts

All ML modules implement strict interfaces defined in `shared/interfaces.py`. **Do not modify interface signatures** — other team members depend on them. Match input/output types exactly when implementing or modifying pipeline nodes.

### Testing Setup

Tests use SQLite in-memory DB (not PostgreSQL). Fixtures in `backend/tests/conftest.py` provide `db`, `test_user`, `test_persona`, `test_session`, `test_messages`, `test_emotion_logs`.

### Frontend Structure

Pages in `frontend/src/pages/`: Login, Register, Dashboard, SessionSetup, TrainingSession (main audio UI), EvaluationReport.

Audio flow: Web Audio API → WebSocket → gapless AudioContext playback using queued chunks from server.
