# VCAI Setup Instructions

## 📋 Prerequisites

Before starting, make sure you have:

| Requirement | Version | Check Command |
|-------------|---------|---------------|
| Python | 3.10 - 3.12 | `python --version` |
| NVIDIA GPU | Any RTX/GTX | `nvidia-smi` |
| CUDA | 11.8 or 12.x | `nvidia-smi` (top right) |
| Git | Any | `git --version` |

> ⚠️ **Python 3.13+ is NOT supported** (PyTorch doesn't support it yet)

---

## 🚀 Quick Setup (5 minutes)

### Option A: Using Anaconda (Recommended)

```bash
# 1. Create conda environment
conda create -n vcai python=3.12 -y
conda activate vcai

# 2. Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Clone and setup
cd C:\VCAI
pip install -r requirements.txt

# 4. Verify
python scripts/verify_setup.py
```

### Option B: Using venv (Python 3.10-3.12 required)

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 2. Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install other requirements
pip install -r requirements.txt

# 4. Verify
python scripts/verify_setup.py
```

---

## 🔧 Detailed Setup

### Step 1: Check Your CUDA Version

```bash
nvidia-smi
```

Look at the top right for "CUDA Version: XX.X"

### Step 2: Install PyTorch with Matching CUDA

| Your CUDA Version | Install Command |
|-------------------|-----------------|
| CUDA 11.8 | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` |
| CUDA 12.1 | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121` |
| CUDA 12.4+ | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124` |
| No GPU | `pip install torch torchvision torchaudio` (CPU only) |

### Step 3: Verify PyTorch CUDA

```python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

Should output: `CUDA available: True`

### Step 4: Install Requirements

```bash
pip install -r requirements.txt
```

### Step 5: Run Verification Script

```bash
python scripts/verify_setup.py
```

---

## 🧪 Test Individual Components

### Test STT (Speech-to-Text)
```bash
python -m stt.realtime_stt
```

### Test Mocks
```bash
python scripts/test_mocks.py
```

### Test Orchestration
```bash
python scripts/test_orchestration.py
```

---

## ❌ Common Problems

### Problem: "CUDA not available"

**Cause:** PyTorch installed without CUDA support

**Fix:**
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Problem: "No module named 'faster_whisper'"

**Fix:**
```bash
pip install faster-whisper
```

### Problem: "Python version not supported"

**Cause:** Using Python 3.13 or 3.14

**Fix:** Use Python 3.10, 3.11, or 3.12
```bash
conda create -n vcai python=3.12 -y
conda activate vcai
```

### Problem: Model download slow

**Cause:** First run downloads ~1.5GB model

**Fix:** Just wait, or use a faster network. Model is cached after first download.

---

## 📁 Project Structure

```
C:\VCAI\
├── backend/           # FastAPI server
├── frontend/          # React UI
├── orchestration/     # LangGraph pipeline ✅
│   ├── agent.py       # Main agent
│   ├── nodes/         # Pipeline nodes
│   ├── graphs/        # LangGraph workflows
│   └── mocks/         # Mock functions ✅
├── stt/               # Speech-to-Text ✅
├── tts/               # Text-to-Speech (Person B)
├── emotion/           # Emotion Detection (Person C)
├── llm/               # LLM Agent (Person D)
├── rag/               # RAG Agent (Person D)
├── memory/            # Memory Agent (Person D)
├── persona/           # Persona Agent (Person B)
├── shared/            # Shared types/interfaces ✅
├── scripts/           # Utility scripts
└── docs/              # Documentation
```

---

## 👥 Team Components

| Person | Components | Folder |
|--------|------------|--------|
| A (Ali) | STT, Orchestration, Backend, UI | `stt/`, `orchestration/`, `backend/`, `frontend/` |
| B | TTS, Persona | `tts/`, `persona/` |
| C | Emotion Detection | `emotion/` |
| D | LLM, RAG, Memory | `llm/`, `rag/`, `memory/` |

---

## 🔗 Useful Links

- [PyTorch Installation](https://pytorch.org/get-started/locally/)
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [FastAPI](https://fastapi.tiangolo.com/)