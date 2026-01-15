# Contributing to VCAI

<div align="center">

![Contributors Guide](https://img.shields.io/badge/VCAI-Contributors%20Guide-green?style=for-the-badge)

**Read this ENTIRE document before writing any code!**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Team Assignments](#-team-assignments)
- [Setup Instructions](#-setup-instructions)
- [Person B: TTS + Persona](#-person-b-tts--persona)
- [Person C: Emotion Detection](#-person-c-emotion-detection)
- [Person D: RAG + Memory + LLM](#-person-d-rag--memory--llm)
- [Testing Your Implementation](#-testing-your-implementation)
- [Submission Workflow](#-submission-workflow)
- [FAQ & Troubleshooting](#-faq--troubleshooting)

---

## 🎯 Overview

### What is VCAI?

VCAI is a sales training platform where salespeople practice with AI customers. The system flow is:

```
Salesperson Speaks → STT → Emotion Detection → LLM Response → TTS → Customer Responds
```

### Your Role

Each team member implements specific functions. Your code will be integrated into the main system. **You don't need to run the full system** - just implement your functions and pass the tests.

### Golden Rules

1. ✅ **Match the interface EXACTLY** - function names, parameters, return types
2. ✅ **Run the test script** before pushing
3. ✅ **All tests must pass** before pushing
4. ❌ **Don't modify** other people's code
5. ❌ **Don't change** the interface signatures

---

## 👥 Team Assignments

| Person | Components | Files to Create |
|--------|------------|-----------------|
| **Person A (Ali)** | STT, Backend, Integration | ✅ Done |
| **Person B** | TTS, Persona | `tts/tts_agent.py` |
| **Person C** | Emotion Detection | `emotion/emotion_agent.py` |
| **Person D** | RAG, Memory, LLM | `rag/rag_agent.py`, `memory/memory_agent.py`, `llm/llm_agent.py` |

---

## ⚙️ Setup Instructions

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-org/VCAI.git
cd VCAI
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Base Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Install Your Component's Dependencies

See your specific section below for additional packages to install.

### Step 5: Verify Setup

```bash
# Make sure you can import numpy
python -c "import numpy as np; print('Setup OK')"
```

---

## 🔊 Person B: TTS + Persona

### Overview

You are responsible for converting Arabic text to natural Egyptian speech.

### Recommended Technology

**Chatterbox TTS with Multilingual Fine-tuning**
- Repository: https://github.com/gokhaneraslan/chatterbox-finetuning
- Great for Egyptian Arabic voice cloning

### Installation

```bash
# Install Chatterbox dependencies
pip install torch torchaudio
pip install chatterbox-tts  # or clone the repo above

# If using the fine-tuning repo:
git clone https://github.com/gokhaneraslan/chatterbox-finetuning.git
cd chatterbox-finetuning
pip install -r requirements.txt
```

### File to Create

```
tts/
└── tts_agent.py
```

### Function to Implement

```python
# tts/tts_agent.py
"""
Text-to-Speech Agent using Chatterbox TTS.
Person B Implementation.
"""

import numpy as np

# Your imports here (chatterbox, torch, etc.)


def text_to_speech(
    text: str,
    voice_id: str = "default",
    emotion: str = "neutral"
) -> np.ndarray:
    """
    Convert Arabic text to speech audio.
    
    Args:
        text: str
            - Arabic text to speak
            - Example: "الشقة دي في موقع ممتاز"
            - Max length: 500 characters
        
        voice_id: str
            - Voice to use
            - Options: "default", "egyptian_male_01", "egyptian_female_01"
            - Default: "default"
        
        emotion: str
            - Emotional tone
            - Options: "neutral", "happy", "frustrated", "interested", "hesitant"
            - Default: "neutral"
    
    Returns:
        np.ndarray:
            - Audio samples
            - dtype: float32
            - sample_rate: 22050 Hz
            - shape: (n_samples,) - 1D array
            - values: between -1.0 and 1.0
    
    Example:
        >>> audio = text_to_speech("مرحبا بيك", voice_id="egyptian_male_01")
        >>> audio.shape
        (44100,)  # ~2 seconds
        >>> audio.dtype
        dtype('float32')
    """
    
    # ═══════════════════════════════════════════════════════════════════
    # YOUR IMPLEMENTATION HERE
    # ═══════════════════════════════════════════════════════════════════
    
    # Example structure:
    # 1. Load/get your TTS model
    # 2. Select voice based on voice_id
    # 3. Adjust for emotion (pitch, speed, etc.)
    # 4. Generate audio
    # 5. Convert to float32 numpy array
    # 6. Ensure sample rate is 22050 Hz
    # 7. Return 1D array
    
    pass
```

### Implementation Tips

```python
# Example implementation structure (adapt to your model):

import numpy as np
import torch

# Global model (load once)
_model = None

def _load_model():
    global _model
    if _model is None:
        # Load your Chatterbox model here
        # _model = ChatterboxTTS.load(...)
        pass
    return _model

def text_to_speech(text: str, voice_id: str = "default", emotion: str = "neutral") -> np.ndarray:
    model = _load_model()
    
    # Map voice_id to your voice files/embeddings
    voice_map = {
        "default": "path/to/default_voice",
        "egyptian_male_01": "path/to/male_voice",
        "egyptian_female_01": "path/to/female_voice",
    }
    
    # Generate audio with your model
    # audio = model.synthesize(text, voice=voice_map[voice_id])
    
    # Convert to numpy float32
    # audio_np = audio.cpu().numpy().astype(np.float32)
    
    # Ensure 1D
    # audio_np = audio_np.flatten()
    
    # Normalize to [-1, 1]
    # if audio_np.max() > 1.0:
    #     audio_np = audio_np / np.abs(audio_np).max()
    
    # return audio_np
    
    pass
```

### Test Your Implementation

```bash
cd C:\VCAI
python scripts/teammate_tests/test_tts.py
```

### Expected Output

```
[Test 1] Checking if text_to_speech function exists...
   ✅ Function imported successfully
[Test 2] Checking function signature...
   ✅ Parameters correct
...
🎉 ALL TESTS PASSED! Your implementation is ready.
```

---

## 😤 Person C: Emotion Detection

### Overview

You detect emotions from both text (Arabic) and voice (audio waveform).

### Recommended Technologies

| Component | Technology | Purpose |
|-----------|------------|---------|
| Voice Emotion | **Wav2Vec 2.0** | Extract emotions from audio |
| Text Emotion | **CAMeL Tools** | Arabic NLP and sentiment |

### Installation

```bash
# Wav2Vec for voice emotion
pip install transformers torch torchaudio

# CAMeL Tools for Arabic text
pip install camel-tools

# Additional (if needed)
pip install scipy librosa
```

### File to Create

```
emotion/
└── emotion_agent.py
```

### Functions to Implement

```python
# emotion/emotion_agent.py
"""
Emotion Detection Agent.
Uses Wav2Vec for voice and CAMeL for text.
Person C Implementation.
"""

import numpy as np

# Your imports here


def detect_emotion(text: str, audio: np.ndarray) -> dict:
    """
    Detect emotion from text and audio.
    
    Args:
        text: str
            - Arabic transcription from STT
            - Example: "ده غالي أوي!"
        
        audio: np.ndarray
            - Raw audio from speaker
            - dtype: float32
            - sample_rate: 16000 Hz
            - shape: (n_samples,) - 1D array
    
    Returns:
        dict: EmotionResult with these EXACT keys:
            {
                "primary_emotion": str,    # "happy", "sad", "angry", "fearful", 
                                           # "surprised", "disgusted", "neutral"
                "confidence": float,       # 0.0 to 1.0
                "voice_emotion": str,      # Emotion from audio analysis
                "text_emotion": str,       # Emotion from text analysis
                "intensity": str,          # "low", "medium", "high"
                "scores": {                # All emotions with scores
                    "happy": float,        # 0.0 to 1.0
                    "sad": float,
                    "angry": float,
                    "fearful": float,
                    "surprised": float,
                    "disgusted": float,
                    "neutral": float
                }
            }
    
    Example:
        >>> emotion = detect_emotion("ده غالي أوي!", audio_data)
        >>> emotion["primary_emotion"]
        "angry"
        >>> emotion["confidence"]
        0.87
    """
    
    # ═══════════════════════════════════════════════════════════════════
    # YOUR IMPLEMENTATION HERE
    # ═══════════════════════════════════════════════════════════════════
    
    # Suggested approach:
    # 1. Analyze text with CAMeL → get text_emotion
    # 2. Analyze audio with Wav2Vec → get voice_emotion
    # 3. Combine both (weighted average or voting)
    # 4. Determine primary_emotion and confidence
    # 5. Calculate intensity based on confidence
    # 6. Return the dict with ALL required keys
    
    pass


def analyze_emotional_context(
    current_emotion: dict,
    history: list
) -> dict:
    """
    Analyze emotional context with conversation history.
    
    Args:
        current_emotion: dict
            - Output from detect_emotion()
        
        history: list[dict]
            - Previous messages, each with:
              {"id": str, "turn": int, "speaker": str, "text": str, ...}
    
    Returns:
        dict: EmotionalContext with these EXACT keys:
            {
                "current": dict,           # The current_emotion passed in
                "trend": str,              # "improving", "worsening", "stable"
                "recommendation": str,     # "be_gentle", "be_firm", "show_empathy"
                "risk_level": str          # "low", "medium", "high"
            }
    
    Example:
        >>> context = analyze_emotional_context(emotion, history)
        >>> context["trend"]
        "worsening"
        >>> context["recommendation"]
        "show_empathy"
    """
    
    # ═══════════════════════════════════════════════════════════════════
    # YOUR IMPLEMENTATION HERE
    # ═══════════════════════════════════════════════════════════════════
    
    # Suggested approach:
    # 1. Look at emotion history in messages
    # 2. Determine trend (is customer getting angrier or calmer?)
    # 3. Set risk_level based on current emotion + trend
    # 4. Generate recommendation based on situation
    
    pass
```

### Implementation Tips for Wav2Vec

```python
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import torch

# Load model (do this once globally)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
)

def analyze_voice_emotion(audio: np.ndarray) -> dict:
    """Analyze emotion from audio using Wav2Vec."""
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.softmax(outputs.logits, dim=-1)[0]
    
    # Map to emotion labels
    emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    emotion_scores = {e: float(scores[i]) for i, e in enumerate(emotions)}
    
    return emotion_scores
```

### Implementation Tips for CAMeL

```python
from camel_tools.sentiment import SentimentAnalyzer

# Initialize (do once)
sa = SentimentAnalyzer.pretrained()

def analyze_text_emotion(text: str) -> str:
    """Analyze emotion from Arabic text."""
    result = sa.predict(text)
    
    # Map sentiment to emotion
    sentiment_map = {
        "positive": "happy",
        "negative": "angry",  # or analyze further
        "neutral": "neutral"
    }
    
    return sentiment_map.get(result, "neutral")
```

### Test Your Implementation

```bash
cd C:\VCAI
python scripts/teammate_tests/test_emotion.py
```

---

## 🧠 Person D: RAG + Memory + LLM

### Overview

You handle the "brain" of the system:
- **RAG**: Retrieve relevant property/company information
- **Memory**: Store and retrieve conversation history
- **LLM**: Generate customer responses in Egyptian Arabic

### Recommended Technologies

| Component | Technology | Purpose |
|-----------|------------|---------|
| RAG | **FAISS** + **Sentence Transformers** | Vector search |
| Memory | **Simple Dict/JSON** | Conversation storage |
| LLM | **Qwen 2.5 (GGUF)** | Response generation |

### Installation

```bash
# RAG dependencies
pip install faiss-cpu sentence-transformers

# LLM dependencies (for GGUF/llama.cpp)
pip install llama-cpp-python

# Or if using Hugging Face Transformers directly:
pip install transformers accelerate bitsandbytes
```

### Files to Create

```
rag/
└── rag_agent.py

memory/
└── memory_agent.py

llm/
└── llm_agent.py
```

---

### RAG Agent

```python
# rag/rag_agent.py
"""
RAG Agent using FAISS and Sentence Transformers.
Person D Implementation.
"""

def retrieve_context(query: str, top_k: int = 3) -> dict:
    """
    Retrieve relevant documents for a query.
    
    Args:
        query: str
            - Search query in Arabic
            - Example: "شقق في التجمع الخامس"
        
        top_k: int
            - Number of documents to retrieve
            - Default: 3
            - Range: 1-10
    
    Returns:
        dict: RAGContext with these EXACT keys:
            {
                "query": str,              # The original query
                "documents": [             # List of retrieved docs
                    {
                        "content": str,    # Document text
                        "source": str,     # Source file (e.g., "properties.pdf")
                        "score": float,    # Relevance score 0.0-1.0
                        "metadata": dict   # Additional info
                    },
                    ...
                ],
                "total_found": int         # Total matching documents
            }
    
    Example:
        >>> context = retrieve_context("شقق التجمع الخامس", top_k=3)
        >>> len(context["documents"])
        3
        >>> context["documents"][0]["content"]
        "شقة 120 متر في التجمع الخامس بسعر 850,000 جنيه"
    """
    
    # ═══════════════════════════════════════════════════════════════════
    # YOUR IMPLEMENTATION HERE
    # ═══════════════════════════════════════════════════════════════════
    
    # Suggested approach:
    # 1. Load/initialize FAISS index and sentence transformer
    # 2. Encode query to vector
    # 3. Search FAISS index
    # 4. Return top_k results with scores
    
    pass
```

### Implementation Tips for RAG

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Global variables (initialize once)
_model = None
_index = None
_documents = []

def _initialize():
    global _model, _index, _documents
    
    if _model is None:
        # Load multilingual model (supports Arabic)
        _model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        # Load your documents
        _documents = [
            {"content": "شقة 120 متر في التجمع الخامس...", "source": "properties.pdf"},
            {"content": "أسعار الشقق تبدأ من 500,000 جنيه...", "source": "pricing.pdf"},
            # Add more documents from data/documents/
        ]
        
        # Create embeddings
        texts = [d["content"] for d in _documents]
        embeddings = _model.encode(texts, convert_to_numpy=True)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        _index = faiss.IndexFlatIP(dimension)  # Inner product
        faiss.normalize_L2(embeddings)
        _index.add(embeddings)

def retrieve_context(query: str, top_k: int = 3) -> dict:
    _initialize()
    
    # Encode query
    query_vector = _model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vector)
    
    # Search
    scores, indices = _index.search(query_vector, top_k)
    
    # Build results
    documents = []
    for i, idx in enumerate(indices[0]):
        if idx < len(_documents):
            doc = _documents[idx].copy()
            doc["score"] = float(scores[0][i])
            doc["metadata"] = {}
            documents.append(doc)
    
    return {
        "query": query,
        "documents": documents,
        "total_found": len(documents)
    }
```

---

### Memory Agent

```python
# memory/memory_agent.py
"""
Memory Agent for conversation history.
Person D Implementation.
"""

from datetime import datetime
from typing import Optional

# Simple in-memory storage (you can use Redis, SQLite, etc.)
_sessions = {}


def store_message(session_id: str, message: dict) -> bool:
    """
    Store a conversation message.
    
    Args:
        session_id: str - Session identifier
        message: dict - Message with keys: id, turn, speaker, text, ...
    
    Returns:
        bool: True if successful
    """
    
    # ═══════════════════════════════════════════════════════════════════
    # YOUR IMPLEMENTATION HERE
    # ═══════════════════════════════════════════════════════════════════
    
    pass


def get_recent_messages(session_id: str, last_n: int = 10) -> list:
    """
    Get recent messages from a session.
    
    Args:
        session_id: str - Session identifier
        last_n: int - Number of recent messages (default: 10)
    
    Returns:
        list[dict]: Recent messages (oldest first)
    """
    
    pass


def store_checkpoint(session_id: str, checkpoint: dict) -> bool:
    """
    Store a memory checkpoint (summary of conversation).
    
    Args:
        session_id: str
        checkpoint: dict with keys: id, session_id, turn_range, summary, ...
    
    Returns:
        bool: True if successful
    """
    
    pass


def get_checkpoints(session_id: str) -> list:
    """Get all checkpoints for a session."""
    
    pass


def get_session_memory(session_id: str) -> dict:
    """
    Get full session memory.
    
    Returns:
        dict: SessionMemory with EXACT keys:
            {
                "session_id": str,
                "checkpoints": list,       # List of checkpoint dicts
                "recent_messages": list,   # List of recent messages
                "total_turns": int
            }
    """
    
    # ═══════════════════════════════════════════════════════════════════
    # YOUR IMPLEMENTATION HERE
    # ═══════════════════════════════════════════════════════════════════
    
    pass
```

### Simple Memory Implementation

```python
# Simple implementation example:

_sessions = {}

def store_message(session_id: str, message: dict) -> bool:
    if session_id not in _sessions:
        _sessions[session_id] = {"messages": [], "checkpoints": []}
    _sessions[session_id]["messages"].append(message)
    return True

def get_recent_messages(session_id: str, last_n: int = 10) -> list:
    if session_id not in _sessions:
        return []
    return _sessions[session_id]["messages"][-last_n:]

def get_session_memory(session_id: str) -> dict:
    if session_id not in _sessions:
        return {
            "session_id": session_id,
            "checkpoints": [],
            "recent_messages": [],
            "total_turns": 0
        }
    
    session = _sessions[session_id]
    return {
        "session_id": session_id,
        "checkpoints": session.get("checkpoints", []),
        "recent_messages": session["messages"][-10:],
        "total_turns": len(session["messages"]) // 2
    }
```

---

### LLM Agent

```python
# llm/llm_agent.py
"""
LLM Agent using Qwen 2.5 (GGUF format).
Person D Implementation.
"""


def generate_response(
    customer_text: str,
    emotion: dict,
    emotional_context: dict,
    persona: dict,
    memory: dict,
    rag_context: dict
) -> str:
    """
    Generate customer response using LLM.
    
    Args:
        customer_text: str
            - What the salesperson said
            - Example: "الشقة دي بكام؟"
        
        emotion: dict
            - EmotionResult from detect_emotion()
        
        emotional_context: dict
            - EmotionalContext from analyze_emotional_context()
        
        persona: dict
            - Persona configuration with personality_prompt
        
        memory: dict
            - SessionMemory with conversation history
        
        rag_context: dict
            - RAGContext with relevant documents
    
    Returns:
        str: Customer response in Egyptian Arabic
    
    Example:
        >>> response = generate_response(...)
        >>> response
        "الشقة دي ب 850 ألف، بس ممكن نتكلم في السعر لو جاد"
    """
    
    # ═══════════════════════════════════════════════════════════════════
    # YOUR IMPLEMENTATION HERE
    # ═══════════════════════════════════════════════════════════════════
    
    # Suggested approach:
    # 1. Build system prompt from persona
    # 2. Add context from RAG documents
    # 3. Add conversation history from memory
    # 4. Include emotional guidance
    # 5. Generate response with Qwen
    # 6. Return Egyptian Arabic text
    
    pass
```

### Implementation Tips for Qwen GGUF

```python
from llama_cpp import Llama

# Global model (load once)
_llm = None

def _load_model():
    global _llm
    if _llm is None:
        _llm = Llama(
            model_path="path/to/qwen2.5-7b-instruct.Q4_K_M.gguf",
            n_ctx=4096,
            n_threads=4,
        )
    return _llm

def generate_response(
    customer_text: str,
    emotion: dict,
    emotional_context: dict,
    persona: dict,
    memory: dict,
    rag_context: dict
) -> str:
    
    llm = _load_model()
    
    # Build prompt
    system_prompt = f"""أنت {persona['name']}، عميل في شركة عقارات.
{persona['personality_prompt']}

معلومات متاحة:
{chr(10).join([d['content'] for d in rag_context.get('documents', [])])}

الحالة العاطفية الحالية: {emotion.get('primary_emotion', 'neutral')}
مستوى الخطر: {emotional_context.get('risk_level', 'low')}

رد بالعامية المصرية. كن طبيعي وواقعي."""

    # Build conversation
    messages = []
    for msg in memory.get('recent_messages', [])[-6:]:
        role = "user" if msg['speaker'] == 'salesperson' else "assistant"
        messages.append({"role": role, "content": msg['text']})
    
    messages.append({"role": "user", "content": customer_text})
    
    # Generate
    response = llm.create_chat_completion(
        messages=[{"role": "system", "content": system_prompt}] + messages,
        max_tokens=256,
        temperature=0.7,
    )
    
    return response['choices'][0]['message']['content']
```

### Test Your Implementations

```bash
cd C:\VCAI

# Test RAG
python scripts/teammate_tests/test_rag.py

# Test Memory
python scripts/teammate_tests/test_memory.py

# Test LLM
python scripts/teammate_tests/test_llm.py
```

---

## 🧪 Testing Your Implementation

### Before Pushing - ALWAYS Run Tests!

```bash
cd C:\VCAI

# Activate virtual environment
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Run YOUR test script
python scripts/teammate_tests/test_<your_component>.py
```

### Test Scripts by Person

| Person | Command |
|--------|---------|
| B | `python scripts/teammate_tests/test_tts.py` |
| C | `python scripts/teammate_tests/test_emotion.py` |
| D | `python scripts/teammate_tests/test_rag.py` |
| D | `python scripts/teammate_tests/test_memory.py` |
| D | `python scripts/teammate_tests/test_llm.py` |

### What Tests Check

1. ✅ Function exists with correct name
2. ✅ Function has correct parameters
3. ✅ Function returns correct type
4. ✅ Return value has required keys
5. ✅ Values are in valid ranges
6. ✅ Works with different inputs

### Expected Output (Success)

```
============================================================
[Component] Validation Tests
============================================================

[Test 1] Checking function exists...
   ✅ Function imported successfully
[Test 2] Checking function signature...
   ✅ Parameters correct
...

============================================================
TEST SUMMARY
============================================================
  ✅ Function exists
  ✅ Function signature
  ✅ Basic call
  ...

Passed: 10/10

🎉 ALL TESTS PASSED! Your implementation is ready.
```

### Expected Output (Failure)

```
[Test 4] Checking return type...
   ❌ Wrong return type!
      Expected: np.ndarray
      Got: <class 'list'>

⚠️ 1 test(s) failed. Please fix the issues before pushing.
```

---

## 📤 Submission Workflow

### Step 1: Get Latest Code

```bash
cd C:\VCAI
git pull origin main
```

### Step 2: Create Your File

Create the file in the correct location:
- Person B: `tts/tts_agent.py`
- Person C: `emotion/emotion_agent.py`
- Person D: `rag/rag_agent.py`, `memory/memory_agent.py`, `llm/llm_agent.py`

### Step 3: Implement Your Function(s)

Follow the interface exactly as documented above.

### Step 4: Run Tests

```bash
python scripts/teammate_tests/test_<your_component>.py
```

### Step 5: Fix Any Failures

Keep fixing until ALL tests pass.

### Step 6: Commit and Push

```bash
git add .
git commit -m "Implement [component] - Person [X]"
git push origin main
```

### Step 7: Notify Ali

Let Ali know you've pushed. He will:
1. Pull your code
2. Run `python scripts/validate_all.py`
3. Test integration with the full system
4. Let you know if anything needs fixing

---

## ❓ FAQ & Troubleshooting

### Q: I don't have a GPU. Will it work?

**A:** Yes! All components can run on CPU:
- Wav2Vec: Works on CPU (slower but fine)
- CAMeL: CPU only
- FAISS: Use `faiss-cpu`
- Qwen GGUF: Works on CPU with llama-cpp-python

### Q: My test says "Import failed"

**A:** Check that:
1. Your file is in the correct location
2. Your function has the correct name
3. You have all required imports at the top

### Q: My test says "Wrong return type"

**A:** Check the interface documentation. Make sure you're returning exactly what's expected:
- TTS: `np.ndarray` (not list, not tensor)
- Emotion: `dict` (not class instance)
- RAG: `dict` (with specific keys)

### Q: How do I convert PyTorch tensor to numpy?

```python
# From tensor to numpy
audio_np = audio_tensor.cpu().numpy().astype(np.float32)

# Make sure it's 1D
audio_np = audio_np.flatten()
```

### Q: Where do I get the documents for RAG?

**A:** Check the `data/documents/` folder. You can also create your own test documents.

### Q: Can I use a different model than recommended?

**A:** Yes, as long as:
1. Your function matches the interface exactly
2. All tests pass
3. It works on CPU (for teammates without GPU)

### Q: I'm getting memory errors

**A:** Try:
- Using a smaller/quantized model
- Reducing batch size
- Using CPU instead of GPU for testing

---

## 📞 Contact

- **Ali (Person A)**: Integration issues, backend questions
- **GitHub Issues**: Bug reports, feature requests

---

<div align="center">

**Thank you for contributing to VCAI! 🎉**

Remember: Run your tests before pushing!

</div>