# shared/interfaces.py
"""
Interface definitions for all VCAI components.
Each team member must implement their functions matching these signatures.

PERSON A (Ali): STT + Orchestration Agent
PERSON B: TTS + Persona Agent  
PERSON C: Emotion Detection + Emotional Agent
PERSON D: RAG + LLM + Memory Agent
"""

from typing import Optional
import numpy as np

from shared.types import (
    AudioData,
    Persona, PersonaSummary,
    Scenario, ScenarioSummary,
    EmotionResult, EmotionalContext,
    Message, SessionMemory, MemoryCheckpoint,
    RAGDocument, RAGContext,
    ConversationState,
    Evaluation
)


# ══════════════════════════════════════════════════════════════════════════════
# PERSON A: STT (Speech-to-Text) - Ali's Component ✅
# ══════════════════════════════════════════════════════════════════════════════

def transcribe_audio(audio_data: np.ndarray) -> str:
    """
    Convert speech audio to text.
    
    OWNER: Person A (Ali)
    STATUS: ✅ Completed
    
    INPUT:
        audio_data: np.ndarray
            - Shape: (n_samples,) - 1D array
            - Sample rate: 16000 Hz
            - Dtype: float32
            - Duration: 1-15 seconds recommended
    
    OUTPUT:
        str: Transcribed Arabic text
    
    EXAMPLE:
        >>> audio = np.random.randn(16000 * 5).astype(np.float32)  # 5 seconds
        >>> text = transcribe_audio(audio)
        >>> print(text)
        "أنا مش متأكد من السعر ده"
    """
    pass


# ══════════════════════════════════════════════════════════════════════════════
# PERSON B: TTS (Text-to-Speech)
# ══════════════════════════════════════════════════════════════════════════════

def text_to_speech(
    text: str,
    voice_id: str = "default",
    emotion: str = "neutral"
) -> np.ndarray:
    """
    Convert text to speech audio.
    
    OWNER: Person B
    STATUS: 🔶 Pending
    
    INPUT:
        text: str
            - Arabic text to speak
            - Example: "الشقة دي في موقع ممتاز"
            - Max length: 500 characters recommended
        
        voice_id: str
            - Which cloned voice to use
            - Options: "default", "egyptian_male_01", "egyptian_female_01"
            - Default: "default"
        
        emotion: str
            - Emotional tone for speech
            - Options: "neutral", "happy", "frustrated", "interested", "hesitant"
            - Default: "neutral"
    
    OUTPUT:
        np.ndarray:
            - Shape: (n_samples,) - 1D array
            - Sample rate: 22050 Hz
            - Dtype: float32
    
    EXAMPLE:
        >>> audio = text_to_speech("مرحبا بيك", voice_id="egyptian_male_01", emotion="friendly")
        >>> audio.shape
        (44100,)  # ~2 seconds of audio
    """
    pass


# ══════════════════════════════════════════════════════════════════════════════
# PERSON B: Persona Agent
# ══════════════════════════════════════════════════════════════════════════════

def get_persona(persona_id: str) -> Persona:
    """
    Get full persona configuration by ID.
    
    OWNER: Person B
    STATUS: 🔶 Pending
    
    INPUT:
        persona_id: str
            - Unique persona identifier
            - Examples: "difficult_customer", "friendly_customer", "rushed_customer"
    
    OUTPUT:
        Persona: Full persona configuration dict
            {
                "id": str,
                "name": str,                    # "عميل صعب"
                "name_en": str,                 # "Difficult Customer"
                "description": str,             # "عميل متشكك وبيفاصل كتير"
                "personality_prompt": str,      # System prompt for LLM
                "voice_id": str,                # For TTS
                "default_emotion": str,         # "neutral", "frustrated", etc.
                "difficulty": str,              # "easy", "medium", "hard"
                "traits": list[str],            # ["متشكك", "بيفاصل"]
                "avatar_url": str | None
            }
    
    RAISES:
        PersonaNotFoundError: If persona_id doesn't exist
    
    EXAMPLE:
        >>> persona = get_persona("difficult_customer")
        >>> print(persona["name"])
        "عميل صعب"
    """
    pass


def list_personas() -> list[PersonaSummary]:
    """
    Get list of all available personas.
    
    OWNER: Person B
    STATUS: 🔶 Pending
    
    INPUT:
        None
    
    OUTPUT:
        list[PersonaSummary]: List of persona summaries
            [
                {
                    "id": str,
                    "name": str,
                    "name_en": str,
                    "difficulty": str,
                    "avatar_url": str | None
                },
                ...
            ]
    
    EXAMPLE:
        >>> personas = list_personas()
        >>> len(personas)
        5
    """
    pass


# ══════════════════════════════════════════════════════════════════════════════
# PERSON C: Emotion Detection
# ══════════════════════════════════════════════════════════════════════════════

def detect_emotion(text: str, audio: np.ndarray) -> EmotionResult:
    """
    Detect emotion from text and audio.
    
    OWNER: Person C
    STATUS: 🔶 Pending
    
    INPUT:
        text: str
            - Arabic transcription from STT
            - Example: "ده غالي أوي!"
        
        audio: np.ndarray
            - Raw audio from speaker
            - Shape: (n_samples,) - 1D array
            - Sample rate: 16000 Hz
            - Dtype: float32
    
    OUTPUT:
        EmotionResult: Emotion detection result
            {
                "primary_emotion": str,         # "frustrated", "happy", "neutral", etc.
                "confidence": float,            # 0.0 to 1.0
                "voice_emotion": str,           # From audio analysis
                "text_emotion": str,            # From text analysis
                "intensity": str,               # "low", "medium", "high"
                "scores": {
                    "happy": float,             # 0.0 to 1.0
                    "sad": float,
                    "angry": float,
                    "fearful": float,
                    "surprised": float,
                    "disgusted": float,
                    "neutral": float
                }
            }
    
    EXAMPLE:
        >>> emotion = detect_emotion("ده غالي أوي!", audio_data)
        >>> print(emotion["primary_emotion"])
        "frustrated"
        >>> print(emotion["confidence"])
        0.87
    """
    pass


# ══════════════════════════════════════════════════════════════════════════════
# PERSON C: Emotional Agent
# ══════════════════════════════════════════════════════════════════════════════

def analyze_emotional_context(
    current_emotion: EmotionResult,
    history: list[Message]
) -> EmotionalContext:
    """
    Analyze emotional context with conversation history.
    
    OWNER: Person C
    STATUS: 🔶 Pending
    
    INPUT:
        current_emotion: EmotionResult
            - Output from detect_emotion()
        
        history: list[Message]
            - Previous conversation messages with emotions
    
    OUTPUT:
        EmotionalContext: Analysis result
            {
                "current": EmotionResult,       # Current emotion
                "trend": str,                   # "improving", "worsening", "stable"
                "recommendation": str,          # "be_gentle", "be_firm", "show_empathy"
                "risk_level": str               # "low", "medium", "high"
            }
    
    EXAMPLE:
        >>> context = analyze_emotional_context(current_emotion, history)
        >>> print(context["trend"])
        "worsening"
        >>> print(context["recommendation"])
        "show_empathy"
    """
    pass


# ══════════════════════════════════════════════════════════════════════════════
# PERSON D: RAG Agent
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_context(query: str, top_k: int = 3) -> RAGContext:
    """
    Retrieve relevant documents for a query.
    
    OWNER: Person D
    STATUS: 🔶 Pending
    
    INPUT:
        query: str
            - Search query (usually from conversation)
            - Example: "شقق في التجمع الخامس"
        
        top_k: int
            - Number of documents to retrieve
            - Default: 3
            - Range: 1-10
    
    OUTPUT:
        RAGContext: Retrieved documents
            {
                "query": str,
                "documents": [
                    {
                        "content": str,         # Document text
                        "source": str,          # "properties.pdf"
                        "score": float,         # 0.0 to 1.0
                        "metadata": dict
                    },
                    ...
                ],
                "total_found": int
            }
    
    EXAMPLE:
        >>> context = retrieve_context("شقق التجمع الخامس", top_k=3)
        >>> print(context["documents"][0]["content"])
        "شقة 120 متر في التجمع الخامس بسعر 850,000 جنيه"
    """
    pass


# ══════════════════════════════════════════════════════════════════════════════
# PERSON D: Memory Agent
# ══════════════════════════════════════════════════════════════════════════════

def store_message(session_id: str, message: Message) -> bool:
    """
    Store a conversation message.
    
    OWNER: Person D
    STATUS: 🔶 Pending
    
    INPUT:
        session_id: str
            - Session identifier
        
        message: Message
            - Message to store
    
    OUTPUT:
        bool: True if successful
    
    EXAMPLE:
        >>> success = store_message("session_123", message)
        >>> print(success)
        True
    """
    pass


def get_recent_messages(session_id: str, last_n: int = 10) -> list[Message]:
    """
    Get recent messages from a session.
    
    OWNER: Person D
    STATUS: 🔶 Pending
    
    INPUT:
        session_id: str
            - Session identifier
        
        last_n: int
            - Number of recent messages to retrieve
            - Default: 10
    
    OUTPUT:
        list[Message]: Recent messages (oldest first)
    
    EXAMPLE:
        >>> messages = get_recent_messages("session_123", last_n=5)
        >>> len(messages)
        5
    """
    pass


def store_checkpoint(session_id: str, checkpoint: MemoryCheckpoint) -> bool:
    """
    Store a memory checkpoint (summary).
    
    OWNER: Person D
    STATUS: 🔶 Pending
    
    INPUT:
        session_id: str
            - Session identifier
        
        checkpoint: MemoryCheckpoint
            - Checkpoint data to store
    
    OUTPUT:
        bool: True if successful
    """
    pass


def get_checkpoints(session_id: str) -> list[MemoryCheckpoint]:
    """
    Get all checkpoints for a session.
    
    OWNER: Person D
    STATUS: 🔶 Pending
    
    INPUT:
        session_id: str
            - Session identifier
    
    OUTPUT:
        list[MemoryCheckpoint]: All checkpoints (oldest first)
    """
    pass


def get_session_memory(session_id: str) -> SessionMemory:
    """
    Get full session memory (checkpoints + recent messages).
    
    OWNER: Person D
    STATUS: 🔶 Pending
    
    INPUT:
        session_id: str
            - Session identifier
    
    OUTPUT:
        SessionMemory: Full memory context
            {
                "session_id": str,
                "checkpoints": list[MemoryCheckpoint],
                "recent_messages": list[Message],
                "total_turns": int
            }
    """
    pass


# ══════════════════════════════════════════════════════════════════════════════
# PERSON D: LLM Agent
# ══════════════════════════════════════════════════════════════════════════════

def generate_response(
    customer_text: str,
    emotion: EmotionResult,
    emotional_context: EmotionalContext,
    persona: Persona,
    memory: SessionMemory,
    rag_context: RAGContext
) -> str:
    """
    Generate VC response using LLM.
    
    OWNER: Person D
    STATUS: 🔶 Pending
    
    INPUT:
        customer_text: str
            - What the salesperson said
            - Example: "الشقة دي بكام؟"
        
        emotion: EmotionResult
            - Detected emotion from salesperson
        
        emotional_context: EmotionalContext
            - Emotional analysis with history
        
        persona: Persona
            - VC persona configuration
        
        memory: SessionMemory
            - Conversation memory (checkpoints + recent)
        
        rag_context: RAGContext
            - Retrieved relevant documents
    
    OUTPUT:
        str: VC response in Egyptian Arabic
    
    EXAMPLE:
        >>> response = generate_response(
        ...     customer_text="الشقة دي بكام؟",
        ...     emotion=emotion,
        ...     emotional_context=context,
        ...     persona=persona,
        ...     memory=memory,
        ...     rag_context=rag
        ... )
        >>> print(response)
        "الشقة دي ب 850 ألف، بس ممكن نتكلم في السعر لو جاد"
    """
    pass


# ══════════════════════════════════════════════════════════════════════════════
# PERSON A: Orchestration Agent - Ali's Component
# ══════════════════════════════════════════════════════════════════════════════

def process_turn(state: ConversationState) -> ConversationState:
    """
    Process one conversation turn (salesperson speaks → VC responds).
    
    OWNER: Person A (Ali)
    STATUS: 🔶 Pending
    
    INPUT:
        state: ConversationState
            - Current conversation state with audio_input
    
    OUTPUT:
        ConversationState: Updated state with:
            - transcription: STT result
            - emotion: Detected emotion
            - emotional_context: Emotional analysis
            - llm_response: Generated response
            - audio_output: TTS result
            - history: Updated message history
    
    FLOW:
        1. STT: audio_input → transcription
        2. Emotion: transcription + audio → emotion
        3. Emotional Context: emotion + history → emotional_context
        4. Memory: Check if checkpoint needed
        5. RAG: Retrieve relevant context
        6. LLM: Generate response
        7. TTS: Convert response to audio
        8. Update history
    """
    pass


def create_checkpoint_summary(
    messages: list[Message],
    llm_summarize_func: callable
) -> MemoryCheckpoint:
    """
    Create a memory checkpoint from messages.
    
    OWNER: Person A (Ali)
    STATUS: 🔶 Pending
    
    INPUT:
        messages: list[Message]
            - Messages to summarize
        
        llm_summarize_func: callable
            - Function to call LLM for summarization
    
    OUTPUT:
        MemoryCheckpoint: Summary checkpoint
    """
    pass


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION (Shared between Person A orchestration and Backend)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_session(
    session_id: str,
    scenario: Scenario,
    messages: list[Message]
) -> Evaluation:
    """
    Evaluate a completed training session.
    
    OWNER: Person A (Ali) / Backend
    STATUS: 🔶 Pending
    
    INPUT:
        session_id: str
            - Session to evaluate
        
        scenario: Scenario
            - Scenario with objectives
        
        messages: list[Message]
            - Full conversation history
    
    OUTPUT:
        Evaluation: Evaluation result with scores and feedback
    """
    pass