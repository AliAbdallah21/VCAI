# shared/constants.py
"""
Shared constants for VCAI project.
All hardcoded values should be defined here.
"""

# ══════════════════════════════════════════════════════════════════════════════
# AUDIO CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# STT Audio Config
STT_SAMPLE_RATE = 16000              # Hz
STT_CHANNELS = 1                      # Mono
STT_DTYPE = "float32"
STT_MIN_DURATION = 0.5                # Minimum audio duration in seconds
STT_MAX_DURATION = 15.0               # Maximum audio duration in seconds

# TTS Audio Config
TTS_SAMPLE_RATE = 24000              # Hz
TTS_CHANNELS = 1                      # Mono
TTS_DTYPE = "float32"

# VAD (Voice Activity Detection)
VAD_SILENCE_THRESHOLD = 0.01          # Amplitude threshold for silence
VAD_SILENCE_DURATION = 0.8            # Seconds of silence to trigger processing
VAD_MIN_SPEECH_DURATION = 1.0         # Minimum speech duration to process
VAD_MAX_SPEECH_DURATION = 15.0        # Maximum speech duration before forced processing


# ══════════════════════════════════════════════════════════════════════════════
# MODEL CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# STT Model
STT_MODEL_NAME = "large-v3-turbo"
STT_DEVICE = "cuda"                   # "cuda" or "cpu"
STT_COMPUTE_TYPE = "float16"          # "float16", "int8", "float32"
STT_LANGUAGE = "ar"                   # Arabic
STT_BEAM_SIZE = 5

# LLM Model
LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LLM_MAX_TOKENS = 512
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.9

# Emotion Model
EMOTION_VOICE_MODEL = "emotion2vec"
EMOTION_TEXT_MODEL = "aubmindlab/bert-base-arabertv2"

# TTS Model
TTS_MODEL_NAME = "chatterbox"
TTS_DEFAULT_VOICE = "egyptian_male_01"

# RAG/Embeddings
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
FAISS_INDEX_TYPE = "IndexFlatIP"      # Inner product similarity
RAG_TOP_K = 3                         # Number of documents to 
DOCUMENTS_DIR = "data/documents"



# ══════════════════════════════════════════════════════════════════════════════
# MEMORY CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

CHECKPOINT_INTERVAL = 5               # Create checkpoint every N turns
RECENT_MESSAGES_COUNT = 10            # Number of recent messages to keep in context
MAX_CHECKPOINTS_IN_CONTEXT = 5        # Maximum checkpoints to include in LLM context


# ══════════════════════════════════════════════════════════════════════════════
# SESSION CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

SESSION_MAX_DURATION = 600            # Maximum session duration in seconds (10 min)
SESSION_IDLE_TIMEOUT = 60             # Timeout for idle sessions in seconds
SESSION_MAX_TURNS = 50                # Maximum turns per session


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

GRADE_THRESHOLDS = {
    "A": 90,
    "B": 80,
    "C": 70,
    "D": 60,
    "F": 0
}

EVALUATION_WEIGHTS = {
    "objectives_completed": 0.4,       # 40% weight
    "communication_quality": 0.3,      # 30% weight
    "emotional_handling": 0.2,         # 20% weight
    "response_time": 0.1               # 10% weight
}


# ══════════════════════════════════════════════════════════════════════════════
# PERSONA CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

PERSONA_DIFFICULTIES = ["easy", "medium", "hard"]

PERSONA_EMOTIONS = [
    "neutral",
    "friendly", 
    "frustrated",
    "hesitant",
    "angry",
    "interested"
]

DEFAULT_PERSONA_ID = "neutral_customer"


# ══════════════════════════════════════════════════════════════════════════════
# API CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"

# Rate limiting
RATE_LIMIT_REQUESTS = 100             # Requests per window
RATE_LIMIT_WINDOW = 60                # Window in seconds

# Pagination
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100


# ══════════════════════════════════════════════════════════════════════════════
# WEBSOCKET CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

WS_PING_INTERVAL = 30                 # Seconds between pings
WS_PING_TIMEOUT = 10                  # Seconds to wait for pong
WS_MAX_MESSAGE_SIZE = 10 * 1024 * 1024  # 10 MB max message size

WS_MESSAGE_TYPES = [
    "audio_chunk",
    "transcription",
    "emotion",
    "vc_response_text",
    "vc_response_audio",
    "session_start",
    "session_end",
    "error",
    "ping",
    "pong"
]


# ══════════════════════════════════════════════════════════════════════════════
# AUTH CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours
REFRESH_TOKEN_EXPIRE_DAYS = 7          # 7 days
PASSWORD_MIN_LENGTH = 8
PASSWORD_MAX_LENGTH = 128

JWT_ALGORITHM = "HS256"


# ══════════════════════════════════════════════════════════════════════════════
# FILE PATHS
# ══════════════════════════════════════════════════════════════════════════════

DATA_DIR = "data"
PERSONAS_DIR = f"{DATA_DIR}/personas"
SCENARIOS_DIR = f"{DATA_DIR}/scenarios"
DOCUMENTS_DIR = f"{DATA_DIR}/documents"
VOICES_DIR = f"{DATA_DIR}/voices"
AUDIO_SESSIONS_DIR = f"{DATA_DIR}/audio_sessions"


# ══════════════════════════════════════════════════════════════════════════════
# ERROR CODES
# ══════════════════════════════════════════════════════════════════════════════

ERROR_CODES = {
    # Auth errors (1xxx)
    "AUTH_INVALID_CREDENTIALS": 1001,
    "AUTH_TOKEN_EXPIRED": 1002,
    "AUTH_TOKEN_INVALID": 1003,
    "AUTH_UNAUTHORIZED": 1004,
    
    # User errors (2xxx)
    "USER_NOT_FOUND": 2001,
    "USER_ALREADY_EXISTS": 2002,
    "USER_INVALID_DATA": 2003,
    
    # Session errors (3xxx)
    "SESSION_NOT_FOUND": 3001,
    "SESSION_ALREADY_ACTIVE": 3002,
    "SESSION_EXPIRED": 3003,
    "SESSION_MAX_DURATION": 3004,
    
    # Persona errors (4xxx)
    "PERSONA_NOT_FOUND": 4001,
    
    # Scenario errors (5xxx)
    "SCENARIO_NOT_FOUND": 5001,
    
    # Audio errors (6xxx)
    "AUDIO_INVALID_FORMAT": 6001,
    "AUDIO_TOO_SHORT": 6002,
    "AUDIO_TOO_LONG": 6003,
    "AUDIO_PROCESSING_FAILED": 6004,
    
    # STT errors (7xxx)
    "STT_TRANSCRIPTION_FAILED": 7001,
    "STT_MODEL_NOT_LOADED": 7002,
    
    # Emotion errors (8xxx)
    "EMOTION_DETECTION_FAILED": 8001,
    
    # LLM errors (9xxx)
    "LLM_GENERATION_FAILED": 9001,
    "LLM_MODEL_NOT_LOADED": 9002,
    
    # TTS errors (10xxx)
    "TTS_SYNTHESIS_FAILED": 10001,
    "TTS_VOICE_NOT_FOUND": 10002,
    
    # RAG errors (11xxx)
    "RAG_RETRIEVAL_FAILED": 11001,
    "RAG_INDEX_NOT_FOUND": 11002,
    
    # WebSocket errors (12xxx)
    "WS_CONNECTION_FAILED": 12001,
    "WS_MESSAGE_INVALID": 12002,
    "WS_SESSION_NOT_FOUND": 12003,
}