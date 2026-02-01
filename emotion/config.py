"""
Configuration Module
Shared configuration and constants for emotion detection
"""

# ══════════════════════════════════════════════════════════════════════════════
# EMOTION LABELS
# ══════════════════════════════════════════════════════════════════════════════

EMOTION_LABELS = {
    "angry": 0,
    "happy": 1,
    "hesitant": 2,
    "interested": 3,
    "neutral": 4
}

ID_TO_LABEL = {v: k for k, v in EMOTION_LABELS.items()}


# ══════════════════════════════════════════════════════════════════════════════
# THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════════

# Intensity thresholds for emotion strength
INTENSITY_THRESHOLDS = {
    "high": 0.7,    # Confidence >= 0.7
    "medium": 0.4,  # Confidence >= 0.4
    "low": 0.0      # Confidence < 0.4
}

# Risk assessment thresholds
RISK_THRESHOLDS = {
    "high": 0.75,   # Confidence >= 0.75 (escalate to human)
    "medium": 0.50, # Confidence >= 0.50 (extra caution)
    "low": 0.0      # Confidence < 0.50 (normal handling)
}


# ══════════════════════════════════════════════════════════════════════════════
# MODEL CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Default model path
DEFAULT_MODEL_PATH = "C:/VCAI/emotion/model/final"
# Audio preprocessing settings
SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH = 80000  # 5 seconds at 16kHz


# ══════════════════════════════════════════════════════════════════════════════
# AGENT BEHAVIOR MAPPING
# ══════════════════════════════════════════════════════════════════════════════

# Map emotions to positive/negative categories
POSITIVE_EMOTIONS = {"happy", "interested"}
NEGATIVE_EMOTIONS = {"angry", "hesitant"}
NEUTRAL_EMOTIONS = {"neutral"}

# Agent behavior descriptions for each recommendation
AGENT_BEHAVIORS = {
    "show_empathy": {
        "description": "Customer is angry - be apologetic and understanding",
        "tone": "empathetic",
        "priority": "high"
    },
    "be_gentle": {
        "description": "Customer is hesitant - be patient and clear",
        "tone": "gentle",
        "priority": "medium"
    },
    "be_firm": {
        "description": "Customer is positive - be confident and direct",
        "tone": "confident",
        "priority": "low"
    },
    "normal": {
        "description": "Standard professional interaction",
        "tone": "professional",
        "priority": "low"
    }
}


# ══════════════════════════════════════════════════════════════════════════════
# CONVERSATION HISTORY SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

# How many recent messages to consider for trend analysis
TREND_ANALYSIS_WINDOW = 5

# Minimum messages required for meaningful trend analysis
MIN_HISTORY_FOR_TREND = 2

# Number of recent messages to check for sustained negative emotions
RISK_ASSESSMENT_WINDOW = 3
