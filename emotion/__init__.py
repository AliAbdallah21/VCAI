"""
Emotion Detection Package
Provides voice emotion detection, text emotion detection, fusion, and emotional context analysis
"""

from .voice_emotion import detect_emotion, EmotionResult
from .text_emotion import detect_text_emotion, TextEmotionResult
from .fusion import fuse_emotions, FusedEmotionResult, compare_modalities
from .agent import analyze_emotional_context, EmotionalContext, Message
from .config import (
    EMOTION_LABELS,
    ID_TO_LABEL,
    INTENSITY_THRESHOLDS,
    RISK_THRESHOLDS,
    POSITIVE_EMOTIONS,
    NEGATIVE_EMOTIONS,
    AGENT_BEHAVIORS
)

__all__ = [
    # Main functions
    'detect_emotion',              # Audio emotion detection
    'detect_text_emotion',         # Text emotion detection
    'fuse_emotions',               # Combine audio + text
    'analyze_emotional_context',   # Emotional context analysis
    'compare_modalities',          # Compare audio vs text
    
    # Types
    'EmotionResult',
    'TextEmotionResult',
    'FusedEmotionResult',
    'EmotionalContext',
    'Message',
    
    # Configuration
    'EMOTION_LABELS',
    'ID_TO_LABEL',
    'INTENSITY_THRESHOLDS',
    'RISK_THRESHOLDS',
    'POSITIVE_EMOTIONS',
    'NEGATIVE_EMOTIONS',
    'AGENT_BEHAVIORS',
]

__version__ = '1.0.0'
