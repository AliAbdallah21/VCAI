"""
Emotion Detection Package
Provides voice emotion detection and emotional context analysis
"""

from .voice_emotion import detect_emotion, EmotionResult
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
    'detect_emotion',
    'analyze_emotional_context',
    
    # Types
    'EmotionResult',
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
