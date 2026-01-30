# __init__.py

"""
Emotion Recognition Module for VCAI
Provides voice, text, and multimodal emotion recognition
"""

from .agent import EmotionAgent
from .voice_emotion import EmotionalAgent
from .text_emotion import TextEmotionAnalyzer
from .fusion import EmotionFusion
from .config import EmotionConfig

__version__ = "1.0.0"
__author__ = "VCAI Team"

__all__ = [
    'EmotionAgent',          # Main orchestrator
    'EmotionalAgent',        # Voice emotion
    'TextEmotionAnalyzer',   # Text emotion
    'EmotionFusion',         # Fusion
    'EmotionConfig',         # Config
]

# For easy imports:
# from emotion import EmotionAgent