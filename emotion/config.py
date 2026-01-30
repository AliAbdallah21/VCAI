# config.py

from pathlib import Path

class EmotionConfig:
    """Configuration for emotion recognition module"""
    
    # Paths
    BASE_DIR = Path(__file__).parent
    VOICE_MODEL_PATH = BASE_DIR / "models" / "emotion-recognition-model"
    
    # Voice Emotion Settings
    VOICE_EMOTIONS = {
        0: "angry",
        1: "happy",
        2: "hesitant",
        3: "interested",
        4: "neutral"
    }
    
    # Text Emotion Settings
    TEXT_EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
    TEXT_EMOTIONS = ["anger", "joy", "sadness", "fear", "surprise", "neutral"]
    
    # Audio Processing
    SAMPLE_RATE = 16000
    MAX_AUDIO_LENGTH = 5  # seconds
    
    # Fusion Settings
    VOICE_WEIGHT = 0.6
    TEXT_WEIGHT = 0.4
    
    # Device
    DEVICE = "cuda"  # or "cpu"
    
    # Thresholds
    MIN_CONFIDENCE = 0.5
    
    @classmethod
    def get_voice_model_path(cls):
        """Get voice model path"""
        return str(cls.VOICE_MODEL_PATH)
    
    @classmethod
    def get_emotion_labels(cls):
        """Get emotion labels"""
        return cls.VOICE_EMOTIONS