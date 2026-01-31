"""
Voice Emotion Detection Module
Detects emotion from audio using Wav2Vec2 model
"""

import torch
import numpy as np
from typing import Dict
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor
)
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# TYPE DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

class EmotionResult(dict):
    """Emotion detection result"""
    def __init__(self, primary_emotion: str, confidence: float, intensity: str, scores: Dict[str, float]):
        super().__init__(
            primary_emotion=primary_emotion,
            confidence=confidence,
            intensity=intensity,
            scores=scores
        )


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

EMOTION_LABELS = {
    "angry": 0,
    "happy": 1,
    "hesitant": 2,
    "interested": 3,
    "neutral": 4
}

ID_TO_LABEL = {v: k for k, v in EMOTION_LABELS.items()}

# Intensity thresholds
INTENSITY_THRESHOLDS = {
    "high": 0.7,
    "medium": 0.4,
    "low": 0.0
}


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADER (SINGLETON)
# ══════════════════════════════════════════════════════════════════════════════

class EmotionDetector:
    """Singleton emotion detector model"""
    _instance = None
    _model = None
    _feature_extractor = None
    
    @classmethod
    def get_instance(cls, model_path: str = "./emotion-recognition-model/final"):
        """Get or create singleton instance"""
        if cls._instance is None:
            cls._instance = cls(model_path)
        return cls._instance
    
    def __init__(self, model_path: str):
        """Initialize model and feature extractor"""
        if EmotionDetector._model is None:
            print(f"Loading emotion detection model from {model_path}...")
            EmotionDetector._model = Wav2Vec2ForSequenceClassification.from_pretrained(
                model_path
            )
            EmotionDetector._feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                model_path
            )
            EmotionDetector._model.eval()
            
            # Move to GPU if available
            if torch.cuda.is_available():
                EmotionDetector._model = EmotionDetector._model.cuda()
                print("Model loaded on GPU")
            else:
                print("Model loaded on CPU")
    
    @property
    def model(self):
        return EmotionDetector._model
    
    @property
    def feature_extractor(self):
        return EmotionDetector._feature_extractor


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _get_intensity(confidence: float) -> str:
    """Determine emotion intensity from confidence score"""
    if confidence >= INTENSITY_THRESHOLDS["high"]:
        return "high"
    elif confidence >= INTENSITY_THRESHOLDS["medium"]:
        return "medium"
    else:
        return "low"


def _preprocess_audio(audio: np.ndarray, max_length: int = 80000) -> np.ndarray:
    """Preprocess audio array to fixed length"""
    # Ensure audio is 1D
    if len(audio.shape) > 1:
        audio = audio.flatten()
    
    # Pad or truncate to max_length
    if len(audio) > max_length:
        audio = audio[:max_length]
    else:
        audio = np.pad(audio, (0, max_length - len(audio)))
    
    return audio


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION: DETECT EMOTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_emotion(
    text: str,
    audio: np.ndarray,
    model_path: str = "./emotion-recognition-model/final"
) -> EmotionResult:
    """
    Detect emotion from audio.
    
    OWNER: Person C
    STATUS: ✅ Implemented
    
    Args:
        text: Arabic transcription (not currently used but kept for future text analysis)
        audio: Raw audio from speaker
            - Shape: (n_samples,) - 1D array
            - Sample rate: 16000 Hz
            - Dtype: float32
        model_path: Path to trained emotion model
    
    Returns:
        EmotionResult with:
            - primary_emotion: Detected emotion ("angry", "happy", "hesitant", "interested", "neutral")
            - confidence: Confidence score (0.0 to 1.0)
            - intensity: Emotion intensity ("low", "medium", "high")
            - scores: Dictionary with scores for all emotions
    
    Example:
        >>> emotion = detect_emotion("ده غالي أوي!", audio_data)
        >>> print(emotion["primary_emotion"])
        "angry"
        >>> print(emotion["confidence"])
        0.87
    """
    try:
        # Get model instance
        detector = EmotionDetector.get_instance(model_path)
        
        # Preprocess audio
        audio_processed = _preprocess_audio(audio)
        
        # Extract features
        inputs = detector.feature_extractor(
            audio_processed,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = detector.model(**inputs)
            logits = outputs.logits
        
        # Get probabilities
        probabilities = F.softmax(logits, dim=-1)
        probs = probabilities[0].cpu().numpy()
        
        # Get prediction
        predicted_id = int(np.argmax(probs))
        primary_emotion = ID_TO_LABEL[predicted_id]
        confidence = float(probs[predicted_id])
        
        # Build scores dictionary
        scores = {
            emotion: float(probs[idx])
            for emotion, idx in EMOTION_LABELS.items()
        }
        
        # Determine intensity
        intensity = _get_intensity(confidence)
        
        return EmotionResult(
            primary_emotion=primary_emotion,
            confidence=confidence,
            intensity=intensity,
            scores=scores
        )
        
    except Exception as e:
        # Fallback to neutral emotion if detection fails
        print(f"Error in emotion detection: {e}")
        return EmotionResult(
            primary_emotion="neutral",
            confidence=0.5,
            intensity="low",
            scores={emotion: 0.2 for emotion in EMOTION_LABELS.keys()}
        )
