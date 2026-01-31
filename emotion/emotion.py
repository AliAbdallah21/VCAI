"""
Emotion Detection Module
Provides emotion detection from audio and emotional context analysis
"""

import torch
import numpy as np
from typing import Dict, List, TypedDict
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor
)
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# TYPE DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

class EmotionResult(TypedDict):
    """Emotion detection result"""
    primary_emotion: str
    confidence: float
    intensity: str
    scores: Dict[str, float]


class Message(TypedDict):
    """Message with emotion history"""
    text: str
    emotion: EmotionResult
    timestamp: float


class EmotionalContext(TypedDict):
    """Emotional context analysis result"""
    current: EmotionResult
    trend: str
    recommendation: str
    risk_level: str


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

# Risk assessment thresholds
RISK_THRESHOLDS = {
    "high": 0.75,
    "medium": 0.50,
    "low": 0.0
}


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADER
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
# PERSON C: Emotion Detection
# ══════════════════════════════════════════════════════════════════════════════

def detect_emotion(
    text: str,
    audio: np.ndarray,
    model_path: str = "./emotion-recognition-model/final"
) -> EmotionResult:
    """
    Detect emotion from audio.
    
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


# ══════════════════════════════════════════════════════════════════════════════
# PERSON C: Emotional Agent
# ══════════════════════════════════════════════════════════════════════════════

def analyze_emotional_context(
    current_emotion: EmotionResult,
    history: List[Message]
) -> EmotionalContext:
    """
    Analyze emotional context with conversation history.
    
    Args:
        current_emotion: Current emotion from detect_emotion()
        history: List of previous messages with emotions
    
    Returns:
        EmotionalContext with:
            - current: Current emotion result
            - trend: Emotion trend ("improving", "worsening", "stable")
            - recommendation: Agent recommendation ("be_gentle", "be_firm", "show_empathy", "normal")
            - risk_level: Risk level ("low", "medium", "high")
    
    Example:
        >>> context = analyze_emotional_context(current_emotion, history)
        >>> print(context["trend"])
        "worsening"
        >>> print(context["recommendation"])
        "show_empathy"
    """
    try:
        # If no history, return default context
        if not history or len(history) < 2:
            return EmotionalContext(
                current=current_emotion,
                trend="stable",
                recommendation=_get_recommendation(current_emotion),
                risk_level=_assess_risk(current_emotion, [])
            )
        
        # Analyze trend
        trend = _analyze_trend(current_emotion, history)
        
        # Get recommendation
        recommendation = _get_recommendation(current_emotion, trend)
        
        # Assess risk
        risk_level = _assess_risk(current_emotion, history)
        
        return EmotionalContext(
            current=current_emotion,
            trend=trend,
            recommendation=recommendation,
            risk_level=risk_level
        )
        
    except Exception as e:
        print(f"Error in emotional context analysis: {e}")
        return EmotionalContext(
            current=current_emotion,
            trend="stable",
            recommendation="normal",
            risk_level="low"
        )


def _analyze_trend(current_emotion: EmotionResult, history: List[Message]) -> str:
    """Analyze emotional trend from conversation history"""
    if len(history) < 2:
        return "stable"
    
    # Get recent emotions (last 3-5 messages)
    recent_messages = history[-5:]
    
    # Define positive and negative emotions
    positive_emotions = {"happy", "interested"}
    negative_emotions = {"angry", "hesitant"}
    
    # Calculate sentiment scores
    current_is_positive = current_emotion["primary_emotion"] in positive_emotions
    current_is_negative = current_emotion["primary_emotion"] in negative_emotions
    
    # Count positive/negative in history
    positive_count = sum(
        1 for msg in recent_messages
        if msg["emotion"]["primary_emotion"] in positive_emotions
    )
    negative_count = sum(
        1 for msg in recent_messages
        if msg["emotion"]["primary_emotion"] in negative_emotions
    )
    
    # Determine trend
    if current_is_positive:
        if positive_count > negative_count:
            return "improving"
        elif negative_count > positive_count:
            return "improving"  # Turning positive
        else:
            return "stable"
    
    elif current_is_negative:
        if negative_count > positive_count:
            return "worsening"
        elif positive_count > negative_count:
            return "worsening"  # Turning negative
        else:
            return "stable"
    
    else:  # neutral
        return "stable"


def _get_recommendation(current_emotion: EmotionResult, trend: str = "stable") -> str:
    """Get agent behavior recommendation based on emotion and trend"""
    emotion = current_emotion["primary_emotion"]
    confidence = current_emotion["confidence"]
    
    # High confidence angry -> show empathy
    if emotion == "angry" and confidence >= 0.6:
        return "show_empathy"
    
    # Hesitant with worsening trend -> be gentle
    if emotion == "hesitant" and trend == "worsening":
        return "be_gentle"
    
    # Hesitant but stable -> be gentle
    if emotion == "hesitant":
        return "be_gentle"
    
    # Happy or interested -> be firm (can be more direct)
    if emotion in ["happy", "interested"]:
        return "be_firm"
    
    # Angry but low confidence or improving -> normal
    if emotion == "angry" and (confidence < 0.6 or trend == "improving"):
        return "normal"
    
    # Default
    return "normal"


def _assess_risk(current_emotion: EmotionResult, history: List[Message]) -> str:
    """Assess risk level based on current emotion and history"""
    emotion = current_emotion["primary_emotion"]
    confidence = current_emotion["confidence"]
    
    # High confidence angry is high risk
    if emotion == "angry" and confidence >= RISK_THRESHOLDS["high"]:
        return "high"
    
    # Check for sustained negative emotions
    if history and len(history) >= 3:
        recent = history[-3:]
        angry_count = sum(
            1 for msg in recent
            if msg["emotion"]["primary_emotion"] == "angry"
        )
        
        # Multiple angry messages in a row
        if angry_count >= 2:
            return "high"
    
    # Medium confidence angry or hesitant is medium risk
    if emotion in ["angry", "hesitant"] and confidence >= RISK_THRESHOLDS["medium"]:
        return "medium"
    
    # Default is low risk
    return "low"


# ══════════════════════════════════════════════════════════════════════════════
# TESTING / EXAMPLE USAGE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Example usage
    print("Emotion Detection Module")
    print("=" * 80)
    
    # Create dummy audio (would be real audio in production)
    dummy_audio = np.random.randn(16000).astype(np.float32)
    
    # Detect emotion
    emotion = detect_emotion("ده غالي أوي!", dummy_audio)
    print(f"\nDetected emotion: {emotion['primary_emotion']}")
    print(f"Confidence: {emotion['confidence']:.2f}")
    print(f"Intensity: {emotion['intensity']}")
    print(f"Scores: {emotion['scores']}")
    
    # Analyze context
    history = []
    context = analyze_emotional_context(emotion, history)
    print(f"\nTrend: {context['trend']}")
    print(f"Recommendation: {context['recommendation']}")
    print(f"Risk level: {context['risk_level']}")
