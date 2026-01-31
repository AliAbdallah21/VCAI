"""
Emotion Fusion Module
Combines audio and text emotion detection for more accurate results
"""

from typing import Dict
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# TYPE DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

class FusedEmotionResult(dict):
    """Fused emotion result combining audio and text"""
    def __init__(
        self, 
        primary_emotion: str, 
        confidence: float, 
        intensity: str,
        voice_emotion: str,
        text_emotion: str,
        scores: Dict[str, float],
        fusion_method: str
    ):
        super().__init__(
            primary_emotion=primary_emotion,
            confidence=confidence,
            intensity=intensity,
            voice_emotion=voice_emotion,
            text_emotion=text_emotion,
            scores=scores,
            fusion_method=fusion_method
        )


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Weights for fusion strategies
FUSION_WEIGHTS = {
    "audio_primary": {
        "audio": 0.7,
        "text": 0.3
    },
    "text_primary": {
        "audio": 0.3,
        "text": 0.7
    },
    "balanced": {
        "audio": 0.5,
        "text": 0.5
    },
    "adaptive": {
        # Weights adjusted based on confidence
        "high_confidence_threshold": 0.75,
        "low_confidence_threshold": 0.4
    }
}

# Emotion labels
EMOTION_LABELS = ["angry", "happy", "hesitant", "interested", "neutral"]


# ══════════════════════════════════════════════════════════════════════════════
# FUSION STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════

def _weighted_average_fusion(
    voice_scores: Dict[str, float],
    text_scores: Dict[str, float],
    audio_weight: float,
    text_weight: float
) -> Dict[str, float]:
    """
    Combine scores using weighted average.
    
    Args:
        voice_scores: Emotion scores from audio
        text_scores: Emotion scores from text
        audio_weight: Weight for audio scores (0-1)
        text_weight: Weight for text scores (0-1)
    
    Returns:
        Combined emotion scores
    """
    fused_scores = {}
    
    for emotion in EMOTION_LABELS:
        voice_score = voice_scores.get(emotion, 0.0)
        text_score = text_scores.get(emotion, 0.0)
        
        fused_scores[emotion] = (
            audio_weight * voice_score + 
            text_weight * text_score
        )
    
    # Normalize scores to sum to 1.0
    total = sum(fused_scores.values())
    if total > 0:
        fused_scores = {k: v / total for k, v in fused_scores.items()}
    
    return fused_scores


def _adaptive_fusion(
    voice_scores: Dict[str, float],
    text_scores: Dict[str, float],
    voice_confidence: float,
    text_confidence: float
) -> Dict[str, float]:
    """
    Adaptively weight audio and text based on their confidence levels.
    
    Args:
        voice_scores: Emotion scores from audio
        text_scores: Emotion scores from text
        voice_confidence: Confidence of voice detection
        text_confidence: Confidence of text detection
    
    Returns:
        Combined emotion scores
    """
    # Calculate adaptive weights based on confidence
    total_confidence = voice_confidence + text_confidence
    
    if total_confidence > 0:
        audio_weight = voice_confidence / total_confidence
        text_weight = text_confidence / total_confidence
    else:
        # Default to balanced if both have zero confidence
        audio_weight = 0.5
        text_weight = 0.5
    
    return _weighted_average_fusion(
        voice_scores,
        text_scores,
        audio_weight,
        text_weight
    )


def _agreement_boosting_fusion(
    voice_scores: Dict[str, float],
    text_scores: Dict[str, float],
    voice_emotion: str,
    text_emotion: str
) -> Dict[str, float]:
    """
    Boost confidence when both modalities agree, reduce when they disagree.
    
    Args:
        voice_scores: Emotion scores from audio
        text_scores: Emotion scores from text
        voice_emotion: Primary emotion from audio
        text_emotion: Primary emotion from text
    
    Returns:
        Combined emotion scores with agreement boosting
    """
    # Start with balanced fusion
    fused_scores = _weighted_average_fusion(
        voice_scores,
        text_scores,
        0.5,
        0.5
    )
    
    # If both agree on the same emotion, boost its score
    if voice_emotion == text_emotion:
        boost_factor = 1.3
        fused_scores[voice_emotion] *= boost_factor
        
        # Renormalize
        total = sum(fused_scores.values())
        if total > 0:
            fused_scores = {k: v / total for k, v in fused_scores.items()}
    
    return fused_scores


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION: FUSE EMOTIONS
# ══════════════════════════════════════════════════════════════════════════════

def fuse_emotions(
    voice_emotion_result: dict,
    text_emotion_result: dict,
    fusion_strategy: str = "adaptive"
) -> FusedEmotionResult:
    """
    Fuse audio and text emotion detection results.
    
    Args:
        voice_emotion_result: Result from detect_emotion() (audio)
        text_emotion_result: Result from detect_text_emotion() (text)
        fusion_strategy: Strategy to use for fusion
            - "audio_primary": Favor audio (70% audio, 30% text)
            - "text_primary": Favor text (30% audio, 70% text)
            - "balanced": Equal weight (50% audio, 50% text)
            - "adaptive": Weight by confidence (recommended)
            - "agreement": Boost when modalities agree
    
    Returns:
        FusedEmotionResult with:
            - primary_emotion: Final detected emotion
            - confidence: Combined confidence score
            - intensity: Emotion intensity
            - voice_emotion: Emotion detected from audio
            - text_emotion: Emotion detected from text
            - scores: Combined scores for all emotions
            - fusion_method: The fusion strategy used
    
    Example:
        >>> from voice_emotion import detect_emotion
        >>> from text_emotion import detect_text_emotion
        >>> 
        >>> voice_result = detect_emotion(text, audio)
        >>> text_result = detect_text_emotion(text)
        >>> fused = fuse_emotions(voice_result, text_result, "adaptive")
        >>> 
        >>> print(f"Voice detected: {fused['voice_emotion']}")
        >>> print(f"Text detected: {fused['text_emotion']}")
        >>> print(f"Final emotion: {fused['primary_emotion']}")
    """
    try:
        # Extract information from results
        voice_emotion = voice_emotion_result["primary_emotion"]
        text_emotion = text_emotion_result["primary_emotion"]
        voice_confidence = voice_emotion_result["confidence"]
        text_confidence = text_emotion_result["confidence"]
        voice_scores = voice_emotion_result["scores"]
        text_scores = text_emotion_result["scores"]
        
        # Apply fusion strategy
        if fusion_strategy == "audio_primary":
            fused_scores = _weighted_average_fusion(
                voice_scores,
                text_scores,
                FUSION_WEIGHTS["audio_primary"]["audio"],
                FUSION_WEIGHTS["audio_primary"]["text"]
            )
        
        elif fusion_strategy == "text_primary":
            fused_scores = _weighted_average_fusion(
                voice_scores,
                text_scores,
                FUSION_WEIGHTS["text_primary"]["audio"],
                FUSION_WEIGHTS["text_primary"]["text"]
            )
        
        elif fusion_strategy == "balanced":
            fused_scores = _weighted_average_fusion(
                voice_scores,
                text_scores,
                FUSION_WEIGHTS["balanced"]["audio"],
                FUSION_WEIGHTS["balanced"]["text"]
            )
        
        elif fusion_strategy == "agreement":
            fused_scores = _agreement_boosting_fusion(
                voice_scores,
                text_scores,
                voice_emotion,
                text_emotion
            )
        
        else:  # "adaptive" (default)
            fused_scores = _adaptive_fusion(
                voice_scores,
                text_scores,
                voice_confidence,
                text_confidence
            )
        
        # Get final emotion and confidence
        primary_emotion = max(fused_scores, key=fused_scores.get)
        confidence = fused_scores[primary_emotion]
        
        # Determine intensity based on confidence
        if confidence >= 0.7:
            intensity = "high"
        elif confidence >= 0.4:
            intensity = "medium"
        else:
            intensity = "low"
        
        return FusedEmotionResult(
            primary_emotion=primary_emotion,
            confidence=confidence,
            intensity=intensity,
            voice_emotion=voice_emotion,
            text_emotion=text_emotion,
            scores=fused_scores,
            fusion_method=fusion_strategy
        )
        
    except Exception as e:
        print(f"Error in emotion fusion: {e}")
        # Fallback to voice emotion if fusion fails
        return FusedEmotionResult(
            primary_emotion=voice_emotion_result["primary_emotion"],
            confidence=voice_emotion_result["confidence"],
            intensity=voice_emotion_result.get("intensity", "medium"),
            voice_emotion=voice_emotion_result["primary_emotion"],
            text_emotion=text_emotion_result.get("primary_emotion", "neutral"),
            scores=voice_emotion_result["scores"],
            fusion_method="fallback_audio"
        )


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def compare_modalities(
    voice_emotion_result: dict,
    text_emotion_result: dict
) -> dict:
    """
    Compare audio and text emotion detection results.
    
    Returns:
        Dictionary with comparison metrics:
            - agreement: Whether both modalities detected the same emotion
            - confidence_difference: Difference in confidence levels
            - dominant_modality: Which modality has higher confidence
    """
    voice_emotion = voice_emotion_result["primary_emotion"]
    text_emotion = text_emotion_result["primary_emotion"]
    voice_confidence = voice_emotion_result["confidence"]
    text_confidence = text_emotion_result["confidence"]
    
    return {
        "agreement": voice_emotion == text_emotion,
        "voice_emotion": voice_emotion,
        "text_emotion": text_emotion,
        "confidence_difference": abs(voice_confidence - text_confidence),
        "dominant_modality": "audio" if voice_confidence > text_confidence else "text",
        "voice_confidence": voice_confidence,
        "text_confidence": text_confidence
    }


# ══════════════════════════════════════════════════════════════════════════════
# TESTING
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test fusion with dummy data
    print("Testing Emotion Fusion")
    print("=" * 80)
    
    # Simulate voice emotion result
    voice_result = {
        "primary_emotion": "angry",
        "confidence": 0.85,
        "intensity": "high",
        "scores": {
            "angry": 0.85,
            "happy": 0.05,
            "hesitant": 0.03,
            "interested": 0.02,
            "neutral": 0.05
        }
    }
    
    # Simulate text emotion result
    text_result = {
        "primary_emotion": "angry",
        "confidence": 0.75,
        "scores": {
            "angry": 0.75,
            "happy": 0.10,
            "hesitant": 0.05,
            "interested": 0.05,
            "neutral": 0.05
        }
    }
    
    # Test different fusion strategies
    strategies = ["adaptive", "audio_primary", "text_primary", "balanced", "agreement"]
    
    for strategy in strategies:
        print(f"\n{strategy.upper()} Fusion:")
        fused = fuse_emotions(voice_result, text_result, strategy)
        print(f"  Voice: {fused['voice_emotion']} ({voice_result['confidence']:.2f})")
        print(f"  Text:  {fused['text_emotion']} ({text_result['confidence']:.2f})")
        print(f"  Fused: {fused['primary_emotion']} ({fused['confidence']:.2f})")
        print(f"  Intensity: {fused['intensity']}")
    
    # Test comparison
    print("\n" + "=" * 80)
    print("MODALITY COMPARISON:")
    comparison = compare_modalities(voice_result, text_result)
    print(f"  Agreement: {comparison['agreement']}")
    print(f"  Dominant: {comparison['dominant_modality']}")
    print(f"  Confidence difference: {comparison['confidence_difference']:.2f}")
