# orchestration/mocks/mock_emotion.py
"""
Mock Emotion Detection functions.
Replace with real imports from emotion.agent when Person C completes their work.

USAGE:
    # Now (development):
    from orchestration.mocks.mock_emotion import detect_emotion, analyze_emotional_context
    
    # Later (integration):
    from emotion.agent import detect_emotion, analyze_emotional_context
"""

import random
import numpy as np
from typing import Optional

from shared.types import (
    EmotionResult, 
    EmotionalContext, 
    EmotionScores,
    Message
)


# Keywords that suggest certain emotions (simple rule-based mock)
EMOTION_KEYWORDS = {
    "angry": ["غالي", "كتير", "مش معقول", "ده إيه", "بتضحكوا", "نصب", "غلط"],
    "happy": ["حلو", "جميل", "ممتاز", "عظيم", "تمام", "كويس أوي", "رائع"],
    "sad": ["للأسف", "مش قادر", "صعب", "حزين", "مفيش أمل"],
    "fearful": ["خايف", "قلقان", "مش متأكد", "محتار", "مش عارف"],
    "surprised": ["بجد", "معقول", "إزاي", "مش مصدق", "غريب"],
    "neutral": []
}


def _analyze_text_emotion(text: str) -> tuple[str, float]:
    """Simple keyword-based emotion detection from text."""
    text_lower = text.lower()
    
    for emotion, keywords in EMOTION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                return emotion, 0.7 + random.uniform(0, 0.2)
    
    return "neutral", 0.8 + random.uniform(0, 0.15)


def _generate_emotion_scores(primary: str) -> EmotionScores:
    """Generate realistic emotion scores with primary emotion having highest score."""
    # Start with low base scores
    scores = {
        "happy": random.uniform(0.05, 0.15),
        "sad": random.uniform(0.05, 0.15),
        "angry": random.uniform(0.05, 0.15),
        "fearful": random.uniform(0.05, 0.15),
        "surprised": random.uniform(0.05, 0.15),
        "disgusted": random.uniform(0.02, 0.08),
        "neutral": random.uniform(0.1, 0.3)
    }
    
    # Set primary emotion to high score
    if primary in scores:
        scores[primary] = random.uniform(0.6, 0.85)
    
    # Normalize to sum to 1.0
    total = sum(scores.values())
    scores = {k: v / total for k, v in scores.items()}
    
    return scores


def detect_emotion(text: str, audio: np.ndarray) -> EmotionResult:
    """
    Mock emotion detection - uses simple keyword matching.
    
    INPUT:
        text: str - Arabic transcription from STT
        audio: np.ndarray - Audio data (ignored in mock)
    
    OUTPUT:
        EmotionResult - Detected emotion with confidence
    """
    # Analyze text (simple keyword matching)
    text_emotion, text_confidence = _analyze_text_emotion(text)
    
    # Mock voice emotion (random but correlated with text)
    if random.random() < 0.7:
        voice_emotion = text_emotion  # Usually matches text
    else:
        voice_emotion = random.choice(["neutral", "happy", "angry", "sad"])
    
    # Combine (in real implementation, this would be ML fusion)
    if text_emotion == voice_emotion:
        primary_emotion = text_emotion
        confidence = min(0.95, text_confidence + 0.1)
    else:
        # Text usually more reliable for this mock
        primary_emotion = text_emotion
        confidence = text_confidence * 0.9
    
    # Determine intensity
    if confidence > 0.8:
        intensity = "high"
    elif confidence > 0.6:
        intensity = "medium"
    else:
        intensity = "low"
    
    result: EmotionResult = {
        "primary_emotion": primary_emotion,
        "confidence": round(confidence, 3),
        "voice_emotion": voice_emotion,
        "text_emotion": text_emotion,
        "intensity": intensity,
        "scores": _generate_emotion_scores(primary_emotion)
    }
    
    print(f"[MOCK EMOTION] Detected: {primary_emotion} ({confidence:.2f})")
    print(f"[MOCK EMOTION] Text: {text_emotion}, Voice: {voice_emotion}")
    
    return result


def analyze_emotional_context(
    current_emotion: EmotionResult,
    history: list[Message]
) -> EmotionalContext:
    """
    Mock emotional context analysis.
    
    INPUT:
        current_emotion: EmotionResult - Current detected emotion
        history: list[Message] - Previous conversation messages
    
    OUTPUT:
        EmotionalContext - Analysis with trend and recommendations
    """
    # Analyze trend from history
    if len(history) < 2:
        trend = "stable"
    else:
        # Look at recent emotions
        recent_emotions = []
        for msg in history[-5:]:
            if msg.get("emotion"):
                recent_emotions.append(msg["emotion"].get("primary_emotion", "neutral"))
        
        negative_emotions = {"angry", "sad", "fearful", "frustrated"}
        positive_emotions = {"happy", "interested", "friendly"}
        
        recent_negative = sum(1 for e in recent_emotions if e in negative_emotions)
        recent_positive = sum(1 for e in recent_emotions if e in positive_emotions)
        
        current_is_negative = current_emotion["primary_emotion"] in negative_emotions
        current_is_positive = current_emotion["primary_emotion"] in positive_emotions
        
        if current_is_positive and recent_negative > recent_positive:
            trend = "improving"
        elif current_is_negative and recent_positive > recent_negative:
            trend = "worsening"
        else:
            trend = "stable"
    
    # Generate recommendation based on emotion
    recommendations = {
        "angry": "show_empathy",
        "frustrated": "be_gentle",
        "sad": "show_empathy",
        "fearful": "be_reassuring",
        "happy": "maintain_energy",
        "interested": "provide_details",
        "neutral": "be_engaging",
        "surprised": "clarify_information"
    }
    
    recommendation = recommendations.get(
        current_emotion["primary_emotion"], 
        "be_professional"
    )
    
    # Determine risk level
    high_risk_emotions = {"angry", "frustrated"}
    medium_risk_emotions = {"sad", "fearful", "hesitant"}
    
    if current_emotion["primary_emotion"] in high_risk_emotions:
        risk_level = "high"
    elif current_emotion["primary_emotion"] in medium_risk_emotions:
        risk_level = "medium"
    else:
        risk_level = "low"
    
    # Increase risk if trend is worsening
    if trend == "worsening" and risk_level == "low":
        risk_level = "medium"
    elif trend == "worsening" and risk_level == "medium":
        risk_level = "high"
    
    context: EmotionalContext = {
        "current": current_emotion,
        "trend": trend,
        "recommendation": recommendation,
        "risk_level": risk_level
    }
    
    print(f"[MOCK EMOTION] Context: trend={trend}, risk={risk_level}")
    print(f"[MOCK EMOTION] Recommendation: {recommendation}")
    
    return context


# For testing
if __name__ == "__main__":
    # Test detect_emotion
    test_cases = [
        "الشقة دي حلوة أوي",
        "ده غالي كتير، مش معقول",
        "أنا مش متأكد، محتاج أفكر",
        "تمام، الكلام ده منطقي"
    ]
    
    for text in test_cases:
        print(f"\nText: '{text}'")
        audio = np.zeros(16000, dtype=np.float32)
        result = detect_emotion(text, audio)
        print(f"Result: {result['primary_emotion']} ({result['confidence']:.2f})")
        print("-" * 50)
    
    # Test analyze_emotional_context
    print("\n\nTesting emotional context:")
    mock_history = [
        {"speaker": "salesperson", "text": "مرحبا", "emotion": {"primary_emotion": "neutral"}},
        {"speaker": "salesperson", "text": "السعر غالي", "emotion": {"primary_emotion": "frustrated"}}
    ]
    
    current = detect_emotion("ده كتير أوي عليا", np.zeros(16000, dtype=np.float32))
    context = analyze_emotional_context(current, mock_history)
    print(f"Final context: {context}")