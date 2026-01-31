"""
Emotional Agent Module
Analyzes emotional context and provides recommendations for agent behavior
"""

from typing import List, Dict


# ══════════════════════════════════════════════════════════════════════════════
# TYPE DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

class EmotionResult(dict):
    """Emotion detection result (should match voice_emotion.py)"""
    pass


class Message(dict):
    """Message with emotion history"""
    pass


class EmotionalContext(dict):
    """Emotional context analysis result"""
    def __init__(self, current: EmotionResult, trend: str, recommendation: str, risk_level: str):
        super().__init__(
            current=current,
            trend=trend,
            recommendation=recommendation,
            risk_level=risk_level
        )


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Risk assessment thresholds
RISK_THRESHOLDS = {
    "high": 0.75,
    "medium": 0.50,
    "low": 0.0
}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION: ANALYZE EMOTIONAL CONTEXT
# ══════════════════════════════════════════════════════════════════════════════

def analyze_emotional_context(
    current_emotion: EmotionResult,
    history: List[Message]
) -> EmotionalContext:
    """
    Analyze emotional context with conversation history.
    
    OWNER: Person C
    STATUS: ✅ Implemented
    
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


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

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
