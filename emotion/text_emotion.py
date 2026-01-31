"""
Text Emotion Detection Module
Detects emotion from Arabic text using sentiment analysis
"""

from typing import Dict
from transformers import pipeline
import re


# ══════════════════════════════════════════════════════════════════════════════
# TYPE DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

class TextEmotionResult(dict):
    """Text emotion detection result"""
    def __init__(self, primary_emotion: str, confidence: float, scores: Dict[str, float]):
        super().__init__(
            primary_emotion=primary_emotion,
            confidence=confidence,
            scores=scores
        )


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Emotion keywords for Arabic text (fallback method)
EMOTION_KEYWORDS = {
    "angry": [
        "غاضب", "زعلان", "غالي", "مش عاجبني", "وحش", "سيء", "مزعج",
        "غلط", "خطأ", "مش كويس", "فاشل", "ظلم", "حرام"
    ],
    "happy": [
        "سعيد", "فرحان", "ممتاز", "رائع", "جميل", "حلو", "كويس",
        "عظيم", "مبسوط", "تمام", "جيد", "حاجة حلوة", "شكرا"
    ],
    "hesitant": [
        "مش متأكد", "ممكن", "يمكن", "مش عارف", "حاسس", "خايف",
        "متردد", "مش فاهم", "محتار", "شاك", "قلقان"
    ],
    "interested": [
        "عايز", "محتاج", "نفسي", "ممكن أعرف", "أيه", "كيف", "إزاي",
        "فين", "متى", "ليه", "أقدر", "ممكن تقولي", "مهتم"
    ],
    "neutral": [
        "أهلا", "مرحبا", "السلام عليكم", "صباح الخير", "مساء الخير",
        "تمام", "حاضر", "ماشي", "طيب"
    ]
}


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADER (SINGLETON)
# ══════════════════════════════════════════════════════════════════════════════

class TextEmotionDetector:
    """Singleton text emotion detector"""
    _instance = None
    _sentiment_analyzer = None
    
    @classmethod
    def get_instance(cls):
        """Get or create singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize sentiment analyzer"""
        if TextEmotionDetector._sentiment_analyzer is None:
            try:
                print("Loading Arabic sentiment analysis model...")
                # Try to load Arabic sentiment model
                # You can replace this with a better Arabic model if you have one
                TextEmotionDetector._sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="CAMeL-Lab/bert-base-arabic-camelbert-msa-sentiment"
                )
                print("Arabic sentiment model loaded successfully")
            except Exception as e:
                print(f"Could not load Arabic model: {e}")
                print("Using keyword-based fallback method")
                TextEmotionDetector._sentiment_analyzer = None
    
    @property
    def analyzer(self):
        return TextEmotionDetector._sentiment_analyzer


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_arabic_text(text: str) -> str:
    """Normalize Arabic text for better matching"""
    # Remove diacritics
    text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)
    # Normalize alef variants
    text = re.sub(r'[إأآا]', 'ا', text)
    # Normalize teh marbuta
    text = re.sub(r'ة', 'ه', text)
    return text.strip()


def _keyword_based_detection(text: str) -> TextEmotionResult:
    """Fallback: Detect emotion using keyword matching"""
    text_normalized = _normalize_arabic_text(text.lower())
    
    # Count matches for each emotion
    emotion_scores = {}
    for emotion, keywords in EMOTION_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            keyword_normalized = _normalize_arabic_text(keyword.lower())
            if keyword_normalized in text_normalized:
                score += 1
        emotion_scores[emotion] = score
    
    # Find emotion with highest score
    if sum(emotion_scores.values()) == 0:
        # No keywords matched, default to neutral
        return TextEmotionResult(
            primary_emotion="neutral",
            confidence=0.5,
            scores={emotion: 0.2 for emotion in EMOTION_KEYWORDS.keys()}
        )
    
    # Get primary emotion
    primary_emotion = max(emotion_scores, key=emotion_scores.get)
    total_matches = sum(emotion_scores.values())
    
    # Calculate normalized scores
    scores = {
        emotion: count / total_matches if total_matches > 0 else 0.0
        for emotion, count in emotion_scores.items()
    }
    
    confidence = scores[primary_emotion]
    
    return TextEmotionResult(
        primary_emotion=primary_emotion,
        confidence=confidence,
        scores=scores
    )


def _model_based_detection(text: str, analyzer) -> TextEmotionResult:
    """Detect emotion using transformer model"""
    try:
        result = analyzer(text)[0]
        label = result['label'].lower()
        score = result['score']
        
        # Map sentiment labels to emotions
        sentiment_to_emotion = {
            'positive': 'happy',
            'negative': 'angry',
            'neutral': 'neutral',
            'mixed': 'hesitant'
        }
        
        primary_emotion = sentiment_to_emotion.get(label, 'neutral')
        
        # Create scores dictionary
        scores = {
            "angry": 0.0,
            "happy": 0.0,
            "hesitant": 0.0,
            "interested": 0.0,
            "neutral": 0.0
        }
        scores[primary_emotion] = score
        
        # Distribute remaining probability
        remaining = 1.0 - score
        for emotion in scores:
            if emotion != primary_emotion:
                scores[emotion] = remaining / 4
        
        return TextEmotionResult(
            primary_emotion=primary_emotion,
            confidence=score,
            scores=scores
        )
        
    except Exception as e:
        print(f"Model detection failed: {e}, falling back to keywords")
        return _keyword_based_detection(text)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION: DETECT TEXT EMOTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_text_emotion(text: str) -> TextEmotionResult:
    """
    Detect emotion from Arabic text.
    
    Args:
        text: Arabic text to analyze
    
    Returns:
        TextEmotionResult with:
            - primary_emotion: Detected emotion ("angry", "happy", "hesitant", "interested", "neutral")
            - confidence: Confidence score (0.0 to 1.0)
            - scores: Dictionary with scores for all emotions
    
    Example:
        >>> emotion = detect_text_emotion("ده غالي أوي!")
        >>> print(emotion["primary_emotion"])
        "angry"
        >>> print(emotion["confidence"])
        0.75
    """
    if not text or len(text.strip()) == 0:
        return TextEmotionResult(
            primary_emotion="neutral",
            confidence=0.5,
            scores={emotion: 0.2 for emotion in EMOTION_KEYWORDS.keys()}
        )
    
    try:
        # Get detector instance
        detector = TextEmotionDetector.get_instance()
        
        # Use model if available, otherwise use keywords
        if detector.analyzer is not None:
            return _model_based_detection(text, detector.analyzer)
        else:
            return _keyword_based_detection(text)
            
    except Exception as e:
        print(f"Error in text emotion detection: {e}")
        # Fallback to keyword-based detection
        return _keyword_based_detection(text)


# ══════════════════════════════════════════════════════════════════════════════
# TESTING
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test examples
    test_texts = [
        "ده غالي أوي!",  # This is too expensive (angry)
        "رائع جدا، أنا مبسوط",  # Great, I'm happy
        "مش متأكد، ممكن أفكر",  # Not sure, maybe I'll think (hesitant)
        "عايز أعرف التفاصيل",  # I want to know the details (interested)
        "أهلا، إزيك",  # Hello, how are you (neutral)
    ]
    
    print("Testing Text Emotion Detection")
    print("=" * 80)
    
    for text in test_texts:
        emotion = detect_text_emotion(text)
        print(f"\nText: {text}")
        print(f"Emotion: {emotion['primary_emotion']}")
        print(f"Confidence: {emotion['confidence']:.2f}")
        print(f"Scores: {emotion['scores']}")
