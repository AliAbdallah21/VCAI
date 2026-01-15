# scripts/teammate_tests/test_emotion.py
"""
Test script for Person C: Emotion Detection
Run this to validate your implementation before pushing.

Usage:
    cd C:\VCAI
    python scripts/teammate_tests/test_emotion.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# TEST CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

VALID_EMOTIONS = ["happy", "sad", "angry", "fearful", "surprised", "disgusted", "neutral"]
VALID_INTENSITIES = ["low", "medium", "high"]
VALID_TRENDS = ["improving", "worsening", "stable"]
VALID_RISK_LEVELS = ["low", "medium", "high"]

REQUIRED_EMOTION_KEYS = [
    "primary_emotion", "confidence", "voice_emotion", 
    "text_emotion", "intensity", "scores"
]

REQUIRED_SCORE_KEYS = [
    "happy", "sad", "angry", "fearful", 
    "surprised", "disgusted", "neutral"
]

REQUIRED_CONTEXT_KEYS = ["current", "trend", "recommendation", "risk_level"]

TEST_INPUTS = [
    ("مرحبا، أنا مبسوط جداً", "happy"),
    ("ده غالي أوي!", "angry"),
    ("أنا مش متأكد من الموضوع ده", "fearful"),
    ("تمام، ماشي", "neutral"),
]


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: Create sample audio
# ══════════════════════════════════════════════════════════════════════════════

def create_sample_audio(duration=3.0, sample_rate=16000):
    """Create sample audio for testing"""
    samples = int(duration * sample_rate)
    audio = np.random.randn(samples).astype(np.float32) * 0.1
    return audio


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: detect_emotion
# ══════════════════════════════════════════════════════════════════════════════

def test_detect_emotion_exists():
    """Test 1: Check if detect_emotion function exists"""
    print("\n[Test 1] Checking if detect_emotion function exists...")
    
    try:
        from emotion.emotion_agent import detect_emotion
        print("   ✅ Function imported successfully")
        return True, detect_emotion
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        print("   Make sure you created: emotion/emotion_agent.py")
        print("   With function: def detect_emotion(text: str, audio: np.ndarray) -> dict")
        return False, None


def test_detect_emotion_signature(func):
    """Test 2: Check function signature"""
    print("\n[Test 2] Checking detect_emotion signature...")
    
    import inspect
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())
    
    if 'text' in params and 'audio' in params:
        print(f"   ✅ Parameters correct: {params}")
        return True
    else:
        print(f"   ❌ Parameters wrong!")
        print(f"      Expected: ['text', 'audio']")
        print(f"      Got: {params}")
        return False


def test_detect_emotion_basic(func):
    """Test 3: Basic function call"""
    print("\n[Test 3] Testing basic detect_emotion call...")
    
    try:
        audio = create_sample_audio()
        result = func("مرحبا", audio)
        print("   ✅ Function returned without error")
        return True, result
    except Exception as e:
        print(f"   ❌ Function raised exception: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_detect_emotion_return_type(result):
    """Test 4: Check return type is dict"""
    print("\n[Test 4] Checking return type...")
    
    if isinstance(result, dict):
        print("   ✅ Return type is dict")
        return True
    else:
        print(f"   ❌ Wrong return type!")
        print(f"      Expected: dict")
        print(f"      Got: {type(result)}")
        return False


def test_detect_emotion_keys(result):
    """Test 5: Check required keys exist"""
    print("\n[Test 5] Checking required keys...")
    
    missing_keys = [k for k in REQUIRED_EMOTION_KEYS if k not in result]
    
    if not missing_keys:
        print(f"   ✅ All required keys present: {REQUIRED_EMOTION_KEYS}")
        return True
    else:
        print(f"   ❌ Missing keys: {missing_keys}")
        print(f"      Required: {REQUIRED_EMOTION_KEYS}")
        print(f"      Got: {list(result.keys())}")
        return False


def test_detect_emotion_primary(result):
    """Test 6: Check primary_emotion is valid"""
    print("\n[Test 6] Checking primary_emotion value...")
    
    primary = result.get("primary_emotion")
    
    if primary in VALID_EMOTIONS:
        print(f"   ✅ primary_emotion='{primary}' is valid")
        return True
    else:
        print(f"   ❌ Invalid primary_emotion: '{primary}'")
        print(f"      Valid options: {VALID_EMOTIONS}")
        return False


def test_detect_emotion_confidence(result):
    """Test 7: Check confidence is float 0-1"""
    print("\n[Test 7] Checking confidence value...")
    
    conf = result.get("confidence")
    
    if isinstance(conf, (int, float)) and 0.0 <= conf <= 1.0:
        print(f"   ✅ confidence={conf:.2f} is valid (0.0-1.0)")
        return True
    else:
        print(f"   ❌ Invalid confidence: {conf}")
        print("      Expected: float between 0.0 and 1.0")
        return False


def test_detect_emotion_intensity(result):
    """Test 8: Check intensity is valid"""
    print("\n[Test 8] Checking intensity value...")
    
    intensity = result.get("intensity")
    
    if intensity in VALID_INTENSITIES:
        print(f"   ✅ intensity='{intensity}' is valid")
        return True
    else:
        print(f"   ❌ Invalid intensity: '{intensity}'")
        print(f"      Valid options: {VALID_INTENSITIES}")
        return False


def test_detect_emotion_scores(result):
    """Test 9: Check scores dict"""
    print("\n[Test 9] Checking scores dict...")
    
    scores = result.get("scores", {})
    
    if not isinstance(scores, dict):
        print(f"   ❌ scores should be dict, got {type(scores)}")
        return False
    
    missing = [k for k in REQUIRED_SCORE_KEYS if k not in scores]
    if missing:
        print(f"   ❌ Missing score keys: {missing}")
        return False
    
    for key, val in scores.items():
        if not isinstance(val, (int, float)) or not 0.0 <= val <= 1.0:
            print(f"   ❌ Invalid score for '{key}': {val} (should be 0.0-1.0)")
            return False
    
    print(f"   ✅ All scores valid")
    return True


def test_detect_emotion_inputs(func):
    """Test 10: Test with different inputs"""
    print("\n[Test 10] Testing with different inputs...")
    
    all_passed = True
    for text, expected_emotion in TEST_INPUTS:
        try:
            audio = create_sample_audio()
            result = func(text, audio)
            if isinstance(result, dict) and "primary_emotion" in result:
                print(f"   ✅ '{text[:25]}...' -> {result['primary_emotion']}")
            else:
                print(f"   ❌ Invalid result for '{text[:25]}...'")
                all_passed = False
        except Exception as e:
            print(f"   ❌ Exception for '{text[:25]}...': {e}")
            all_passed = False
    
    return all_passed


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: analyze_emotional_context
# ══════════════════════════════════════════════════════════════════════════════

def test_analyze_context_exists():
    """Test 11: Check if analyze_emotional_context exists"""
    print("\n[Test 11] Checking if analyze_emotional_context function exists...")
    
    try:
        from emotion.emotion_agent import analyze_emotional_context
        print("   ✅ Function imported successfully")
        return True, analyze_emotional_context
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False, None


def test_analyze_context_basic(func, emotion_result):
    """Test 12: Basic analyze_emotional_context call"""
    print("\n[Test 12] Testing basic analyze_emotional_context call...")
    
    try:
        # Create sample history
        history = [
            {"id": "1", "turn": 1, "speaker": "salesperson", "text": "مرحبا"},
            {"id": "2", "turn": 1, "speaker": "vc", "text": "أهلاً"},
        ]
        
        result = func(emotion_result, history)
        print("   ✅ Function returned without error")
        return True, result
    except Exception as e:
        print(f"   ❌ Function raised exception: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_analyze_context_keys(result):
    """Test 13: Check required keys in context result"""
    print("\n[Test 13] Checking context result keys...")
    
    if not isinstance(result, dict):
        print(f"   ❌ Result should be dict, got {type(result)}")
        return False
    
    missing = [k for k in REQUIRED_CONTEXT_KEYS if k not in result]
    
    if not missing:
        print(f"   ✅ All required keys present: {REQUIRED_CONTEXT_KEYS}")
        return True
    else:
        print(f"   ❌ Missing keys: {missing}")
        return False


def test_analyze_context_values(result):
    """Test 14: Check context values"""
    print("\n[Test 14] Checking context values...")
    
    all_valid = True
    
    trend = result.get("trend")
    if trend not in VALID_TRENDS:
        print(f"   ❌ Invalid trend: '{trend}' (valid: {VALID_TRENDS})")
        all_valid = False
    else:
        print(f"   ✅ trend='{trend}' is valid")
    
    risk = result.get("risk_level")
    if risk not in VALID_RISK_LEVELS:
        print(f"   ❌ Invalid risk_level: '{risk}' (valid: {VALID_RISK_LEVELS})")
        all_valid = False
    else:
        print(f"   ✅ risk_level='{risk}' is valid")
    
    rec = result.get("recommendation")
    if not isinstance(rec, str) or len(rec) == 0:
        print(f"   ❌ recommendation should be non-empty string")
        all_valid = False
    else:
        print(f"   ✅ recommendation='{rec}'")
    
    return all_valid


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Emotion Detection Validation Tests")
    print("Person C Implementation")
    print("=" * 60)
    
    results = []
    
    # detect_emotion tests
    passed, detect_func = test_detect_emotion_exists()
    results.append(("detect_emotion exists", passed))
    if not passed:
        print("\n❌ Cannot continue without detect_emotion function")
        return
    
    results.append(("detect_emotion signature", test_detect_emotion_signature(detect_func)))
    
    passed, emotion_result = test_detect_emotion_basic(detect_func)
    results.append(("detect_emotion basic call", passed))
    if not passed:
        print("\n❌ Cannot continue - function raises exception")
        return
    
    results.append(("Return type", test_detect_emotion_return_type(emotion_result)))
    results.append(("Required keys", test_detect_emotion_keys(emotion_result)))
    results.append(("primary_emotion valid", test_detect_emotion_primary(emotion_result)))
    results.append(("confidence valid", test_detect_emotion_confidence(emotion_result)))
    results.append(("intensity valid", test_detect_emotion_intensity(emotion_result)))
    results.append(("scores valid", test_detect_emotion_scores(emotion_result)))
    results.append(("Different inputs", test_detect_emotion_inputs(detect_func)))
    
    # analyze_emotional_context tests
    passed, context_func = test_analyze_context_exists()
    results.append(("analyze_emotional_context exists", passed))
    
    if passed:
        passed, context_result = test_analyze_context_basic(context_func, emotion_result)
        results.append(("analyze_emotional_context basic", passed))
        
        if passed:
            results.append(("Context keys", test_analyze_context_keys(context_result)))
            results.append(("Context values", test_analyze_context_values(context_result)))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")
    
    print(f"\nPassed: {passed_count}/{total_count}")
    
    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED! Your implementation is ready.")
    else:
        print(f"\n⚠️ {total_count - passed_count} test(s) failed.")


if __name__ == "__main__":
    main()
