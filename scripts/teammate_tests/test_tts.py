# scripts/teammate_tests/test_tts.py
"""
Test script for Person B: TTS (Text-to-Speech)
Run this to validate your implementation before pushing.

Usage:
    cd C:\VCAI
    python scripts/teammate_tests/test_tts.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# TEST CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

EXPECTED_SAMPLE_RATE = 22050
EXPECTED_DTYPE = np.float32
MIN_AUDIO_DURATION = 0.5  # seconds
MAX_AUDIO_DURATION = 30.0  # seconds

VALID_VOICE_IDS = ["default", "egyptian_male_01", "egyptian_female_01"]
VALID_EMOTIONS = ["neutral", "happy", "frustrated", "interested", "hesitant"]

TEST_TEXTS = [
    "مرحبا بيك",
    "الشقة دي في موقع ممتاز",
    "السعر ده مناسب جداً للمنطقة",
    "أنا فاهم قلقك بس خليني أشرحلك",
]


# ══════════════════════════════════════════════════════════════════════════════
# TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_function_exists():
    """Test 1: Check if function exists and is importable"""
    print("\n[Test 1] Checking if text_to_speech function exists...")
    
    try:
        from tts.tts_agent import text_to_speech
        print("   ✅ Function imported successfully")
        return True, text_to_speech
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        print("   Make sure you created: tts/tts_agent.py")
        print("   With function: def text_to_speech(text, voice_id='default', emotion='neutral') -> np.ndarray")
        return False, None


def test_function_signature(func):
    """Test 2: Check function signature"""
    print("\n[Test 2] Checking function signature...")
    
    import inspect
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())
    
    expected_params = ['text', 'voice_id', 'emotion']
    
    if params[:3] == expected_params:
        print(f"   ✅ Parameters correct: {params}")
        return True
    else:
        print(f"   ❌ Parameters wrong!")
        print(f"      Expected: {expected_params}")
        print(f"      Got: {params}")
        return False


def test_basic_call(func):
    """Test 3: Basic function call"""
    print("\n[Test 3] Testing basic function call...")
    
    try:
        result = func("مرحبا")
        print(f"   ✅ Function returned without error")
        return True, result
    except Exception as e:
        print(f"   ❌ Function raised exception: {e}")
        return False, None


def test_return_type(result):
    """Test 4: Check return type is numpy array"""
    print("\n[Test 4] Checking return type...")
    
    if isinstance(result, np.ndarray):
        print(f"   ✅ Return type is np.ndarray")
        return True
    else:
        print(f"   ❌ Wrong return type!")
        print(f"      Expected: np.ndarray")
        print(f"      Got: {type(result)}")
        return False


def test_array_dtype(result):
    """Test 5: Check array dtype is float32"""
    print("\n[Test 5] Checking array dtype...")
    
    if result.dtype == EXPECTED_DTYPE:
        print(f"   ✅ Dtype is {result.dtype}")
        return True
    else:
        print(f"   ❌ Wrong dtype!")
        print(f"      Expected: {EXPECTED_DTYPE}")
        print(f"      Got: {result.dtype}")
        return False


def test_array_shape(result):
    """Test 6: Check array is 1D"""
    print("\n[Test 6] Checking array shape...")
    
    if len(result.shape) == 1:
        print(f"   ✅ Array is 1D with shape {result.shape}")
        return True
    else:
        print(f"   ❌ Array should be 1D!")
        print(f"      Expected: (n_samples,)")
        print(f"      Got: {result.shape}")
        return False


def test_audio_duration(result):
    """Test 7: Check audio duration is reasonable"""
    print("\n[Test 7] Checking audio duration...")
    
    duration = len(result) / EXPECTED_SAMPLE_RATE
    
    if MIN_AUDIO_DURATION <= duration <= MAX_AUDIO_DURATION:
        print(f"   ✅ Duration is {duration:.2f}s (valid range: {MIN_AUDIO_DURATION}-{MAX_AUDIO_DURATION}s)")
        return True
    else:
        print(f"   ❌ Duration out of range!")
        print(f"      Got: {duration:.2f}s")
        print(f"      Expected: {MIN_AUDIO_DURATION}-{MAX_AUDIO_DURATION}s")
        return False


def test_audio_values(result):
    """Test 8: Check audio values are in valid range"""
    print("\n[Test 8] Checking audio values...")
    
    min_val, max_val = result.min(), result.max()
    
    if -1.0 <= min_val and max_val <= 1.0:
        print(f"   ✅ Values in range [-1, 1]: min={min_val:.3f}, max={max_val:.3f}")
        return True
    else:
        print(f"   ❌ Values out of range!")
        print(f"      Got: min={min_val:.3f}, max={max_val:.3f}")
        print(f"      Expected: values between -1.0 and 1.0")
        return False


def test_different_texts(func):
    """Test 9: Test with different texts"""
    print("\n[Test 9] Testing with different texts...")
    
    all_passed = True
    for text in TEST_TEXTS:
        try:
            result = func(text)
            if not isinstance(result, np.ndarray) or len(result) == 0:
                print(f"   ❌ Failed for: '{text[:30]}...'")
                all_passed = False
            else:
                print(f"   ✅ '{text[:30]}...' -> {len(result)} samples")
        except Exception as e:
            print(f"   ❌ Exception for '{text[:30]}...': {e}")
            all_passed = False
    
    return all_passed


def test_voice_ids(func):
    """Test 10: Test different voice IDs"""
    print("\n[Test 10] Testing different voice IDs...")
    
    all_passed = True
    for voice_id in VALID_VOICE_IDS:
        try:
            result = func("مرحبا", voice_id=voice_id)
            if isinstance(result, np.ndarray) and len(result) > 0:
                print(f"   ✅ voice_id='{voice_id}' works")
            else:
                print(f"   ❌ voice_id='{voice_id}' returned invalid result")
                all_passed = False
        except Exception as e:
            print(f"   ❌ voice_id='{voice_id}' raised exception: {e}")
            all_passed = False
    
    return all_passed


def test_emotions(func):
    """Test 11: Test different emotions"""
    print("\n[Test 11] Testing different emotions...")
    
    all_passed = True
    for emotion in VALID_EMOTIONS:
        try:
            result = func("مرحبا", emotion=emotion)
            if isinstance(result, np.ndarray) and len(result) > 0:
                print(f"   ✅ emotion='{emotion}' works")
            else:
                print(f"   ❌ emotion='{emotion}' returned invalid result")
                all_passed = False
        except Exception as e:
            print(f"   ❌ emotion='{emotion}' raised exception: {e}")
            all_passed = False
    
    return all_passed


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("TTS (Text-to-Speech) Validation Tests")
    print("Person B Implementation")
    print("=" * 60)
    
    results = []
    
    # Test 1: Function exists
    passed, func = test_function_exists()
    results.append(("Function exists", passed))
    if not passed:
        print("\n" + "=" * 60)
        print("❌ FAILED: Cannot continue without the function")
        print("=" * 60)
        return
    
    # Test 2: Function signature
    passed = test_function_signature(func)
    results.append(("Function signature", passed))
    
    # Test 3: Basic call
    passed, result = test_basic_call(func)
    results.append(("Basic call", passed))
    if not passed:
        print("\n" + "=" * 60)
        print("❌ FAILED: Function raises exception")
        print("=" * 60)
        return
    
    # Test 4-8: Return value checks
    results.append(("Return type", test_return_type(result)))
    results.append(("Array dtype", test_array_dtype(result)))
    results.append(("Array shape", test_array_shape(result)))
    results.append(("Audio duration", test_audio_duration(result)))
    results.append(("Audio values", test_audio_values(result)))
    
    # Test 9-11: Different inputs
    results.append(("Different texts", test_different_texts(func)))
    results.append(("Voice IDs", test_voice_ids(func)))
    results.append(("Emotions", test_emotions(func)))
    
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
        print("You can now push your code to GitHub.")
    else:
        print(f"\n⚠️ {total_count - passed_count} test(s) failed.")
        print("Please fix the issues before pushing.")


if __name__ == "__main__":
    main()
