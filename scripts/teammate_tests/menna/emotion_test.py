# C:\VCAI\scripts\teammate_tests\menna\emotion_test.py
"""
Comprehensive test script for Emotion Detection module
Tests against shared/interfaces.py requirements
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

print(f"Project root: {project_root}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1: IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

def test_imports():
    print("\n" + "=" * 60)
    print("TEST 1: Import Emotion modules")
    print("=" * 60)
    
    results = {}
    
    # Core imports (required by interfaces.py)
    try:
        from emotion import detect_emotion
        print("✅ emotion.detect_emotion")
        results["detect_emotion"] = True
    except ImportError as e:
        print(f"❌ emotion.detect_emotion: {e}")
        results["detect_emotion"] = False
    
    try:
        from emotion import analyze_emotional_context
        print("✅ emotion.analyze_emotional_context")
        results["analyze_emotional_context"] = True
    except ImportError as e:
        print(f"❌ emotion.analyze_emotional_context: {e}")
        results["analyze_emotional_context"] = False
    
    # Additional imports
    try:
        from emotion import detect_text_emotion
        print("✅ emotion.detect_text_emotion")
        results["detect_text_emotion"] = True
    except ImportError as e:
        print(f"❌ emotion.detect_text_emotion: {e}")
        results["detect_text_emotion"] = False
    
    try:
        from emotion import fuse_emotions
        print("✅ emotion.fuse_emotions")
        results["fuse_emotions"] = True
    except ImportError as e:
        print(f"❌ emotion.fuse_emotions: {e}")
        results["fuse_emotions"] = False
    
    try:
        from emotion import compare_modalities
        print("✅ emotion.compare_modalities")
        results["compare_modalities"] = True
    except ImportError as e:
        print(f"❌ emotion.compare_modalities: {e}")
        results["compare_modalities"] = False
    
    # Types
    try:
        from emotion import EmotionResult, TextEmotionResult, FusedEmotionResult, EmotionalContext
        print("✅ All type classes imported")
        results["types"] = True
    except ImportError as e:
        print(f"❌ Type imports: {e}")
        results["types"] = False
    
    # Config
    try:
        from emotion import EMOTION_LABELS, ID_TO_LABEL, AGENT_BEHAVIORS
        print(f"✅ Config imported - Labels: {list(EMOTION_LABELS.keys())}")
        results["config"] = True
    except ImportError as e:
        print(f"❌ Config imports: {e}")
        results["config"] = False
    
    return all(results.values()), results


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2: FUNCTION SIGNATURES
# ══════════════════════════════════════════════════════════════════════════════

def test_signatures():
    print("\n" + "=" * 60)
    print("TEST 2: Check function signatures vs interfaces.py")
    print("=" * 60)
    
    import inspect
    from emotion import detect_emotion, analyze_emotional_context
    
    # Check detect_emotion signature
    sig = inspect.signature(detect_emotion)
    params = list(sig.parameters.keys())
    print(f"\ndetect_emotion({', '.join(params)})")
    
    expected_params = ["text", "audio"]
    missing = [p for p in expected_params if p not in params]
    
    if missing:
        print(f"  ⚠️ Missing parameters: {missing}")
    else:
        print(f"  ✅ All required parameters present")
    
    # Check optional params
    optional_params = [p for p in params if p not in expected_params]
    if optional_params:
        print(f"  ℹ️ Extra parameters (OK if defaults): {optional_params}")
    
    # Check analyze_emotional_context signature
    sig2 = inspect.signature(analyze_emotional_context)
    params2 = list(sig2.parameters.keys())
    print(f"\nanalyze_emotional_context({', '.join(params2)})")
    
    expected_params2 = ["current_emotion", "history"]
    missing2 = [p for p in expected_params2 if p not in params2]
    
    if missing2:
        print(f"  ⚠️ Missing parameters: {missing2}")
    else:
        print(f"  ✅ All required parameters present")
    
    return len(missing) == 0 and len(missing2) == 0


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3: MODEL FILE CHECK
# ══════════════════════════════════════════════════════════════════════════════

def test_model_files():
    print("\n" + "=" * 60)
    print("TEST 3: Check emotion model files")
    print("=" * 60)
    
    # Check default model path from config
    try:
        from emotion.config import DEFAULT_MODEL_PATH
        model_path = DEFAULT_MODEL_PATH
        print(f"Config model path: {model_path}")
    except:
        model_path = "./emotion-recognition-model/final"
        print(f"Using default path: {model_path}")
    
    # Check relative to project root
    paths_to_check = [
        Path(model_path),
        project_root / model_path,
        project_root / "emotion" / "models" / "emotion-recognition-model",
        project_root / "emotion-recognition-model" / "final",
        project_root / "emotion-recognition-model",
    ]
    
    found = False
    for path in paths_to_check:
        if path.exists():
            print(f"  ✅ Found: {path}")
            # List contents
            if path.is_dir():
                for f in path.iterdir():
                    size_mb = f.stat().st_size / (1024*1024) if f.is_file() else 0
                    print(f"      {f.name} ({size_mb:.1f} MB)" if size_mb > 0 else f"      {f.name}/")
            found = True
            break
        else:
            print(f"  ❌ Not found: {path}")
    
    if not found:
        print("\n  ⚠️ Emotion model not found!")
        print("  The voice emotion detector needs a trained Wav2Vec2 model.")
        print("  Text emotion will still work (keyword-based or Arabic BERT).")
        print("  Ask Person C (Menna) for the model files.")
    
    return found


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4: TEXT EMOTION DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def test_text_emotion():
    print("\n" + "=" * 60)
    print("TEST 4: Text Emotion Detection")
    print("=" * 60)
    
    from emotion import detect_text_emotion
    
    test_cases = [
        ("ده غالي أوي!", "angry"),
        ("رائع جدا أنا مبسوط", "happy"),
        ("مش متأكد ممكن أفكر", "hesitant"),
        ("عايز أعرف التفاصيل", "interested"),
        ("أهلا السلام عليكم", "neutral"),
        ("", "neutral"),  # Empty text
    ]
    
    all_passed = True
    
    for text, expected in test_cases:
        try:
            start = time.time()
            result = detect_text_emotion(text)
            elapsed = time.time() - start
            
            emotion = result["primary_emotion"]
            confidence = result["confidence"]
            scores = result.get("scores", {})
            
            match = "✅" if emotion == expected else "⚠️"
            print(f"\n  {match} Text: '{text}'")
            print(f"     Expected: {expected} | Got: {emotion} ({confidence:.2f})")
            print(f"     Scores: {scores}")
            print(f"     Time: {elapsed:.3f}s")
            
            # Validate output format
            if "primary_emotion" not in result:
                print(f"     ❌ Missing 'primary_emotion' in result")
                all_passed = False
            if "confidence" not in result:
                print(f"     ❌ Missing 'confidence' in result")
                all_passed = False
            if "scores" not in result:
                print(f"     ❌ Missing 'scores' in result")
                all_passed = False
                
        except Exception as e:
            print(f"\n  ❌ Text: '{text}' -> Error: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    return all_passed


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5: VOICE EMOTION DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def test_voice_emotion():
    print("\n" + "=" * 60)
    print("TEST 5: Voice Emotion Detection (detect_emotion)")
    print("=" * 60)
    
    from emotion import detect_emotion
    
    # Create synthetic audio samples
    sample_rate = 16000
    duration = 3  # seconds
    
    test_cases = [
        ("السلام عليكم", np.random.randn(sample_rate * duration).astype(np.float32) * 0.1, "Random noise"),
        ("ده غالي أوي", np.zeros(sample_rate * duration, dtype=np.float32), "Silence"),
        ("مرحبا", np.sin(2 * np.pi * 440 * np.linspace(0, duration, sample_rate * duration)).astype(np.float32), "440Hz tone"),
    ]
    
    all_passed = True
    
    for text, audio, desc in test_cases:
        try:
            print(f"\n  Testing: {desc} ({text})")
            print(f"     Audio: shape={audio.shape}, dtype={audio.dtype}, range=[{audio.min():.3f}, {audio.max():.3f}]")
            
            start = time.time()
            result = detect_emotion(text=text, audio=audio)
            elapsed = time.time() - start
            
            print(f"     Emotion: {result['primary_emotion']} ({result['confidence']:.2f})")
            print(f"     Intensity: {result.get('intensity', 'N/A')}")
            print(f"     Scores: {result.get('scores', {})}")
            print(f"     Time: {elapsed:.3f}s")
            
            # Validate output format matches interfaces.py
            required_keys = ["primary_emotion", "confidence"]
            for key in required_keys:
                if key not in result:
                    print(f"     ❌ Missing required key: '{key}'")
                    all_passed = False
            
            # Check confidence range
            if not 0.0 <= result["confidence"] <= 1.0:
                print(f"     ❌ Confidence out of range: {result['confidence']}")
                all_passed = False
            
            # Check emotion is valid
            valid_emotions = {"angry", "happy", "hesitant", "interested", "neutral"}
            if result["primary_emotion"] not in valid_emotions:
                print(f"     ⚠️ Unknown emotion: {result['primary_emotion']}")
            
            print(f"     ✅ Output format valid")
            
        except Exception as e:
            print(f"     ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    return all_passed


# ══════════════════════════════════════════════════════════════════════════════
# TEST 6: EMOTION FUSION
# ══════════════════════════════════════════════════════════════════════════════

def test_fusion():
    print("\n" + "=" * 60)
    print("TEST 6: Emotion Fusion")
    print("=" * 60)
    
    from emotion import fuse_emotions, compare_modalities
    
    # Simulate voice and text results
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
    
    strategies = ["adaptive", "audio_primary", "text_primary", "balanced", "agreement"]
    all_passed = True
    
    for strategy in strategies:
        try:
            fused = fuse_emotions(voice_result, text_result, strategy)
            print(f"\n  {strategy.upper()}:")
            print(f"     Fused: {fused['primary_emotion']} ({fused['confidence']:.2f})")
            print(f"     Voice: {fused.get('voice_emotion', 'N/A')}")
            print(f"     Text: {fused.get('text_emotion', 'N/A')}")
            
            # Validate output
            if "primary_emotion" not in fused:
                print(f"     ❌ Missing 'primary_emotion'")
                all_passed = False
            else:
                print(f"     ✅ Valid output")
                
        except Exception as e:
            print(f"     ❌ {strategy} failed: {e}")
            all_passed = False
    
    # Test comparison
    try:
        comparison = compare_modalities(voice_result, text_result)
        print(f"\n  Comparison:")
        print(f"     Agreement: {comparison['agreement']}")
        print(f"     Dominant: {comparison['dominant_modality']}")
        print(f"     ✅ Comparison works")
    except Exception as e:
        print(f"     ❌ Comparison failed: {e}")
        all_passed = False
    
    return all_passed


# ══════════════════════════════════════════════════════════════════════════════
# TEST 7: EMOTIONAL CONTEXT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def test_emotional_context():
    print("\n" + "=" * 60)
    print("TEST 7: Emotional Context Analysis")
    print("=" * 60)
    
    from emotion import analyze_emotional_context
    
    all_passed = True
    
    # Test 1: No history
    print("\n  Test 7a: No history (first turn)")
    try:
        emotion = {
            "primary_emotion": "neutral",
            "confidence": 0.6,
            "scores": {"angry": 0.1, "happy": 0.1, "hesitant": 0.1, "interested": 0.1, "neutral": 0.6}
        }
        context = analyze_emotional_context(emotion, [])
        print(f"     Trend: {context['trend']}")
        print(f"     Recommendation: {context['recommendation']}")
        print(f"     Risk: {context['risk_level']}")
        print(f"     ✅ Works with empty history")
    except Exception as e:
        print(f"     ❌ Error: {e}")
        all_passed = False
    
    # Test 2: With history (worsening)
    print("\n  Test 7b: Worsening emotional trend")
    try:
        history = [
            {"role": "user", "text": "مرحبا", "emotion": {"primary_emotion": "happy", "confidence": 0.7}},
            {"role": "user", "text": "مش متأكد", "emotion": {"primary_emotion": "hesitant", "confidence": 0.6}},
            {"role": "user", "text": "ده غالي", "emotion": {"primary_emotion": "angry", "confidence": 0.8}},
        ]
        angry_emotion = {
            "primary_emotion": "angry",
            "confidence": 0.85,
            "scores": {"angry": 0.85, "happy": 0.05, "hesitant": 0.03, "interested": 0.02, "neutral": 0.05}
        }
        context = analyze_emotional_context(angry_emotion, history)
        print(f"     Trend: {context['trend']}")
        print(f"     Recommendation: {context['recommendation']}")
        print(f"     Risk: {context['risk_level']}")
        
        if context["risk_level"] in ["medium", "high"]:
            print(f"     ✅ Correctly identified elevated risk")
        else:
            print(f"     ⚠️ Expected medium/high risk for angry customer")
        
    except Exception as e:
        print(f"     ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # Test 3: With history (improving)
    print("\n  Test 7c: Improving emotional trend")
    try:
        history = [
            {"role": "user", "text": "ده غالي", "emotion": {"primary_emotion": "angry", "confidence": 0.8}},
            {"role": "user", "text": "طيب", "emotion": {"primary_emotion": "neutral", "confidence": 0.6}},
            {"role": "user", "text": "كويس", "emotion": {"primary_emotion": "happy", "confidence": 0.7}},
        ]
        happy_emotion = {
            "primary_emotion": "happy",
            "confidence": 0.8,
            "scores": {"angry": 0.05, "happy": 0.8, "hesitant": 0.05, "interested": 0.05, "neutral": 0.05}
        }
        context = analyze_emotional_context(happy_emotion, history)
        print(f"     Trend: {context['trend']}")
        print(f"     Recommendation: {context['recommendation']}")
        print(f"     Risk: {context['risk_level']}")
        
        if context["risk_level"] == "low":
            print(f"     ✅ Correctly identified low risk")
        
    except Exception as e:
        print(f"     ❌ Error: {e}")
        all_passed = False
    
    # Validate output format
    print("\n  Test 7d: Output format validation")
    try:
        context = analyze_emotional_context(
            {"primary_emotion": "neutral", "confidence": 0.5, "scores": {}},
            []
        )
        required_keys = ["current", "trend", "recommendation", "risk_level"]
        for key in required_keys:
            if key in context:
                print(f"     ✅ Has '{key}': {context[key]}")
            else:
                print(f"     ❌ Missing '{key}'")
                all_passed = False
                
    except Exception as e:
        print(f"     ❌ Error: {e}")
        all_passed = False
    
    return all_passed


# ══════════════════════════════════════════════════════════════════════════════
# TEST 8: INTEGRATION WITH ORCHESTRATION NODE
# ══════════════════════════════════════════════════════════════════════════════

def test_orchestration_integration():
    print("\n" + "=" * 60)
    print("TEST 8: Integration with emotion_node.py")
    print("=" * 60)
    
    try:
        from orchestration.nodes.emotion_node import emotion_node
        from orchestration.config import OrchestrationConfig
        
        # Create mock state
        state = {
            "transcription": "ده غالي أوي يا باشا",
            "audio_input": np.random.randn(16000 * 3).astype(np.float32) * 0.1,
            "history": [],
            "node_timings": {},
            "error": None,
        }
        
        config = OrchestrationConfig()
        config.use_mocks = False  # Use real emotion detection
        config.verbose = True
        config.enable_emotion = True
        
        print("\n  Running emotion_node with real detection...")
        result = emotion_node(state, config)
        
        emotion = result.get("emotion", {})
        context = result.get("emotional_context", {})
        
        print(f"\n  Results:")
        print(f"     Emotion: {emotion.get('primary_emotion', 'N/A')}")
        print(f"     Confidence: {emotion.get('confidence', 'N/A')}")
        print(f"     Context trend: {context.get('trend', 'N/A')}")
        print(f"     Risk: {context.get('risk_level', 'N/A')}")
        print(f"     Time: {result['node_timings'].get('emotion', 'N/A')}s")
        
        # Check if error occurred
        if result.get("error") and "Emotion" in str(result["error"]):
            print(f"\n  ⚠️ Emotion node had error: {result['error']}")
            print(f"     (Used fallback defaults)")
            return False
        else:
            print(f"\n  ✅ Emotion node works with real detection!")
            return True
        
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════════════════════════
# TEST 9: EDGE CASES
# ══════════════════════════════════════════════════════════════════════════════

def test_edge_cases():
    print("\n" + "=" * 60)
    print("TEST 9: Edge Cases")
    print("=" * 60)
    
    from emotion import detect_emotion, detect_text_emotion
    
    all_passed = True
    
    # Empty text
    print("\n  9a: Empty text")
    try:
        result = detect_text_emotion("")
        print(f"     ✅ Handled empty text: {result['primary_emotion']}")
    except Exception as e:
        print(f"     ❌ Failed on empty text: {e}")
        all_passed = False
    
    # Very short audio
    print("\n  9b: Very short audio (0.1s)")
    try:
        short_audio = np.random.randn(1600).astype(np.float32) * 0.1
        result = detect_emotion(text="تجربة", audio=short_audio)
        print(f"     ✅ Handled short audio: {result['primary_emotion']}")
    except Exception as e:
        print(f"     ❌ Failed on short audio: {e}")
        all_passed = False
    
    # Silent audio
    print("\n  9c: Silent audio")
    try:
        silent = np.zeros(48000, dtype=np.float32)
        result = detect_emotion(text="صمت", audio=silent)
        print(f"     ✅ Handled silence: {result['primary_emotion']}")
    except Exception as e:
        print(f"     ❌ Failed on silence: {e}")
        all_passed = False
    
    # Very long text
    print("\n  9d: Long text")
    try:
        long_text = "أنا مش عارف " * 100
        result = detect_text_emotion(long_text)
        print(f"     ✅ Handled long text: {result['primary_emotion']}")
    except Exception as e:
        print(f"     ❌ Failed on long text: {e}")
        all_passed = False
    
    # None audio
    print("\n  9e: Zero-length audio")
    try:
        empty_audio = np.array([], dtype=np.float32)
        result = detect_emotion(text="تجربة", audio=empty_audio)
        print(f"     ✅ Handled empty audio: {result['primary_emotion']}")
    except Exception as e:
        print(f"     ⚠️ Empty audio error (acceptable): {e}")
    
    return all_passed


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("VCAI Emotion Module Validation")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Imports
    passed, details = test_imports()
    results["1_imports"] = passed
    if not passed:
        print("\n❌ Cannot proceed without core imports")
        if not details.get("detect_emotion"):
            return
    
    # Test 2: Signatures
    results["2_signatures"] = test_signatures()
    
    # Test 3: Model files
    results["3_model_files"] = test_model_files()
    
    # Test 4: Text emotion
    results["4_text_emotion"] = test_text_emotion()
    
    # Test 5: Voice emotion
    results["5_voice_emotion"] = test_voice_emotion()
    
    # Test 6: Fusion
    results["6_fusion"] = test_fusion()
    
    # Test 7: Emotional context
    results["7_emotional_context"] = test_emotional_context()
    
    # Test 8: Orchestration integration
    results["8_orchestration"] = test_orchestration_integration()
    
    # Test 9: Edge cases
    results["9_edge_cases"] = test_edge_cases()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test}")
    
    total = len(results)
    passed_count = sum(results.values())
    print(f"\nTotal: {passed_count}/{total} tests passed")
    
    if passed_count == total:
        print("\n🎉 All emotion tests passed!")
    else:
        print("\n🔧 Some tests failed - check output above for details")


if __name__ == "__main__":
    main()