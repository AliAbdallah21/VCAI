# scripts/teammate_tests/test_llm.py
"""
Test script for Person D: LLM Agent
Run this to validate your implementation before pushing.

Usage:
    cd C:\VCAI
    python scripts/teammate_tests/test_llm.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ══════════════════════════════════════════════════════════════════════════════
# SAMPLE DATA
# ══════════════════════════════════════════════════════════════════════════════

SAMPLE_EMOTION = {
    "primary_emotion": "neutral",
    "confidence": 0.8,
    "voice_emotion": "neutral",
    "text_emotion": "neutral",
    "intensity": "medium",
    "scores": {
        "happy": 0.1, "sad": 0.1, "angry": 0.1,
        "fearful": 0.1, "surprised": 0.1, "disgusted": 0.1, "neutral": 0.4
    }
}

SAMPLE_EMOTIONAL_CONTEXT = {
    "current": SAMPLE_EMOTION,
    "trend": "stable",
    "recommendation": "be_professional",
    "risk_level": "low"
}

SAMPLE_PERSONA = {
    "id": "friendly_customer",
    "name": "عميل ودود",
    "name_en": "Friendly Customer",
    "description": "عميل لطيف وسهل التعامل",
    "personality_prompt": "أنت عميل ودود ومهتم بشراء شقة",
    "voice_id": "egyptian_male_01",
    "default_emotion": "neutral",
    "difficulty": "easy",
    "traits": ["ودود", "صبور"],
    "avatar_url": None
}

SAMPLE_MEMORY = {
    "session_id": "test_123",
    "checkpoints": [],
    "recent_messages": [],
    "total_turns": 0
}

SAMPLE_RAG_CONTEXT = {
    "query": "شقق",
    "documents": [
        {"content": "شقة 120 متر في التجمع", "source": "props.pdf", "score": 0.8, "metadata": {}}
    ],
    "total_found": 1
}


# ══════════════════════════════════════════════════════════════════════════════
# TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_function_exists():
    """Test 1: Check if generate_response exists"""
    print("\n[Test 1] Checking generate_response function...")
    
    try:
        from llm.llm_agent import generate_response
        print("   ✅ Function imported")
        return True, generate_response
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        print("   Make sure you created: llm/llm_agent.py")
        return False, None


def test_function_signature(func):
    """Test 2: Check function signature"""
    print("\n[Test 2] Checking function signature...")
    
    import inspect
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())
    
    required = ['customer_text', 'emotion', 'emotional_context', 'persona', 'memory', 'rag_context']
    missing = [p for p in required if p not in params]
    
    if not missing:
        print(f"   ✅ All required parameters present")
        return True
    else:
        print(f"   ❌ Missing parameters: {missing}")
        return False


def test_basic_call(func):
    """Test 3: Test basic function call"""
    print("\n[Test 3] Testing basic function call...")
    
    try:
        result = func(
            customer_text="الشقة دي بكام؟",
            emotion=SAMPLE_EMOTION,
            emotional_context=SAMPLE_EMOTIONAL_CONTEXT,
            persona=SAMPLE_PERSONA,
            memory=SAMPLE_MEMORY,
            rag_context=SAMPLE_RAG_CONTEXT
        )
        print("   ✅ Function returned without error")
        return True, result
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_return_type(result):
    """Test 4: Check return type is string"""
    print("\n[Test 4] Checking return type...")
    
    if isinstance(result, str):
        print(f"   ✅ Return type is str")
        return True
    else:
        print(f"   ❌ Expected str, got {type(result)}")
        return False


def test_non_empty_response(result):
    """Test 5: Check response is not empty"""
    print("\n[Test 5] Checking response content...")
    
    if result and len(result.strip()) > 0:
        print(f"   ✅ Response: '{result[:50]}...'")
        return True
    else:
        print("   ❌ Response is empty")
        return False


def test_arabic_response(result):
    """Test 6: Check response contains Arabic"""
    print("\n[Test 6] Checking if response is Arabic...")
    
    # Check for Arabic characters
    arabic_chars = any('\u0600' <= c <= '\u06FF' for c in result)
    
    if arabic_chars:
        print("   ✅ Response contains Arabic text")
        return True
    else:
        print("   ❌ Response should be in Arabic")
        return False


def test_different_inputs(func):
    """Test 7: Test with different customer texts"""
    print("\n[Test 7] Testing different inputs...")
    
    test_texts = [
        "مرحبا",
        "الشقة دي غالية أوي",
        "فين الموقع بالظبط؟",
        "طيب وإيه نظام الدفع؟",
    ]
    
    all_passed = True
    for text in test_texts:
        try:
            result = func(
                customer_text=text,
                emotion=SAMPLE_EMOTION,
                emotional_context=SAMPLE_EMOTIONAL_CONTEXT,
                persona=SAMPLE_PERSONA,
                memory=SAMPLE_MEMORY,
                rag_context=SAMPLE_RAG_CONTEXT
            )
            if isinstance(result, str) and len(result) > 0:
                print(f"   ✅ '{text}' -> '{result[:30]}...'")
            else:
                print(f"   ❌ Invalid response for '{text}'")
                all_passed = False
        except Exception as e:
            print(f"   ❌ Exception for '{text}': {e}")
            all_passed = False
    
    return all_passed


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("LLM Agent Validation Tests")
    print("Person D Implementation")
    print("=" * 60)
    
    results = []
    
    passed, func = test_function_exists()
    results.append(("Function exists", passed))
    if not passed:
        return
    
    results.append(("Function signature", test_function_signature(func)))
    
    passed, result = test_basic_call(func)
    results.append(("Basic call", passed))
    if not passed:
        return
    
    results.append(("Return type", test_return_type(result)))
    results.append(("Non-empty response", test_non_empty_response(result)))
    results.append(("Arabic response", test_arabic_response(result)))
    results.append(("Different inputs", test_different_inputs(func)))
    
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
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️ {total_count - passed_count} test(s) failed.")


if __name__ == "__main__":
    main()
