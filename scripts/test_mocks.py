# scripts/test_mocks.py
"""
Test script to verify all mock functions work correctly.
Run this to ensure mocks are ready for orchestration development.

USAGE:
    cd C:\VCAI
    python scripts/test_mocks.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

def test_tts():
    """Test TTS mock."""
    print("\n" + "="*60)
    print("🔊 Testing TTS Mock")
    print("="*60)
    
    from orchestration.mocks import text_to_speech, get_available_voices
    
    # Test text_to_speech
    text = "مرحبا بيك في شركتنا"
    audio = text_to_speech(text, voice_id="egyptian_male_01", emotion="friendly")
    
    assert isinstance(audio, np.ndarray), "Output should be numpy array"
    assert audio.dtype == np.float32, "Audio should be float32"
    assert len(audio) > 0, "Audio should not be empty"
    
    print(f"✅ Generated audio: {len(audio)} samples, {len(audio)/22050:.2f}s")
    
    # Test get_available_voices
    voices = get_available_voices()
    assert len(voices) > 0, "Should have available voices"
    print(f"✅ Available voices: {len(voices)}")
    
    return True


def test_persona():
    """Test Persona mock."""
    print("\n" + "="*60)
    print("👤 Testing Persona Mock")
    print("="*60)
    
    from orchestration.mocks import get_persona, list_personas
    from shared.exceptions import PersonaNotFoundError
    
    # Test list_personas
    personas = list_personas()
    assert len(personas) > 0, "Should have personas"
    print(f"✅ Found {len(personas)} personas")
    
    # Test get_persona
    persona = get_persona("difficult_customer")
    assert persona["id"] == "difficult_customer"
    assert "personality_prompt" in persona
    print(f"✅ Loaded persona: {persona['name']}")
    
    # Test error handling
    try:
        get_persona("non_existent")
        assert False, "Should raise PersonaNotFoundError"
    except PersonaNotFoundError:
        print("✅ PersonaNotFoundError raised correctly")
    
    return True


def test_emotion():
    """Test Emotion mock."""
    print("\n" + "="*60)
    print("🎭 Testing Emotion Mock")
    print("="*60)
    
    from orchestration.mocks import detect_emotion, analyze_emotional_context
    
    # Test detect_emotion
    text = "ده غالي أوي!"
    audio = np.random.randn(16000 * 3).astype(np.float32)
    
    emotion = detect_emotion(text, audio)
    
    assert "primary_emotion" in emotion
    assert "confidence" in emotion
    assert 0 <= emotion["confidence"] <= 1
    print(f"✅ Detected emotion: {emotion['primary_emotion']} ({emotion['confidence']:.2f})")
    
    # Test analyze_emotional_context
    history = [
        {"speaker": "salesperson", "text": "مرحبا", "emotion": {"primary_emotion": "neutral"}}
    ]
    
    context = analyze_emotional_context(emotion, history)
    
    assert "trend" in context
    assert "risk_level" in context
    assert "recommendation" in context
    print(f"✅ Emotional context: trend={context['trend']}, risk={context['risk_level']}")
    
    return True


def test_rag():
    """Test RAG mock."""
    print("\n" + "="*60)
    print("📚 Testing RAG Mock")
    print("="*60)
    
    from orchestration.mocks import retrieve_context, get_document_count
    
    # Test retrieve_context
    result = retrieve_context("شقة في التجمع الخامس", top_k=3)
    
    assert "documents" in result
    assert "query" in result
    assert len(result["documents"]) <= 3
    print(f"✅ Retrieved {len(result['documents'])} documents")
    
    for i, doc in enumerate(result["documents"]):
        print(f"   {i+1}. {doc['source']} (score: {doc['score']:.2f})")
    
    # Test document count
    count = get_document_count()
    assert count > 0
    print(f"✅ Total documents: {count}")
    
    return True


def test_memory():
    """Test Memory mock."""
    print("\n" + "="*60)
    print("🧠 Testing Memory Mock")
    print("="*60)
    
    from orchestration.mocks import (
        store_message, get_recent_messages,
        store_checkpoint, get_checkpoints,
        get_session_memory, clear_session
    )
    
    session_id = "test_session_001"
    
    # Clear any existing data
    clear_session(session_id)
    
    # Test store_message
    message1 = {"turn": 1, "speaker": "salesperson", "text": "مرحبا", "emotion": None}
    message2 = {"turn": 1, "speaker": "vc", "text": "أهلاً بيك", "emotion": None}
    
    assert store_message(session_id, message1)
    assert store_message(session_id, message2)
    print("✅ Stored 2 messages")
    
    # Test get_recent_messages
    messages = get_recent_messages(session_id, last_n=10)
    assert len(messages) == 2
    print(f"✅ Retrieved {len(messages)} messages")
    
    # Test store_checkpoint
    checkpoint = {
        "turn_range": (1, 1),
        "summary": "بداية المحادثة - تحية",
        "key_points": ["تحية"],
        "customer_preferences": {},
        "objections_raised": []
    }
    
    assert store_checkpoint(session_id, checkpoint)
    print("✅ Stored checkpoint")
    
    # Test get_checkpoints
    checkpoints = get_checkpoints(session_id)
    assert len(checkpoints) == 1
    print(f"✅ Retrieved {len(checkpoints)} checkpoint(s)")
    
    # Test get_session_memory
    memory = get_session_memory(session_id)
    assert memory["session_id"] == session_id
    assert len(memory["checkpoints"]) == 1
    assert len(memory["recent_messages"]) == 2
    print(f"✅ Session memory: {memory['total_turns']} turns")
    
    # Cleanup
    clear_session(session_id)
    
    return True


def test_llm():
    """Test LLM mock."""
    print("\n" + "="*60)
    print("🤖 Testing LLM Mock")
    print("="*60)
    
    from orchestration.mocks import generate_response, summarize_conversation
    
    # Mock inputs
    customer_text = "الشقة دي بكام؟"
    
    emotion = {
        "primary_emotion": "neutral",
        "confidence": 0.8,
        "voice_emotion": "neutral",
        "text_emotion": "neutral",
        "intensity": "medium",
        "scores": {}
    }
    
    emotional_context = {
        "current": emotion,
        "trend": "stable",
        "recommendation": "be_professional",
        "risk_level": "low"
    }
    
    persona = {
        "id": "test",
        "name": "عميل تجريبي",
        "personality_prompt": "test"
    }
    
    memory = {
        "session_id": "test",
        "checkpoints": [],
        "recent_messages": [],
        "total_turns": 0
    }
    
    rag_context = {
        "query": customer_text,
        "documents": [
            {
                "content": "شقة 120 متر في التجمع الخامس بسعر 850,000 جنيه",
                "source": "test.pdf",
                "score": 0.9,
                "metadata": {}
            }
        ],
        "total_found": 1
    }
    
    # Test generate_response
    response = generate_response(
        customer_text=customer_text,
        emotion=emotion,
        emotional_context=emotional_context,
        persona=persona,
        memory=memory,
        rag_context=rag_context
    )
    
    assert isinstance(response, str)
    assert len(response) > 0
    print(f"✅ Generated response: '{response[:50]}...'")
    
    # Test summarize_conversation
    messages = [
        {"speaker": "salesperson", "text": "عايز شقة في التجمع"},
        {"speaker": "vc", "text": "عندنا خيارات كتير"},
    ]
    
    summary = summarize_conversation(messages)
    assert isinstance(summary, str)
    print(f"✅ Summary: '{summary}'")
    
    return True


def test_full_flow():
    """Test full conversation flow with all mocks."""
    print("\n" + "="*60)
    print("🔄 Testing Full Flow (All Mocks Together)")
    print("="*60)
    
    from orchestration.mocks import (
        detect_emotion,
        analyze_emotional_context,
        get_persona,
        retrieve_context,
        store_message,
        get_session_memory,
        generate_response,
        text_to_speech
    )
    
    session_id = "full_flow_test"
    
    # Step 1: Get persona
    persona = get_persona("difficult_customer")
    print(f"1️⃣ Loaded persona: {persona['name']}")
    
    # Step 2: Simulate STT output
    customer_text = "أنا عايز شقة في التجمع الخامس، بس مش عايز حاجة غالية"
    audio = np.random.randn(16000 * 5).astype(np.float32)
    print(f"2️⃣ STT output: '{customer_text}'")
    
    # Step 3: Detect emotion
    emotion = detect_emotion(customer_text, audio)
    print(f"3️⃣ Emotion: {emotion['primary_emotion']}")
    
    # Step 4: Analyze emotional context
    history = []
    emotional_context = analyze_emotional_context(emotion, history)
    print(f"4️⃣ Emotional context: risk={emotional_context['risk_level']}")
    
    # Step 5: Store message
    message = {
        "turn": 1,
        "speaker": "salesperson",
        "text": customer_text,
        "emotion": emotion
    }
    store_message(session_id, message)
    print("5️⃣ Message stored")
    
    # Step 6: Get memory
    memory = get_session_memory(session_id)
    print(f"6️⃣ Memory: {memory['total_turns']} turns")
    
    # Step 7: RAG retrieval
    rag_context = retrieve_context(customer_text, top_k=2)
    print(f"7️⃣ RAG: {len(rag_context['documents'])} documents")
    
    # Step 8: Generate response
    response = generate_response(
        customer_text=customer_text,
        emotion=emotion,
        emotional_context=emotional_context,
        persona=persona,
        memory=memory,
        rag_context=rag_context
    )
    print(f"8️⃣ LLM response: '{response[:50]}...'")
    
    # Step 9: TTS
    audio_output = text_to_speech(response, voice_id=persona["voice_id"])
    print(f"9️⃣ TTS: {len(audio_output)} samples ({len(audio_output)/22050:.2f}s)")
    
    # Store VC response
    vc_message = {
        "turn": 1,
        "speaker": "vc",
        "text": response,
        "emotion": None
    }
    store_message(session_id, vc_message)
    print("🔟 VC message stored")
    
    print("\n✅ Full flow completed successfully!")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("🧪 VCAI Mock Tests")
    print("="*60)
    
    tests = [
        ("TTS", test_tts),
        ("Persona", test_persona),
        ("Emotion", test_emotion),
        ("RAG", test_rag),
        ("Memory", test_memory),
        ("LLM", test_llm),
        ("Full Flow", test_full_flow),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ {name} test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {name}: {status}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Mocks are ready for orchestration development.")
        return 0
    else:
        print("\n⚠️ Some tests failed. Please fix before continuing.")
        return 1


if __name__ == "__main__":
    exit(main())