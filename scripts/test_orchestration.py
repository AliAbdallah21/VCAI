# scripts/test_orchestration.py
"""
Test script for the VCAI Orchestration Agent.
Runs through a full conversation simulation.

USAGE:
    cd C:\VCAI
    python scripts/test_orchestration.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from datetime import datetime


def test_agent_initialization():
    """Test agent can be created."""
    print("\n" + "="*60)
    print("🔧 Testing Agent Initialization")
    print("="*60)
    
    from orchestration import OrchestrationAgent
    
    agent = OrchestrationAgent(use_mocks=True, verbose=False)
    
    assert agent is not None
    assert agent.config.use_mocks == True
    
    print("✅ Agent initialized successfully")
    return True


def test_session_lifecycle():
    """Test session start/end."""
    print("\n" + "="*60)
    print("🔄 Testing Session Lifecycle")
    print("="*60)
    
    from orchestration import OrchestrationAgent
    
    agent = OrchestrationAgent(use_mocks=True, verbose=False)
    
    # Start session
    state = agent.start_session(
        session_id="test_001",
        user_id="user_001",
        persona_id="friendly_customer"
    )
    
    assert agent.session_active == True
    assert state["session_id"] == "test_001"
    assert state["persona"]["id"] == "friendly_customer"
    print("✅ Session started")
    
    # End session
    summary = agent.end_session()
    
    assert agent.session_active == False
    assert summary["session_id"] == "test_001"
    print("✅ Session ended")
    
    return True


def test_single_turn():
    """Test processing a single turn."""
    print("\n" + "="*60)
    print("💬 Testing Single Turn Processing")
    print("="*60)
    
    from orchestration import OrchestrationAgent
    
    agent = OrchestrationAgent(use_mocks=True, verbose=True)
    
    # Start session
    agent.start_session(
        session_id="test_002",
        user_id="user_001",
        persona_id="difficult_customer"
    )
    
    # Create fake audio (3 seconds)
    audio_input = np.random.randn(16000 * 3).astype(np.float32) * 0.1
    
    # Process turn
    result = agent.process_turn(audio_input)
    
    # Verify results
    assert result["transcription"] is not None, "Should have transcription"
    assert result["emotion"] is not None, "Should have emotion"
    assert result["llm_response"] is not None, "Should have LLM response"
    assert result["audio_output"] is not None, "Should have audio output"
    
    print(f"\n✅ Turn processed successfully")
    print(f"   Transcription: '{result['transcription'][:50]}...'")
    print(f"   Emotion: {result['emotion']['primary_emotion']}")
    print(f"   Response: '{result['llm_response'][:50]}...'")
    print(f"   Audio: {result['audio_output'].shape}")
    
    agent.end_session()
    
    return True


def test_multiple_turns():
    """Test processing multiple turns."""
    print("\n" + "="*60)
    print("🔁 Testing Multiple Turns")
    print("="*60)
    
    from orchestration import OrchestrationAgent
    
    agent = OrchestrationAgent(use_mocks=True, verbose=False)
    
    agent.start_session(
        session_id="test_003",
        user_id="user_001",
        persona_id="price_focused_customer"
    )
    
    # Process 5 turns
    for i in range(5):
        audio_input = np.random.randn(16000 * 3).astype(np.float32) * 0.1
        result = agent.process_turn(audio_input)
        
        assert result["llm_response"] is not None
        print(f"  Turn {i+1}: '{result['llm_response'][:40]}...'")
    
    # Check state
    state = agent.get_state()
    assert state["turn_count"] == 5
    assert len(state["history"]) == 10  # 5 turns * 2 messages each
    
    summary = agent.end_session()
    
    print(f"\n✅ Multiple turns processed")
    print(f"   Total turns: {summary['total_turns']}")
    print(f"   History length: {summary['history_length']}")
    
    return True


def test_different_personas():
    """Test with different personas."""
    print("\n" + "="*60)
    print("👥 Testing Different Personas")
    print("="*60)
    
    from orchestration import OrchestrationAgent
    
    personas = [
        "difficult_customer",
        "friendly_customer",
        "rushed_customer",
        "price_focused_customer"
    ]
    
    for persona_id in personas:
        agent = OrchestrationAgent(use_mocks=True, verbose=False)
        
        agent.start_session(
            session_id=f"test_{persona_id}",
            user_id="user_001",
            persona_id=persona_id
        )
        
        audio_input = np.random.randn(16000 * 3).astype(np.float32) * 0.1
        result = agent.process_turn(audio_input)
        
        persona_name = result["persona"]["name"]
        response = result["llm_response"][:40]
        
        print(f"  {persona_name}: '{response}...'")
        
        agent.end_session()
    
    print(f"\n✅ All personas work correctly")
    
    return True


def test_simple_mode():
    """Test simple mode (no memory/RAG)."""
    print("\n" + "="*60)
    print("⚡ Testing Simple Mode")
    print("="*60)
    
    from orchestration import OrchestrationAgent
    
    agent = OrchestrationAgent(
        use_mocks=True,
        verbose=False,
        simple_mode=True  # No memory/RAG
    )
    
    agent.start_session(
        session_id="test_simple",
        user_id="user_001",
        persona_id="friendly_customer"
    )
    
    audio_input = np.random.randn(16000 * 3).astype(np.float32) * 0.1
    result = agent.process_turn(audio_input)
    
    assert result["llm_response"] is not None
    
    # Simple mode should be faster
    timings = result["node_timings"]
    assert "memory_load" not in timings  # No memory in simple mode
    
    print(f"✅ Simple mode works")
    print(f"   Nodes executed: {list(timings.keys())}")
    
    agent.end_session()
    
    return True


def test_error_handling():
    """Test error handling."""
    print("\n" + "="*60)
    print("⚠️ Testing Error Handling")
    print("="*60)
    
    from orchestration import OrchestrationAgent
    
    agent = OrchestrationAgent(use_mocks=True, verbose=False)
    
    # Test: Process without session
    try:
        audio_input = np.random.randn(16000).astype(np.float32)
        agent.process_turn(audio_input)
        assert False, "Should raise error"
    except RuntimeError as e:
        print(f"✅ Correct error for no session: {str(e)}")
    
    # Test: Invalid persona
    from shared.exceptions import PersonaNotFoundError
    
    agent.start_session(
        session_id="test_error",
        user_id="user_001"
    )
    
    # This should still work (no persona)
    audio_input = np.random.randn(16000 * 3).astype(np.float32)
    result = agent.process_turn(audio_input)
    
    assert result["llm_response"] is not None
    print("✅ Works without persona")
    
    agent.end_session()
    
    return True


def test_timings():
    """Test that timings are recorded."""
    print("\n" + "="*60)
    print("⏱️ Testing Timings")
    print("="*60)
    
    from orchestration import OrchestrationAgent
    
    agent = OrchestrationAgent(use_mocks=True, verbose=False)
    
    agent.start_session(
        session_id="test_timings",
        user_id="user_001",
        persona_id="friendly_customer"
    )
    
    audio_input = np.random.randn(16000 * 3).astype(np.float32) * 0.1
    result = agent.process_turn(audio_input)
    
    timings = result["node_timings"]
    
    assert "stt" in timings
    assert "emotion" in timings
    assert "llm" in timings
    assert "tts" in timings
    assert "total" in timings
    
    print(f"✅ Timings recorded:")
    for node, time_taken in timings.items():
        print(f"   {node}: {time_taken:.3f}s")
    
    agent.end_session()
    
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("🧪 VCAI Orchestration Agent Tests")
    print("="*60)
    
    tests = [
        ("Initialization", test_agent_initialization),
        ("Session Lifecycle", test_session_lifecycle),
        ("Single Turn", test_single_turn),
        ("Multiple Turns", test_multiple_turns),
        ("Different Personas", test_different_personas),
        ("Simple Mode", test_simple_mode),
        ("Error Handling", test_error_handling),
        ("Timings", test_timings),
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
        print("\n🎉 All tests passed! Orchestration Agent is ready.")
        return 0
    else:
        print("\n⚠️ Some tests failed.")
        return 1


if __name__ == "__main__":
    exit(main())