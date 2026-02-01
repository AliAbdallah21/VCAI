# scripts/debug_orchestration.py
"""
Debug script to test the orchestration agent step by step.
Run: python scripts/debug_orchestration.py
"""

import sys
sys.path.insert(0, r'C:\VCAI')

import numpy as np

print("=" * 60)
print("VCAI Orchestration Debug Script")
print("=" * 60)

# ══════════════════════════════════════════════════════════════════════════════
# TEST 1: Check imports
# ══════════════════════════════════════════════════════════════════════════════
print("\n[TEST 1] Checking imports...")

try:
    from orchestration.agent import OrchestrationAgent
    print("   ✅ OrchestrationAgent imported")
except ImportError as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# TEST 2: Check start_session signature
# ══════════════════════════════════════════════════════════════════════════════
print("\n[TEST 2] Checking start_session signature...")

import inspect
sig = inspect.signature(OrchestrationAgent.start_session)
params = list(sig.parameters.keys())
print(f"   Parameters: {params}")

if 'persona_dict' in params:
    print("   ✅ persona_dict parameter EXISTS")
else:
    print("   ❌ persona_dict parameter MISSING!")
    print("\n   The file needs to be updated. Current signature:")
    print(f"   {sig}")
    
    # Show the file location
    import orchestration.agent as oa
    print(f"\n   File location: {oa.__file__}")
    
    # Show the actual function definition
    print("\n   Actual start_session code (first 20 lines):")
    source_lines = inspect.getsourcelines(OrchestrationAgent.start_session)[0][:20]
    for i, line in enumerate(source_lines):
        print(f"   {i+1}: {line.rstrip()}")

# ══════════════════════════════════════════════════════════════════════════════
# TEST 3: Create agent
# ══════════════════════════════════════════════════════════════════════════════
print("\n[TEST 3] Creating OrchestrationAgent...")

try:
    agent = OrchestrationAgent(use_mocks=False, verbose=True)
    print("   ✅ Agent created")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# TEST 4: Start session WITHOUT persona_dict
# ══════════════════════════════════════════════════════════════════════════════
print("\n[TEST 4] Starting session WITHOUT persona_dict...")

try:
    state = agent.start_session(
        session_id="debug_test_session",
        user_id="debug_user",
        persona_id="test_persona"
    )
    print(f"   ✅ Session started")
    print(f"   Session ID: {state.get('session_id')}")
    print(f"   Turn count: {state.get('turn_count')}")
    print(f"   Session active: {agent.session_active}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()

# ══════════════════════════════════════════════════════════════════════════════
# TEST 5: Start session WITH persona_dict (if supported)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[TEST 5] Starting session WITH persona_dict...")

# Reset agent
agent2 = OrchestrationAgent(use_mocks=False, verbose=False)

persona_dict = {
    "id": "test_persona",
    "name": "عميل اختبار",
    "name_en": "Test Customer",
    "personality_prompt": "أنت عميل مصري للاختبار",
    "difficulty": "easy",
    "traits": ["test"],
    "default_emotion": "neutral"
}

try:
    if 'persona_dict' in params:
        state = agent2.start_session(
            session_id="debug_test_session_2",
            user_id="debug_user",
            persona_id="test_persona",
            persona_dict=persona_dict
        )
        print(f"   ✅ Session started with persona_dict")
    else:
        print("   ⚠️ Skipped - persona_dict not supported")
        # Try alternative: start session then set persona
        state = agent2.start_session(
            session_id="debug_test_session_2",
            user_id="debug_user",
            persona_id="test_persona"
        )
        if hasattr(agent2, 'set_persona'):
            agent2.set_persona(persona_dict)
            print("   ✅ Session started + persona set manually")
        else:
            agent2.state["persona"] = persona_dict
            print("   ✅ Session started + persona set directly on state")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()

# ══════════════════════════════════════════════════════════════════════════════
# TEST 6: Process a turn
# ══════════════════════════════════════════════════════════════════════════════
print("\n[TEST 6] Processing a turn...")

# Create fake audio (3 seconds)
audio_input = np.random.randn(16000 * 3).astype(np.float32) * 0.1

try:
    if agent2.session_active:
        print("   Processing audio through LangGraph...")
        result = agent2.process_turn(audio_input)
        
        print(f"   ✅ Turn processed")
        print(f"   Transcription: {result.get('transcription', 'N/A')[:50]}...")
        print(f"   Response: {result.get('llm_response', 'N/A')[:50]}...")
        print(f"   Turn count: {result.get('turn_count')}")
    else:
        print("   ⚠️ Skipped - no active session")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()

# ══════════════════════════════════════════════════════════════════════════════
# TEST 7: End session
# ══════════════════════════════════════════════════════════════════════════════
print("\n[TEST 7] Ending session...")

try:
    if agent2.session_active:
        summary = agent2.end_session()
        print(f"   ✅ Session ended")
        print(f"   Summary: {summary}")
    else:
        print("   ⚠️ Skipped - no active session")
except Exception as e:
    print(f"   ❌ Failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DEBUG SUMMARY")
print("=" * 60)

print(f"\nFile location: {oa.__file__}")
print(f"persona_dict in signature: {'persona_dict' in params}")
print(f"\nIf persona_dict is MISSING, you need to update the file!")
print("\nTo fix, add this parameter to start_session():")
print("   persona_dict: dict = None")
print("\nAnd add this logic at the start of start_session():")
print("   if persona_dict:")
print("       persona = persona_dict")
print("   elif persona_id:")
print("       persona = self._load_persona(persona_id)")