# scripts/teammate_tests/test_memory.py
r"""
Test script for Person D: Memory Agent
Run this to validate your implementation before pushing.

Usage:
    cd C:\VCAI
    python scripts/teammate_tests/test_memory.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datetime import datetime
from uuid import UUID, uuid4

# ══════════════════════════════════════════════════════════════════════════════
# TEST CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Valid UUID format for testing
TEST_USER_ID = "00000000-0000-0000-0000-000000000001"
TEST_PERSONA_ID = "test_persona"
TEST_SESSION_ID = "12345678-1234-5678-1234-567812345678"

SAMPLE_MESSAGE = {
    "id": str(uuid4()),
    "turn": 1,
    "speaker": "salesperson",
    "text": "مرحبا، أنا عايز أشوف شقة",
    "emotion": None,
    "audio_path": None,
    "timestamp": datetime.now()
}

SAMPLE_MESSAGE_2 = {
    "id": str(uuid4()),
    "turn": 2,
    "speaker": "vc",
    "text": "أهلا، عايز شقة في منطقة إيه؟",
    "emotion": None,
    "audio_path": None,
    "timestamp": datetime.now()
}

SAMPLE_CHECKPOINT = {
    "id": str(uuid4()),
    "session_id": TEST_SESSION_ID,
    "turn_range": (1, 2),
    "summary": "بداية المحادثة: العميل سأل عن شقة والموظف رد بسؤال عن المنطقة",
    "key_points": ["سؤال عن شقة", "سؤال عن المنطقة"],
    "customer_preferences": {},
    "objections_raised": [],
    "created_at": datetime.now()
}

REQUIRED_MEMORY_KEYS = ["session_id", "checkpoints", "recent_messages", "total_turns"]


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE SETUP
# ══════════════════════════════════════════════════════════════════════════════

def setup_test_database():
    """
    Create test user, persona, and session in the database.
    Returns True if setup successful, False otherwise.
    """
    print("\n[Setup] Preparing test data in database...")
    
    try:
        from backend.database import get_db_context
        from backend.models.session import Session, Message, Checkpoint
        from backend.models.user import User
        from backend.models.persona import Persona
        
        test_user_id = UUID(TEST_USER_ID)
        test_session_id = UUID(TEST_SESSION_ID)
        
        with get_db_context() as db:
            # ─────────────────────────────────────────────────────────────
            # 1. Create or verify test user
            # ─────────────────────────────────────────────────────────────
            user = db.query(User).filter(User.id == test_user_id).first()
            if not user:
                user = User(
                    id=test_user_id,
                    email="memory_test@vcai.test",
                    password_hash="not_a_real_hash_just_for_testing",
                    full_name="Memory Test User",
                    company="VCAI Test",
                    role="salesperson",
                    experience_level="beginner",
                    is_active=True
                )
                db.add(user)
                db.flush()
                print("   ✅ Created test user")
            else:
                print("   ✅ Test user exists")
            
            # ─────────────────────────────────────────────────────────────
            # 2. Create or verify test persona
            # ─────────────────────────────────────────────────────────────
            persona = db.query(Persona).filter(Persona.id == TEST_PERSONA_ID).first()
            if not persona:
                persona = Persona(
                    id=TEST_PERSONA_ID,
                    name_ar="عميل اختبار",
                    name_en="Test Customer",
                    description_ar="شخصية اختبارية لاختبار نظام الذاكرة",
                    description_en="Test persona for memory agent tests",
                    personality_prompt="أنت عميل مصري للاختبار. رد بشكل طبيعي.",
                    difficulty="easy",
                    patience_level=50,
                    emotion_sensitivity=50,
                    traits=["test", "friendly"],
                    is_active=True
                )
                db.add(persona)
                db.flush()
                print("   ✅ Created test persona")
            else:
                print("   ✅ Test persona exists")
            
            # ─────────────────────────────────────────────────────────────
            # 3. Create or reset test session
            # ─────────────────────────────────────────────────────────────
            session = db.query(Session).filter(Session.id == test_session_id).first()
            if session:
                # Delete existing messages and checkpoints for clean test
                db.query(Checkpoint).filter(Checkpoint.session_id == test_session_id).delete()
                db.query(Message).filter(Message.session_id == test_session_id).delete()
                session.turn_count = 0
                db.flush()
                print("   ✅ Reset existing test session")
            else:
                session = Session(
                    id=test_session_id,
                    user_id=test_user_id,
                    persona_id=TEST_PERSONA_ID,
                    difficulty="easy",
                    status="active",
                    turn_count=0
                )
                db.add(session)
                print("   ✅ Created test session")
        
        print("   ✅ Database setup complete!")
        return True
        
    except Exception as e:
        print(f"   ❌ Database setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def cleanup_test_database():
    """
    Optional: Clean up test data after tests.
    """
    print("\n[Cleanup] Removing test data...")
    
    try:
        from backend.database import get_db_context
        from backend.models.session import Session, Message, Checkpoint
        
        test_session_id = UUID(TEST_SESSION_ID)
        
        with get_db_context() as db:
            # Delete checkpoints and messages first (foreign key constraints)
            db.query(Checkpoint).filter(Checkpoint.session_id == test_session_id).delete()
            db.query(Message).filter(Message.session_id == test_session_id).delete()
            # Note: We keep the session, user, and persona for future tests
        
        print("   ✅ Cleanup complete!")
        return True
        
    except Exception as e:
        print(f"   ❌ Cleanup failed: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# IMPORT TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_store_message_exists():
    """Test 1: Check store_message exists"""
    print("\n[Test 1] Checking store_message function...")
    
    try:
        from memory.agent import store_message
        print("   ✅ Function imported")
        return True, store_message
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False, None


def test_get_recent_messages_exists():
    """Test 2: Check get_recent_messages exists"""
    print("\n[Test 2] Checking get_recent_messages function...")
    
    try:
        from memory.agent import get_recent_messages
        print("   ✅ Function imported")
        return True, get_recent_messages
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False, None


def test_get_session_memory_exists():
    """Test 3: Check get_session_memory exists"""
    print("\n[Test 3] Checking get_session_memory function...")
    
    try:
        from memory.agent import get_session_memory
        print("   ✅ Function imported")
        return True, get_session_memory
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False, None


def test_store_checkpoint_exists():
    """Test 4: Check store_checkpoint exists"""
    print("\n[Test 4] Checking store_checkpoint function...")
    
    try:
        from memory.agent import store_checkpoint
        print("   ✅ Function imported")
        return True, store_checkpoint
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False, None


def test_get_checkpoints_exists():
    """Test 5: Check get_checkpoints exists"""
    print("\n[Test 5] Checking get_checkpoints function...")
    
    try:
        from memory.agent import get_checkpoints
        print("   ✅ Function imported")
        return True, get_checkpoints
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False, None


# ══════════════════════════════════════════════════════════════════════════════
# FUNCTIONALITY TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_store_message(func):
    """Test 6: Test store_message"""
    print("\n[Test 6] Testing store_message...")
    
    try:
        result = func(TEST_SESSION_ID, SAMPLE_MESSAGE)
        
        if result is True:
            print("   ✅ store_message returned True")
            return True
        else:
            print(f"   ❌ store_message returned: {result}")
            return False
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_store_second_message(func):
    """Test 7: Test storing a second message"""
    print("\n[Test 7] Testing store_message (second message)...")
    
    try:
        result = func(TEST_SESSION_ID, SAMPLE_MESSAGE_2)
        
        if result is True:
            print("   ✅ Second message stored")
            return True
        else:
            print(f"   ❌ store_message returned: {result}")
            return False
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False


def test_get_recent_messages(func):
    """Test 8: Test get_recent_messages"""
    print("\n[Test 8] Testing get_recent_messages...")
    
    try:
        result = func(TEST_SESSION_ID, last_n=5)
        
        if not isinstance(result, list):
            print(f"   ❌ Should return list, got {type(result)}")
            return False
        
        print(f"   ✅ Returned list with {len(result)} messages")
        
        # Verify we got the messages we stored
        if len(result) >= 2:
            print(f"      Message 1: turn={result[0].get('turn')} speaker={result[0].get('speaker')}")
            print(f"      Message 2: turn={result[1].get('turn')} speaker={result[1].get('speaker')}")
        
        return True
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_store_checkpoint(func):
    """Test 9: Test store_checkpoint"""
    print("\n[Test 9] Testing store_checkpoint...")
    
    try:
        result = func(TEST_SESSION_ID, SAMPLE_CHECKPOINT)
        
        if result is True:
            print("   ✅ store_checkpoint returned True")
            return True
        else:
            print(f"   ❌ store_checkpoint returned: {result}")
            return False
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_get_checkpoints(func):
    """Test 10: Test get_checkpoints"""
    print("\n[Test 10] Testing get_checkpoints...")
    
    try:
        result = func(TEST_SESSION_ID)
        
        if not isinstance(result, list):
            print(f"   ❌ Should return list, got {type(result)}")
            return False
        
        print(f"   ✅ Returned list with {len(result)} checkpoints")
        
        if len(result) >= 1:
            cp = result[0]
            print(f"      Checkpoint: turns={cp.get('turn_range')} summary={cp.get('summary')[:50]}...")
        
        return True
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_get_session_memory(func):
    """Test 11: Test get_session_memory"""
    print("\n[Test 11] Testing get_session_memory...")
    
    try:
        result = func(TEST_SESSION_ID)
        
        if not isinstance(result, dict):
            print(f"   ❌ Should return dict, got {type(result)}")
            return False
        
        missing = [k for k in REQUIRED_MEMORY_KEYS if k not in result]
        if missing:
            print(f"   ❌ Missing keys: {missing}")
            return False
        
        print(f"   ✅ All required keys present")
        print(f"      session_id: {result.get('session_id')}")
        print(f"      total_turns: {result.get('total_turns')}")
        print(f"      recent_messages: {len(result.get('recent_messages', []))} messages")
        print(f"      checkpoints: {len(result.get('checkpoints', []))} checkpoints")
        
        return True
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_session_memory_structure(func):
    """Test 12: Check session memory structure"""
    print("\n[Test 12] Checking session memory structure...")
    
    try:
        result = func(TEST_SESSION_ID)
        
        # Check checkpoints is list
        if not isinstance(result.get("checkpoints"), list):
            print("   ❌ checkpoints should be list")
            return False
        
        # Check recent_messages is list
        if not isinstance(result.get("recent_messages"), list):
            print("   ❌ recent_messages should be list")
            return False
        
        # Check total_turns is int
        if not isinstance(result.get("total_turns"), int):
            print("   ❌ total_turns should be int")
            return False
        
        # Check session_id is string
        if not isinstance(result.get("session_id"), str):
            print("   ❌ session_id should be string")
            return False
        
        print("   ✅ Structure is correct")
        return True
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False


def test_memory_data_integrity(get_memory_func):
    """Test 13: Verify data integrity - messages and checkpoints are retrievable"""
    print("\n[Test 13] Testing data integrity...")
    
    try:
        memory = get_memory_func(TEST_SESSION_ID)
        
        # Should have at least 2 messages
        if len(memory.get("recent_messages", [])) < 2:
            print(f"   ❌ Expected at least 2 messages, got {len(memory.get('recent_messages', []))}")
            return False
        
        # Should have at least 1 checkpoint
        if len(memory.get("checkpoints", [])) < 1:
            print(f"   ❌ Expected at least 1 checkpoint, got {len(memory.get('checkpoints', []))}")
            return False
        
        # total_turns should be at least 2
        if memory.get("total_turns", 0) < 2:
            print(f"   ❌ Expected total_turns >= 2, got {memory.get('total_turns')}")
            return False
        
        print("   ✅ Data integrity verified")
        print(f"      Messages: {len(memory['recent_messages'])}")
        print(f"      Checkpoints: {len(memory['checkpoints'])}")
        print(f"      Total turns: {memory['total_turns']}")
        
        return True
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Memory Agent Validation Tests (Full DB)")
    print("Person D Implementation")
    print("=" * 60)
    
    results = []
    
    # ─────────────────────────────────────────────────────────────────────────
    # Database Setup
    # ─────────────────────────────────────────────────────────────────────────
    if not setup_test_database():
        print("\n❌ Database setup failed. Cannot continue tests.")
        print("   Make sure your database is running and configured correctly.")
        return
    
    # ─────────────────────────────────────────────────────────────────────────
    # Import Tests
    # ─────────────────────────────────────────────────────────────────────────
    passed, store_func = test_store_message_exists()
    results.append(("store_message exists", passed))
    
    passed, get_recent_func = test_get_recent_messages_exists()
    results.append(("get_recent_messages exists", passed))
    
    passed, get_memory_func = test_get_session_memory_exists()
    results.append(("get_session_memory exists", passed))
    
    passed, store_cp_func = test_store_checkpoint_exists()
    results.append(("store_checkpoint exists", passed))
    
    passed, get_cp_func = test_get_checkpoints_exists()
    results.append(("get_checkpoints exists", passed))
    
    # ─────────────────────────────────────────────────────────────────────────
    # Functionality Tests
    # ─────────────────────────────────────────────────────────────────────────
    if store_func:
        results.append(("store_message works", test_store_message(store_func)))
        results.append(("store_message (2nd)", test_store_second_message(store_func)))
    
    if get_recent_func:
        results.append(("get_recent_messages works", test_get_recent_messages(get_recent_func)))
    
    if store_cp_func:
        results.append(("store_checkpoint works", test_store_checkpoint(store_cp_func)))
    
    if get_cp_func:
        results.append(("get_checkpoints works", test_get_checkpoints(get_cp_func)))
    
    if get_memory_func:
        results.append(("get_session_memory works", test_get_session_memory(get_memory_func)))
        results.append(("Memory structure", test_session_memory_structure(get_memory_func)))
        results.append(("Data integrity", test_memory_data_integrity(get_memory_func)))
    
    # ─────────────────────────────────────────────────────────────────────────
    # Cleanup (optional - comment out to keep test data)
    # ─────────────────────────────────────────────────────────────────────────
    # cleanup_test_database()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────
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
        print("\nYour Memory Agent is fully functional with the real database!")
    else:
        print(f"\n⚠️ {total_count - passed_count} test(s) failed.")


if __name__ == "__main__":
    main()