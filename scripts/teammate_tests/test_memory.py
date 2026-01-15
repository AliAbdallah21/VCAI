# scripts/teammate_tests/test_memory.py
"""
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

# ══════════════════════════════════════════════════════════════════════════════
# TEST CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

TEST_SESSION_ID = "test_session_12345"

SAMPLE_MESSAGE = {
    "id": "msg_001",
    "turn": 1,
    "speaker": "salesperson",
    "text": "مرحبا، أنا عايز أشوف شقة",
    "emotion": None,
    "audio_path": None,
    "timestamp": datetime.now()
}

REQUIRED_MEMORY_KEYS = ["session_id", "checkpoints", "recent_messages", "total_turns"]


# ══════════════════════════════════════════════════════════════════════════════
# TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_store_message_exists():
    """Test 1: Check store_message exists"""
    print("\n[Test 1] Checking store_message function...")
    
    try:
        from memory.memory_agent import store_message
        print("   ✅ Function imported")
        return True, store_message
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False, None


def test_get_recent_messages_exists():
    """Test 2: Check get_recent_messages exists"""
    print("\n[Test 2] Checking get_recent_messages function...")
    
    try:
        from memory.memory_agent import get_recent_messages
        print("   ✅ Function imported")
        return True, get_recent_messages
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False, None


def test_get_session_memory_exists():
    """Test 3: Check get_session_memory exists"""
    print("\n[Test 3] Checking get_session_memory function...")
    
    try:
        from memory.memory_agent import get_session_memory
        print("   ✅ Function imported")
        return True, get_session_memory
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False, None


def test_store_message(func):
    """Test 4: Test store_message"""
    print("\n[Test 4] Testing store_message...")
    
    try:
        result = func(TEST_SESSION_ID, SAMPLE_MESSAGE)
        
        if result is True or result:
            print("   ✅ store_message returned True/truthy")
            return True
        else:
            print(f"   ❌ store_message returned: {result}")
            return False
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False


def test_get_recent_messages(func):
    """Test 5: Test get_recent_messages"""
    print("\n[Test 5] Testing get_recent_messages...")
    
    try:
        result = func(TEST_SESSION_ID, last_n=5)
        
        if not isinstance(result, list):
            print(f"   ❌ Should return list, got {type(result)}")
            return False
        
        print(f"   ✅ Returned list with {len(result)} messages")
        return True
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False


def test_get_session_memory(func):
    """Test 6: Test get_session_memory"""
    print("\n[Test 6] Testing get_session_memory...")
    
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
        return True
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False


def test_session_memory_structure(func):
    """Test 7: Check session memory structure"""
    print("\n[Test 7] Checking session memory structure...")
    
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
        
        print("   ✅ Structure is correct")
        return True
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Memory Agent Validation Tests")
    print("Person D Implementation")
    print("=" * 60)
    
    results = []
    
    # Import tests
    passed, store_func = test_store_message_exists()
    results.append(("store_message exists", passed))
    
    passed, get_recent_func = test_get_recent_messages_exists()
    results.append(("get_recent_messages exists", passed))
    
    passed, get_memory_func = test_get_session_memory_exists()
    results.append(("get_session_memory exists", passed))
    
    # Function tests
    if store_func:
        results.append(("store_message works", test_store_message(store_func)))
    
    if get_recent_func:
        results.append(("get_recent_messages works", test_get_recent_messages(get_recent_func)))
    
    if get_memory_func:
        results.append(("get_session_memory works", test_get_session_memory(get_memory_func)))
        results.append(("Memory structure", test_session_memory_structure(get_memory_func)))
    
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
