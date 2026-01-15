# orchestration/mocks/mock_memory.py
"""
Mock Memory Agent functions.
Replace with real imports from memory.agent when Person D completes their work.

USAGE:
    # Now (development):
    from orchestration.mocks.mock_memory import (
        store_message, get_recent_messages,
        store_checkpoint, get_checkpoints,
        get_session_memory
    )
    
    # Later (integration):
    from memory.agent import (
        store_message, get_recent_messages,
        store_checkpoint, get_checkpoints,
        get_session_memory
    )
"""

from datetime import datetime
from typing import Optional
import uuid

from shared.types import Message, MemoryCheckpoint, SessionMemory
from shared.constants import RECENT_MESSAGES_COUNT


# In-memory storage (simulates database)
_messages_store: dict[str, list[Message]] = {}
_checkpoints_store: dict[str, list[MemoryCheckpoint]] = {}


def _get_timestamp() -> datetime:
    """Get current timestamp."""
    return datetime.now()


def store_message(session_id: str, message: Message) -> bool:
    """
    Mock store_message - stores message in memory.
    
    INPUT:
        session_id: str - Session identifier
        message: Message - Message to store
    
    OUTPUT:
        bool - True if successful
    """
    if session_id not in _messages_store:
        _messages_store[session_id] = []
    
    # Ensure message has required fields
    if "id" not in message or not message["id"]:
        message["id"] = str(uuid.uuid4())
    
    if "timestamp" not in message or not message["timestamp"]:
        message["timestamp"] = _get_timestamp()
    
    _messages_store[session_id].append(message)
    
    print(f"[MOCK MEMORY] Stored message #{message.get('turn', '?')} from {message['speaker']}")
    print(f"[MOCK MEMORY] Session {session_id} now has {len(_messages_store[session_id])} messages")
    
    return True


def get_recent_messages(session_id: str, last_n: int = 10) -> list[Message]:
    """
    Mock get_recent_messages - retrieves recent messages.
    
    INPUT:
        session_id: str - Session identifier
        last_n: int - Number of recent messages to retrieve
    
    OUTPUT:
        list[Message] - Recent messages (oldest first)
    """
    if session_id not in _messages_store:
        print(f"[MOCK MEMORY] No messages found for session {session_id}")
        return []
    
    messages = _messages_store[session_id]
    recent = messages[-last_n:] if len(messages) > last_n else messages
    
    print(f"[MOCK MEMORY] Retrieved {len(recent)} recent messages for session {session_id}")
    
    return recent


def get_all_messages(session_id: str) -> list[Message]:
    """
    Mock function to get all messages for a session.
    
    INPUT:
        session_id: str - Session identifier
    
    OUTPUT:
        list[Message] - All messages
    """
    return _messages_store.get(session_id, [])


def store_checkpoint(session_id: str, checkpoint: MemoryCheckpoint) -> bool:
    """
    Mock store_checkpoint - stores checkpoint in memory.
    
    INPUT:
        session_id: str - Session identifier
        checkpoint: MemoryCheckpoint - Checkpoint to store
    
    OUTPUT:
        bool - True if successful
    """
    if session_id not in _checkpoints_store:
        _checkpoints_store[session_id] = []
    
    # Ensure checkpoint has required fields
    if "id" not in checkpoint or not checkpoint["id"]:
        checkpoint["id"] = str(uuid.uuid4())
    
    if "session_id" not in checkpoint:
        checkpoint["session_id"] = session_id
    
    if "created_at" not in checkpoint or not checkpoint["created_at"]:
        checkpoint["created_at"] = _get_timestamp()
    
    _checkpoints_store[session_id].append(checkpoint)
    
    turn_range = checkpoint.get("turn_range", (0, 0))
    print(f"[MOCK MEMORY] Stored checkpoint for turns {turn_range[0]}-{turn_range[1]}")
    print(f"[MOCK MEMORY] Session {session_id} now has {len(_checkpoints_store[session_id])} checkpoints")
    
    return True


def get_checkpoints(session_id: str) -> list[MemoryCheckpoint]:
    """
    Mock get_checkpoints - retrieves all checkpoints.
    
    INPUT:
        session_id: str - Session identifier
    
    OUTPUT:
        list[MemoryCheckpoint] - All checkpoints (oldest first)
    """
    if session_id not in _checkpoints_store:
        print(f"[MOCK MEMORY] No checkpoints found for session {session_id}")
        return []
    
    checkpoints = _checkpoints_store[session_id]
    
    print(f"[MOCK MEMORY] Retrieved {len(checkpoints)} checkpoints for session {session_id}")
    
    return checkpoints


def get_session_memory(session_id: str) -> SessionMemory:
    """
    Mock get_session_memory - retrieves full session memory.
    
    INPUT:
        session_id: str - Session identifier
    
    OUTPUT:
        SessionMemory - Full memory with checkpoints and recent messages
    """
    checkpoints = get_checkpoints(session_id)
    recent_messages = get_recent_messages(session_id, RECENT_MESSAGES_COUNT)
    all_messages = get_all_messages(session_id)
    
    memory: SessionMemory = {
        "session_id": session_id,
        "checkpoints": checkpoints,
        "recent_messages": recent_messages,
        "total_turns": len(all_messages)
    }
    
    print(f"[MOCK MEMORY] Session memory: {len(checkpoints)} checkpoints, {len(recent_messages)} recent messages, {len(all_messages)} total turns")
    
    return memory


def clear_session(session_id: str) -> bool:
    """
    Mock function to clear all data for a session.
    
    INPUT:
        session_id: str - Session identifier
    
    OUTPUT:
        bool - True if successful
    """
    if session_id in _messages_store:
        del _messages_store[session_id]
    
    if session_id in _checkpoints_store:
        del _checkpoints_store[session_id]
    
    print(f"[MOCK MEMORY] Cleared all data for session {session_id}")
    
    return True


def clear_all() -> bool:
    """
    Mock function to clear all stored data.
    Use for testing purposes only.
    """
    _messages_store.clear()
    _checkpoints_store.clear()
    print("[MOCK MEMORY] Cleared all data")
    return True


def get_stats() -> dict:
    """
    Mock function to get memory statistics.
    """
    total_sessions = len(_messages_store)
    total_messages = sum(len(msgs) for msgs in _messages_store.values())
    total_checkpoints = sum(len(cps) for cps in _checkpoints_store.values())
    
    return {
        "total_sessions": total_sessions,
        "total_messages": total_messages,
        "total_checkpoints": total_checkpoints
    }


# For testing
if __name__ == "__main__":
    # Clear any existing data
    clear_all()
    
    # Test session ID
    session_id = "test_session_001"
    
    # Store some messages
    print("\n" + "="*60)
    print("Storing messages...")
    print("="*60)
    
    messages = [
        {"turn": 1, "speaker": "salesperson", "text": "مرحبا، أنا عايز أشوف شقق", "emotion": None},
        {"turn": 1, "speaker": "vc", "text": "أهلاً بيك، إيه المنطقة اللي بتفكر فيها؟", "emotion": None},
        {"turn": 2, "speaker": "salesperson", "text": "التجمع الخامس أو الشيخ زايد", "emotion": None},
        {"turn": 2, "speaker": "vc", "text": "عندنا خيارات حلوة في المنطقتين", "emotion": None},
        {"turn": 3, "speaker": "salesperson", "text": "السعر إيه تقريباً؟", "emotion": None},
        {"turn": 3, "speaker": "vc", "text": "الأسعار بتبدأ من 550 ألف", "emotion": None},
    ]
    
    for msg in messages:
        store_message(session_id, msg)
    
    # Get recent messages
    print("\n" + "="*60)
    print("Getting recent messages...")
    print("="*60)
    
    recent = get_recent_messages(session_id, last_n=4)
    for msg in recent:
        print(f"  Turn {msg['turn']} - {msg['speaker']}: {msg['text'][:50]}...")
    
    # Store a checkpoint
    print("\n" + "="*60)
    print("Storing checkpoint...")
    print("="*60)
    
    checkpoint = {
        "turn_range": (1, 3),
        "summary": "العميل بيدور على شقة في التجمع الخامس أو الشيخ زايد. سأل عن الأسعار.",
        "key_points": ["التجمع الخامس", "الشيخ زايد", "الأسعار بتبدأ من 550 ألف"],
        "customer_preferences": {"areas": ["التجمع الخامس", "الشيخ زايد"]},
        "objections_raised": []
    }
    
    store_checkpoint(session_id, checkpoint)
    
    # Get full session memory
    print("\n" + "="*60)
    print("Getting full session memory...")
    print("="*60)
    
    memory = get_session_memory(session_id)
    print(f"Total turns: {memory['total_turns']}")
    print(f"Checkpoints: {len(memory['checkpoints'])}")
    print(f"Recent messages: {len(memory['recent_messages'])}")
    
    # Print stats
    print("\n" + "="*60)
    print("Memory stats:")
    print("="*60)
    print(get_stats())