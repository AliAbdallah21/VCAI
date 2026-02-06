# memory/agent.py
# Person D: Memory Agent (Public Interface)
# IMPORTANT: This file should only expose the required functions.
# Actual DB logic lives in memory/store.py

from _future_ import annotations
from shared.types import Message, MemoryCheckpoint, SessionMemory

# Import internal storage functions (you will implement these in store.py next)
from memory.store import (
    store_message_db,
    get_recent_messages_db,
    store_checkpoint_db,
    get_checkpoints_db,
    get_total_turns_db,
)


def store_message(session_id: str, message: Message) -> bool:
    """
    Store a conversation message.
    OUTPUT:
        bool: True if successful
    """
    return store_message_db(session_id, message)


def get_recent_messages(session_id: str, last_n: int = 10) -> list[Message]:
    """
    Get recent messages from a session (oldest first).
    """
    return get_recent_messages_db(session_id, last_n=last_n)


def store_checkpoint(session_id: str, checkpoint: MemoryCheckpoint) -> bool:
    """
    Store a memory checkpoint (summary).
    """
    return store_checkpoint_db(session_id, checkpoint)


def get_checkpoints(session_id: str) -> list[MemoryCheckpoint]:
    """
    Get all checkpoints for a session (oldest first).
    """
    return get_checkpoints_db(session_id)


def get_session_memory(session_id: str) -> SessionMemory:
    """
    Get full session memory (checkpoints + recent messages).
    OUTPUT:
        SessionMemory: {
            "session_id": str,
            "checkpoints": list[MemoryCheckpoint],
            "recent_messages": list[Message],
            "total_turns": int
        }
    """
    checkpoints = get_checkpoints(session_id)
    recent_messages = get_recent_messages(session_id, last_n=10)
    total_turns = get_total_turns_db(session_id)
    return {
        "session_id": session_id,
        "checkpoints": checkpoints,
        "recent_messages": recent_messages,
        "total_turns": total_turns,
    }