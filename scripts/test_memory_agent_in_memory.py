from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import memory.agent as agent
from shared.types import Message, MemoryCheckpoint


# -----------------------------------------------------------------------------
# In-memory fake storage
# -----------------------------------------------------------------------------

_MESSAGES: Dict[str, List[Message]] = {}
_CHECKPOINTS: Dict[str, List[MemoryCheckpoint]] = {}
_TOTAL_TURNS: Dict[str, int] = {}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def store_message_db_fake(session_id: str, message: Message) -> bool:
    msgs = _MESSAGES.setdefault(session_id, [])
    msgs.append(message)

    # Keep sorted by turn (just in case)
    msgs.sort(key=lambda m: int(m["turn"]))

    # Track total turns
    current = int(_TOTAL_TURNS.get(session_id, 0))
    _TOTAL_TURNS[session_id] = max(current, int(message["turn"]))
    return True


def get_recent_messages_db_fake(session_id: str, last_n: int = 10) -> list[Message]:
    msgs = _MESSAGES.get(session_id, [])
    # return oldest -> newest
    return msgs[-last_n:] if last_n > 0 else []


def store_checkpoint_db_fake(session_id: str, checkpoint: MemoryCheckpoint) -> bool:
    cps = _CHECKPOINTS.setdefault(session_id, [])
    cps.append(checkpoint)

    # Keep sorted by turn_start
    cps.sort(key=lambda c: int(c["turn_range"][0]))
    return True


def get_checkpoints_db_fake(session_id: str) -> list[MemoryCheckpoint]:
    return list(_CHECKPOINTS.get(session_id, []))


def get_total_turns_db_fake(session_id: str) -> int:
    return int(_TOTAL_TURNS.get(session_id, 0))


# -----------------------------------------------------------------------------
# Patch the agent to use fake store instead of DB store
# -----------------------------------------------------------------------------

agent.store_message_db = store_message_db_fake
agent.get_recent_messages_db = get_recent_messages_db_fake
agent.store_checkpoint_db = store_checkpoint_db_fake
agent.get_checkpoints_db = get_checkpoints_db_fake
agent.get_total_turns_db = get_total_turns_db_fake


# -----------------------------------------------------------------------------
# Test scenario
# -----------------------------------------------------------------------------

def main():
    session_id = "session_demo_1"

    # 1) Store messages (manual conversation)
    ok1 = agent.store_message(session_id, {
        "id": "m1",
        "turn": 1,
        "speaker": "salesperson",
        "text": "ألو مساء الخير، معاك خدمة العملاء.",
        "emotion": None,
        "audio_path": None,
        "timestamp": _now(),
    })

    ok2 = agent.store_message(session_id, {
        "id": "m2",
        "turn": 2,
        "speaker": "vc",
        "text": "مساء النور، عندي مشكلة في الطلب.",
        "emotion": None,
        "audio_path": None,
        "timestamp": _now(),
    })

    ok3 = agent.store_message(session_id, {
        "id": "m3",
        "turn": 3,
        "speaker": "salesperson",
        "text": "تمام، ممكن تقولي رقم الطلب؟",
        "emotion": None,
        "audio_path": None,
        "timestamp": _now(),
    })

    print("store_message results:", ok1, ok2, ok3)

    # 2) Store a checkpoint (summary)
    ok_cp = agent.store_checkpoint(session_id, {
        "id": "cp1",
        "session_id": session_id,
        "turn_range": (1, 3),
        "summary": "بداية المكالمة: العميل ذكر مشكلة في الطلب والموظف طلب رقم الطلب.",
        "key_points": ["مشكلة في الطلب", "طلب رقم الطلب"],
        "customer_preferences": {},
        "objections_raised": ["تأخير/مشكلة في الطلب"],
        "created_at": _now(),
    })
    print("store_checkpoint result:", ok_cp)

    # 3) Read recent messages
    recent = agent.get_recent_messages(session_id, last_n=10)
    print("\nRecent messages:")
    for m in recent:
        print(f'  turn {m["turn"]} | {m["speaker"]}: {m["text"]}')

    # 4) Read checkpoints
    cps = agent.get_checkpoints(session_id)
    print("\nCheckpoints:")
    for c in cps:
        print(f'  turns {c["turn_range"]}: {c["summary"]}')

    # 5) Full memory
    mem = agent.get_session_memory(session_id)
    print("\nSessionMemory object:")
    print("  session_id:", mem["session_id"])
    print("  total_turns:", mem["total_turns"])
    print("  recent_messages:", len(mem["recent_messages"]))
    print("  checkpoints:", len(mem["checkpoints"]))


if __name__ == "__main__":
    main()