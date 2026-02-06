# memory/store.py

from _future_ import annotations

from datetime import datetime, timezone
from typing import cast
from uuid import UUID

from backend.database import get_db_context
from backend.models.session import (
    Session,
    Message as MessageDB,
    Checkpoint as CheckpointDB,
)

from shared.types import Message, MemoryCheckpoint


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _now_dt() -> datetime:
    return datetime.now(timezone.utc)


def _parse_session_id(session_id: str) -> UUID:
    """
    Convert incoming session_id string into a UUID.
    Your DB models use UUID types, so this prevents type mismatch bugs.
    """
    return UUID(session_id)


def _to_db_speaker(speaker: str) -> str:
    """
    shared.types Speaker: "salesperson" | "vc"
    DB speaker: "salesperson" | "customer"
    """
    return "customer" if speaker == "vc" else "salesperson"


def _from_db_speaker(speaker: str) -> str:
    return "vc" if speaker == "customer" else "salesperson"


def _db_message_to_typed(m: MessageDB) -> Message:
    """
    Convert DB Message -> shared.types.Message (TypedDict)
    IMPORTANT: Must be called while DB session is still open!
    """
    return cast(Message, {
        "id": str(m.id),
        "turn": int(m.turn_number),
        "speaker": cast(any, _from_db_speaker(m.speaker)),
        "text": m.text,
        "emotion": None,  # optional: map from detected_emotion/emotion_scores later
        "audio_path": m.audio_path,
        "timestamp": m.created_at if m.created_at else _now_dt(),
    })


def _db_checkpoint_to_typed(c: CheckpointDB) -> MemoryCheckpoint:
    """
    Convert DB Checkpoint -> shared.types.MemoryCheckpoint (TypedDict)
    IMPORTANT: Must be called while DB session is still open!
    """
    return cast(MemoryCheckpoint, {
        "id": str(c.id),
        "session_id": str(c.session_id),
        "turn_range": (int(c.turn_start), int(c.turn_end)),
        "summary": c.summary,
        "key_points": list(c.key_points or []),
        "customer_preferences": dict(c.customer_preferences or {}),
        "objections_raised": list(c.objections_raised or []),
        "created_at": c.created_at if c.created_at else _now_dt(),
    })


# -----------------------------------------------------------------------------
# Functions used by agent.py
# -----------------------------------------------------------------------------

def store_message_db(session_id: str, message: Message) -> bool:
    """
    Store a message in existing 'messages' table and update sessions.turn_count.
    """
    try:
        sid = _parse_session_id(session_id)

        with get_db_context() as db:
            sess = db.get(Session, sid)
            if sess is None:
                return False

            m = MessageDB(
                session_id=sid,
                turn_number=int(message["turn"]),
                speaker=_to_db_speaker(message["speaker"]),
                text=message["text"],
                audio_path=message.get("audio_path"),
                # DB column is created_at; if None, DB default func.now() will handle it
                created_at=message.get("timestamp"),
            )

            db.add(m)

            # Keep turn_count consistent (use max to avoid decreasing it)
            current = int(sess.turn_count or 0)
            sess.turn_count = max(current, int(message["turn"]))

        return True
    except Exception:
        return False


def get_recent_messages_db(session_id: str, last_n: int = 10) -> list[Message]:
    """
    Get last_n messages (oldest -> newest).
    """
    sid = _parse_session_id(session_id)

    with get_db_context() as db:
        rows = (
            db.query(MessageDB)
            .filter(MessageDB.session_id == sid)
            .order_by(MessageDB.turn_number.desc())
            .limit(last_n)
            .all()
        )
        
        # ✅ FIX: Convert to typed dicts INSIDE the session context
        # This ensures all attributes are accessed while session is open
        result = [_db_message_to_typed(r) for r in reversed(rows)]

    return result


def store_checkpoint_db(session_id: str, checkpoint: MemoryCheckpoint) -> bool:
    """
    Store checkpoint in existing 'checkpoints' table.
    """
    try:
        sid = _parse_session_id(session_id)

        with get_db_context() as db:
            sess = db.get(Session, sid)
            if sess is None:
                return False

            turn_start, turn_end = checkpoint["turn_range"]

            c = CheckpointDB(
                session_id=sid,
                turn_start=int(turn_start),
                turn_end=int(turn_end),
                summary=checkpoint["summary"],
                key_points=checkpoint.get("key_points") or [],
                customer_preferences=checkpoint.get("customer_preferences") or {},
                objections_raised=checkpoint.get("objections_raised") or [],
                created_at=checkpoint.get("created_at"),
            )

            db.add(c)

        return True
    except Exception:
        return False


def get_checkpoints_db(session_id: str) -> list[MemoryCheckpoint]:
    """
    Get all checkpoints (oldest -> newest).
    """
    sid = _parse_session_id(session_id)

    with get_db_context() as db:
        rows = (
            db.query(CheckpointDB)
            .filter(CheckpointDB.session_id == sid)
            .order_by(CheckpointDB.turn_start.asc())
            .all()
        )
        
        # ✅ FIX: Convert to typed dicts INSIDE the session context
        result = [_db_checkpoint_to_typed(r) for r in rows]

    return result


def get_total_turns_db(session_id: str) -> int:
    """
    Return total turns from sessions.turn_count.
    """
    sid = _parse_session_id(session_id)

    with get_db_context() as db:
        sess = db.get(Session, sid)
        return int(sess.turn_count) if sess and sess.turn_count is not None else 0