# backend/routers/websocket.py
"""
WebSocket endpoint — auth, message loop, session lifecycle.

Split layout:
  ws_connection.py  — ConnectionManager singleton
  ws_handler.py     — ConversationHandler (per-session state + OrchestrationAgent)
  ws_pipeline.py    — Turn processing pipelines + evaluate_salesperson_turn
  websocket.py      — Router + main message loop  (this file)
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import asyncio
import base64
import numpy as np
from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from sqlalchemy.orm import Session

from backend.database import SessionLocal
from backend.services import decode_token, get_session, end_session
from backend.models import Session as TrainingSession, Persona

from backend.routers.ws_connection import manager
from backend.routers.ws_handler   import ConversationHandler
from backend.routers.ws_pipeline  import (
    process_turn_streaming,
    process_turn_non_streaming,
    send_turn_results,
    _process_turn_sync,
)

router = APIRouter()


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    token: str = Query(...),
):
    """
    Real-time conversation WebSocket.

    Client → Server message types:
      audio           — streaming audio chunk (base64)
      audio_complete  — full turn audio (WebM, converted server-side)
      end_speaking    — process buffered audio
      end_session     — terminate session
      ping            — keepalive

    Server → Client message types:
      connected       — handshake ack
      processing      — {status: "started"|"completed"}
      transcription   — {text}
      emotion         — {emotion, mood_score, risk_level, trend, tip}
      evaluation      — {quality, reason, suggestion}
      response        — {text}
      audio_chunk     — {audio_base64, sample_rate, chunk_index, is_final}
      audio           — {audio_base64, sample_rate}  (non-streaming fallback)
      session_ended   — {reason, message}
      ping / pong     — keepalive
      error           — {message}
      info            — {message}  (e.g. "busy")
    """
    token_data = decode_token(token)
    if not token_data:
        await websocket.close(code=4001, reason="Invalid token")
        return

    db = SessionLocal()
    handler: ConversationHandler | None = None
    _keepalive_task = None
    _inactivity_task = None

    try:
        # ── Session auth ──────────────────────────────────────────────────
        training_session = get_session(db, UUID(session_id))
        if str(training_session.user_id) != token_data.user_id:
            await websocket.close(code=4003, reason="Access denied")
            return
        if training_session.status != "active":
            await websocket.close(code=4004, reason="Session not active")
            return

        persona = db.query(Persona).filter(Persona.id == training_session.persona_id).first()

        # ── Build handler (starts OrchestrationAgent + TTS pre-warm) ─────
        handler = ConversationHandler(
            session_id=session_id,
            user_id=token_data.user_id,
            training_session=training_session,
            persona=persona,
        )

        await manager.connect(session_id, websocket)

        await websocket.send_json({
            "type": "connected",
            "data": {
                "session_id":    session_id,
                "persona": {
                    "id":         persona.id,
                    "name_ar":    persona.name_ar,
                    "name_en":    persona.name_en,
                    "difficulty": persona.difficulty,
                },
                "message":       f"متصل مع {persona.name_ar}",
                "orchestration": "LangGraph",
            },
        })

        # ── Server keepalive (25 s) ───────────────────────────────────────
        async def _keepalive():
            while True:
                await asyncio.sleep(25)
                try:
                    await websocket.send_json({"type": "ping"})
                except Exception:
                    break

        _keepalive_task = asyncio.create_task(_keepalive())

        # ── Inactivity auto-end (5 min) ───────────────────────────────────
        import time as _t
        _last_audio = [_t.time()]

        async def _inactivity_watcher():
            while True:
                await asyncio.sleep(60)
                idle = _t.time() - _last_audio[0]
                if idle >= 300:
                    print(f"[WS] Auto-ending {session_id} — idle {idle:.0f}s")
                    try:
                        end_session(db, UUID(session_id))
                        await websocket.send_json({
                            "type": "session_ended",
                            "data": {
                                "reason":  "inactivity",
                                "message": "تم إنهاء الجلسة تلقائياً بسبب عدم النشاط",
                            },
                        })
                    except Exception:
                        pass
                    break

        _inactivity_task = asyncio.create_task(_inactivity_watcher())

        # ── Main message loop ─────────────────────────────────────────────
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            print(f"[WS] Received message type: {msg_type}")

            # ── audio (streaming chunk) ───────────────────────────────────
            if msg_type == "audio":
                handler.add_audio_chunk(data.get("data", {}).get("audio_base64", ""))

            # ── audio_complete (full WebM turn) ───────────────────────────
            elif msg_type == "audio_complete":
                audio_data   = data.get("data", {}).get("audio_base64", "")
                audio_format = data.get("data", {}).get("format", "webm")

                if audio_data:
                    try:
                        import tempfile, subprocess, wave
                        audio_bytes = base64.b64decode(audio_data)
                        with tempfile.NamedTemporaryFile(suffix=f".{audio_format}", delete=False) as f:
                            f.write(audio_bytes)
                            tmp_in = f.name
                        tmp_out = tmp_in.replace(f".{audio_format}", ".wav")
                        subprocess.run(
                            ["ffmpeg", "-y", "-i", tmp_in,
                             "-ar", "16000", "-ac", "1", "-f", "wav", tmp_out],
                            capture_output=True,
                        )
                        with wave.open(tmp_out, "rb") as wf:
                            frames = wf.readframes(wf.getnframes())
                            audio_arr = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                        os.unlink(tmp_in)
                        os.unlink(tmp_out)
                        handler.audio_buffer = [audio_arr]
                        # Stash the raw bytes + format on the handler so the
                        # pipeline can persist them for replay after we know
                        # the turn number.
                        handler.pending_salesperson_audio_bytes = audio_bytes
                        handler.pending_salesperson_audio_format = audio_format
                    except Exception as e:
                        print(f"[WS] Audio conversion error: {e}")
                        await websocket.send_json({"type": "error", "data": {"message": "Audio conversion failed"}})
                        continue

                _last_audio[0] = _t.time()

                if handler.is_processing:
                    await websocket.send_json({"type": "info", "data": {"message": "busy"}})
                    continue

                handler.is_processing = True
                try:
                    await websocket.send_json({"type": "processing", "data": {"status": "started"}})
                    await _run_turn(handler, db, websocket)
                finally:
                    handler.is_processing = False

            # ── end_speaking ──────────────────────────────────────────────
            elif msg_type == "end_speaking":
                if handler.is_processing:
                    continue
                handler.is_processing = True
                try:
                    await websocket.send_json({"type": "processing", "data": {"status": "started"}})
                    await _run_turn(handler, db, websocket)
                finally:
                    handler.is_processing = False

            # ── end_session ───────────────────────────────────────────────
            elif msg_type == "end_session":
                summary = handler.end_orchestration_session()
                end_session(db, UUID(session_id))
                await websocket.send_json({
                    "type": "session_ended",
                    "data": {
                        "total_turns":           handler.turn_count,
                        "final_mood":            handler.customer_mood,
                        "message":               "تم إنهاء الجلسة",
                        "orchestration_summary": summary,
                    },
                })
                break

            # ── ping ──────────────────────────────────────────────────────
            elif msg_type == "ping":
                pass  # client ping — no pong needed (server pings on its own cadence)

    except WebSocketDisconnect:
        print(f"[WS] Client disconnected: {session_id}")

    except Exception as e:
        print(f"[WS] Unhandled error: {e}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.send_json({"type": "error", "data": {"message": str(e)}})
        except Exception:
            pass

    finally:
        if _keepalive_task:
            _keepalive_task.cancel()
        if _inactivity_task:
            _inactivity_task.cancel()
        if handler:
            handler.end_orchestration_session()
        manager.disconnect(session_id)
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _run_turn(
    handler: ConversationHandler,
    db: Session,
    websocket: WebSocket,
) -> None:
    """Run one full turn and push all results to the client."""
    try:
        if handler.agent.config.enable_streaming:
            results = await process_turn_streaming(handler, db, websocket)
            # Streaming already sent audio chunks AND transcription live —
            # skip both here to avoid duplicate 100s-of-KB sends that were
            # delaying / dropping processing:completed.
            await send_turn_results(
                websocket, results,
                skip_transcription=True,
                skip_audio=True,
            )
        else:
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: _process_turn_sync(handler, db),
            )
            await send_turn_results(websocket, results, skip_transcription=False)
    except Exception as e:
        print(f"[WS] _run_turn error (still finalizing): {e}")
        import traceback
        traceback.print_exc()
    finally:
        # ALWAYS send processing:completed — even on errors. Without this
        # the frontend's isProcessing stays true and the record button
        # is permanently disabled.
        try:
            await websocket.send_json({"type": "processing", "data": {"status": "completed"}})
        except Exception as send_err:
            print(f"[WS] Failed to send processing:completed: {send_err}")
