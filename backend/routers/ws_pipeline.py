# backend/routers/ws_pipeline.py
"""
Turn processing pipeline:
  audio → STT → emotion → LLM (streaming) → TTS (per-sentence) → client
Includes detailed per-step debug timing to diagnose latency.
"""

import asyncio
import base64
import time
import numpy as np
from uuid import UUID

from fastapi import WebSocket
from sqlalchemy.orm import Session

from backend.services import add_emotion_log, end_session
from backend.models import Persona
from backend.routers.ws_handler import ConversationHandler


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_salesperson_turn(text: str, persona: Persona) -> dict:
    """Rule-based quality score for the salesperson's response."""
    t = text.lower()
    bad  = ["مش فاهم", "مش عارف", "مش شغلي", "روح", "امشي", "سيبني", "غبي", "احمق", "بايخ"]
    good = ["أهلا", "مرحبا", "اتفضل", "أكيد", "طبعا", "بكل سرور",
            "فاهم", "معاك حق", "نقطة مهمة", "اسمحلي", "خليني اشرح", "ممكن اساعدك"]

    for p in bad:
        if p in t:
            return {"quality": "bad", "reason": "استخدام كلمات أو لهجة غير مناسبة",
                    "suggestion": "حاول تكون أكثر احترافية ولطافة مع العميل"}

    good_count = sum(1 for p in good if p in t)
    if good_count >= 2:
        return {"quality": "good", "reason": "رد مهذب ومحترف", "suggestion": None}
    if good_count >= 1:
        return {"quality": "neutral", "reason": "رد مقبول", "suggestion": "ممكن تكون أكثر ودية مع العميل"}
    return {"quality": "neutral", "reason": "رد محايد", "suggestion": "حاول تضيف تحية أو كلمات إيجابية"}


async def send_turn_results(
    websocket: WebSocket,
    results: dict,
    skip_transcription: bool = False,
    skip_audio: bool = False,
) -> None:
    """
    Push all turn data to the frontend after processing completes.
    In streaming mode transcription + audio chunks were already sent live,
    so we skip them here to avoid resending hundreds of KB of duplicate data
    (which was causing processing:completed to be delayed/dropped).
    """
    if not skip_transcription and results.get("transcription"):
        await websocket.send_json({"type": "transcription", "data": {"text": results["transcription"]}})

    if results.get("emotion_state"):
        es = results["emotion_state"]
        await websocket.send_json({
            "type": "emotion",
            "data": {
                "emotion":    es.customer_emotion,
                "mood_score": es.customer_mood_score,
                "risk_level": es.risk_level,
                "trend":      es.emotion_trend,
                "tip":        es.tip,
            },
        })

    await websocket.send_json({"type": "evaluation", "data": results["evaluation"]})
    await websocket.send_json({"type": "response", "data": {"text": results["response"]}})

    if not skip_audio and results.get("audio_base64"):
        await websocket.send_json({
            "type": "audio",
            "data": {"audio_base64": results["audio_base64"], "sample_rate": 24000},
        })


def _empty_results() -> dict:
    return {
        "transcription": "",
        "emotion": "neutral",
        "response": "",
        "audio_base64": "",
        "evaluation": {"quality": "neutral", "reason": "", "suggestion": ""},
        "emotion_state": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Audio validation (shared by both pipelines)
# ─────────────────────────────────────────────────────────────────────────────

def _validate_audio(audio: np.ndarray) -> str | None:
    """Return an error string if audio is invalid, else None."""
    if len(audio) < 8000:
        return "[صوت قصير جداً]"
    if np.all(audio == 0) or np.abs(audio).max() < 0.001:
        return "[صوت صامت أو تالف]"
    return None


def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Boost low-volume audio."""
    if len(audio) > 0 and np.abs(audio).max() > 0:
        max_val = np.abs(audio).max()
        if max_val < 0.1:
            audio = audio * (0.3 / max_val)
            print(f"[WS] Audio normalized: boosted by {0.3/max_val:.1f}x")
    return audio


# ─────────────────────────────────────────────────────────────────────────────
# Streaming pipeline  (LLM sentence → TTS chunk → client, pipelined)
# ─────────────────────────────────────────────────────────────────────────────

async def process_turn_streaming(
    handler: ConversationHandler,
    db: Session,
    websocket: WebSocket,
) -> dict:
    """
    Main real-time pipeline:
      audio → STT → emotion → [LLM sentence ↔ TTS chunk]* → memory save

    Debug tags in output:
      [PERF] lines show per-step wall-clock durations so you can pinpoint lags.
    """
    results = _empty_results()
    pipeline_start = time.perf_counter()

    try:
        # ── 1. Audio ─────────────────────────────────────────────────────
        audio = _normalize_audio(handler.get_full_audio())
        print(f"[WS] Audio stats: {len(audio)} samples, dtype: {audio.dtype}, "
              f"range: [{audio.min():.3f}, {audio.max():.3f}]")

        err = _validate_audio(audio)
        if err:
            results["transcription"] = err
            return results

        # ── 2. Prepare state ──────────────────────────────────────────────
        print(f"[WS] 🚀 Processing turn (STREAMING mode)...")
        from orchestration.state import reset_turn_state
        from orchestration.nodes import stt_node, emotion_node, memory_load_node, memory_save_node
        from orchestration.nodes.llm_node import llm_node_streaming
        from orchestration.nodes.tts_node import tts_chunk

        agent = handler.agent
        config = agent.config
        agent.state = reset_turn_state(agent.state)
        agent.state["audio_input"] = audio
        state = agent.state

        if config.verbose:
            print(f"\n{'='*60}")
            print(f"[AGENT] Processing turn {state['turn_count'] + 1} (streaming)")
            print(f"{'='*60}")

        # ── 3. Pre-LLM nodes ──────────────────────────────────────────────
        t0 = time.perf_counter()
        state = memory_load_node(state, config)
        print(f"[PERF] memory_load: {time.perf_counter()-t0:.3f}s")

        t0 = time.perf_counter()
        state = stt_node(state, config)
        print(f"[PERF] stt: {time.perf_counter()-t0:.3f}s  → '{state.get('transcription','')[:60]}'")

        t0 = time.perf_counter()
        state = emotion_node(state, config)
        em = state.get("emotion", {})
        print(f"[PERF] emotion: {time.perf_counter()-t0:.3f}s  → {em.get('primary_emotion','?')} "
              f"({em.get('confidence',0):.2f})")

        results["transcription"] = state.get("transcription", "")
        await websocket.send_json({"type": "transcription", "data": {"text": results["transcription"]}})

        emotion_result = state.get("emotion") or {"primary_emotion": "neutral", "confidence": 0.5}
        results["emotion"] = emotion_result.get("primary_emotion", "neutral")

        # ── 4. Streaming LLM → TTS (sequential — Chatterbox is not thread-safe) ─
        # TTS calls are sequential to avoid GPU tensor corruption from concurrent
        # calls on the same singleton model. The LLM worker thread continues
        # buffering the next sentence while TTS runs, so there is still overlap
        # between LLM streaming and TTS synthesis.
        all_audio_chunks: list[np.ndarray] = []
        full_response = ""
        chunk_count = 0
        llm_start = time.perf_counter()

        for sentence, _updated_state in llm_node_streaming(state, config):
            chunk_count += 1
            full_response += (" " + sentence if full_response else sentence)

            tts_t0 = time.perf_counter()
            audio_chunk = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda s=sentence: tts_chunk(s, state, config),
            )
            tts_elapsed = time.perf_counter() - tts_t0
            audio_dur = (len(audio_chunk) / 24000) if audio_chunk is not None and len(audio_chunk) > 0 else 0

            print(f"[PERF] tts_chunk[{chunk_count}]: {tts_elapsed:.3f}s → "
                  f"{audio_dur:.2f}s audio  (RTF={tts_elapsed/max(audio_dur,0.001):.2f}x)  "
                  f"text='{sentence[:40]}'")

            if audio_chunk is not None and len(audio_chunk) > 0:
                all_audio_chunks.append(audio_chunk)
                audio_b64 = base64.b64encode(audio_chunk.astype(np.float32).tobytes()).decode()
                await websocket.send_json({
                    "type": "audio_chunk",
                    "data": {
                        "audio_base64": audio_b64,
                        "sample_rate":  24000,
                        "chunk_index":  chunk_count,
                        "text":         sentence,
                        "is_final":     False,
                    },
                })
                print(f"[WS] 🔊 Chunk {chunk_count}: '{sentence[:30]}' ({tts_elapsed:.2f}s TTS)")

        llm_tts_elapsed = time.perf_counter() - llm_start
        print(f"[PERF] llm+tts total: {llm_tts_elapsed:.3f}s for {chunk_count} sentences")

        state["llm_response"] = full_response.strip()
        state["phase"] = "idle"
        results["response"] = full_response.strip()

        if all_audio_chunks:
            combined = np.concatenate(all_audio_chunks)
            state["audio_output"] = combined
            results["audio_base64"] = base64.b64encode(combined.astype(np.float32).tobytes()).decode()

        await websocket.send_json({
            "type": "audio_chunk",
            "data": {"is_final": True, "total_chunks": chunk_count},
        })
        print(f"[WS] ✅ Streaming done: {chunk_count} chunks in {llm_tts_elapsed:.2f}s")

        # ── 5. Memory save ────────────────────────────────────────────────
        t0 = time.perf_counter()
        state = memory_save_node(state, config)
        print(f"[PERF] memory_save: {time.perf_counter()-t0:.3f}s")
        agent.state = state

        # ── 6. Evaluation + emotion state ─────────────────────────────────
        evaluation = evaluate_salesperson_turn(results["transcription"], handler.persona)
        results["evaluation"] = evaluation
        emotion_state = handler.get_emotion_state(results["emotion"], evaluation["quality"])
        results["emotion_state"] = emotion_state
        handler.turn_count += 1

        # Update DB turn count + emotion log
        try:
            add_emotion_log(db, UUID(handler.session_id), None, emotion_state)
            handler.training_session.turn_count = handler.turn_count
            db.commit()
        except Exception as db_err:
            print(f"[WS] DB save warning (non-critical): {db_err}")

        total = time.perf_counter() - pipeline_start
        state["node_timings"]["total"] = total

        if config.verbose:
            t_str = state.get('transcription', 'N/A')[:50]
            r_str = state.get('llm_response', 'N/A')[:50]
            print(f"\n[TURN SUMMARY]")
            print(f"  Input:  '{t_str}'")
            print(f"  Output: '{r_str}'")
            print(f"\n[TIMINGS]")
            for node, dur in state["node_timings"].items():
                print(f"  {node}: {dur:.3f}s")
        print(f"[PERF] ─── TOTAL TURN: {total:.3f}s ───")

    except Exception as e:
        print(f"[WS] Streaming pipeline error: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to non-streaming
        try:
            print("[WS] Falling back to non-streaming...")
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: _process_turn_sync(handler, db),
            )
        except Exception as fallback_err:
            print(f"[WS] Fallback also failed: {fallback_err}")
            results["response"] = "عذراً، حصل مشكلة. ممكن تعيد الكلام؟"

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Non-streaming pipeline  (used as fallback)
# ─────────────────────────────────────────────────────────────────────────────

async def process_turn_non_streaming(
    handler: ConversationHandler,
    db: Session,
) -> dict:
    """Run the full turn synchronously via the LangGraph agent."""
    results = _empty_results()
    pipeline_start = time.perf_counter()

    try:
        audio = _normalize_audio(handler.get_full_audio())
        err = _validate_audio(audio)
        if err:
            results["transcription"] = err
            return results

        print("[WS] 🚀 Processing turn through LangGraph Orchestration...")
        t0 = time.perf_counter()
        state = handler.agent.process_turn(audio)
        print(f"[PERF] langgraph turn: {time.perf_counter()-t0:.3f}s")

        results["transcription"] = state.get("transcription", "")
        results["response"]      = state.get("llm_response", "")
        emotion_result = state.get("emotion") or {"primary_emotion": "neutral"}
        results["emotion"] = emotion_result.get("primary_emotion", "neutral")

        audio_out = state.get("audio_output")
        if audio_out is not None:
            results["audio_base64"] = base64.b64encode(audio_out.astype(np.float32).tobytes()).decode()

        evaluation  = evaluate_salesperson_turn(results["transcription"], handler.persona)
        results["evaluation"] = evaluation
        emotion_state = handler.get_emotion_state(results["emotion"], evaluation["quality"])
        results["emotion_state"] = emotion_state
        handler.turn_count += 1

        try:
            add_emotion_log(db, UUID(handler.session_id), None, emotion_state)
            handler.training_session.turn_count = handler.turn_count
            db.commit()
        except Exception as db_err:
            print(f"[WS] DB save warning: {db_err}")

        print(f"[PERF] ─── TOTAL TURN (non-streaming): {time.perf_counter()-pipeline_start:.3f}s ───")

    except Exception as e:
        print(f"[WS] Error in non-streaming turn: {e}")
        import traceback
        traceback.print_exc()
        results["response"] = "عذراً، حصل مشكلة. ممكن تعيد الكلام؟"

    return results


def _process_turn_sync(handler: ConversationHandler, db: Session) -> dict:
    """
    Synchronous wrapper for process_turn_non_streaming.
    Used by run_in_executor from the fallback path.
    """
    import asyncio as _asyncio
    try:
        loop = _asyncio.get_event_loop()
    except RuntimeError:
        loop = _asyncio.new_event_loop()
        _asyncio.set_event_loop(loop)
    return loop.run_until_complete(process_turn_non_streaming(handler, db))
