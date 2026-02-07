# backend/routers/websocket.py
"""
WebSocket endpoint for real-time conversation.

This handles:
- Audio streaming from salesperson
- Real-time transcription (via Orchestration Agent)
- Emotion detection & updates
- AI response generation (via Orchestration Agent)
- TTS audio streaming back
- Real-time evaluation tips
- Memory checkpoints (automatic every 5 turns)

NOW USING: LangGraph Orchestration Agent!
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import base64
import asyncio
import numpy as np
from uuid import UUID
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from sqlalchemy.orm import Session

from backend.database import get_db, SessionLocal
from backend.services import get_session, add_message, add_emotion_log, decode_token, end_session
from backend.schemas import MessageCreate, EmotionState
from backend.models import Session as TrainingSession, Persona

# ══════════════════════════════════════════════════════════════════════════════
# IMPORT ORCHESTRATION AGENT
# ══════════════════════════════════════════════════════════════════════════════
from orchestration import OrchestrationAgent
from orchestration.config import OrchestrationConfig
from orchestration.nodes.llm_node import llm_node_streaming
from orchestration.nodes.tts_node import tts_chunk
from orchestration.state import reset_turn_state
import asyncio  # Should already be there
from orchestration.config import get_config  # Add this if not present
router = APIRouter()


class ConnectionManager:
    """Manages active WebSocket connections."""
    
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
    
    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        print(f"[WS] Client connected: {session_id}")
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            print(f"[WS] Client disconnected: {session_id}")
    
    async def send_json(self, session_id: str, data: dict):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(data)
    
    async def send_bytes(self, session_id: str, data: bytes):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_bytes(data)


manager = ConnectionManager()


class ConversationHandler:
    """
    Handles a single conversation session.
    
    NOW USES: OrchestrationAgent for processing turns through LangGraph!
    """
    
    def __init__(self, session_id: str, user_id: str, training_session: TrainingSession, persona: Persona):
        self.session_id = session_id
        self.user_id = user_id
        self.training_session = training_session
        self.persona = persona
        self.turn_count = training_session.turn_count or 0
        self.audio_buffer = []
        self.is_processing = False
        
        # Emotion tracking (for frontend display)
        self.customer_mood = 50  # Start neutral (0-100 scale, 50 = neutral)
        self.emotion_history = []
        
        # ══════════════════════════════════════════════════════════════════
        # INITIALIZE ORCHESTRATION AGENT
        # ══════════════════════════════════════════════════════════════════

        self.agent = OrchestrationAgent(
        use_mocks=False,
        verbose=True,
        enable_streaming=False
        )

        
        # Start the orchestration session
        self._start_orchestration_session()
    
    def _start_orchestration_session(self):
        """Initialize the orchestration agent session."""
        try:
            # Convert persona to dict format expected by orchestration
            persona_dict = {
                "id": self.persona.id,
                "name": self.persona.name_ar,
                "name_en": self.persona.name_en,
                "description": getattr(self.persona, 'description_ar', ''),
                "personality_prompt": self.persona.personality_prompt,
                "voice_id": self.persona.voice_id,
                "default_emotion": "neutral",
                "difficulty": self.persona.difficulty,
                "traits": self.persona.traits or [],
                "avatar_url": getattr(self.persona, 'avatar_url', None)
            }
            
            # Start session in orchestration agent - pass persona_dict directly!
            self.agent.start_session(
                session_id=self.session_id,
                user_id=self.user_id,
                persona_id=self.persona.id,
                persona_dict=persona_dict  # Pass directly to avoid loading
            )
            
            print(f"[WS] Orchestration session started for {self.session_id}")
            
        except Exception as e:
            print(f"[WS] Error starting orchestration session: {e}")
            import traceback
            traceback.print_exc()
    
    def add_audio_chunk(self, audio_base64: str):
        """Add audio chunk to buffer."""
        try:
            # Decode base64
            audio_bytes = base64.b64decode(audio_base64)
            
            # Convert bytes to numpy array (Little Endian Float32)
            audio_array = np.frombuffer(audio_bytes, dtype='<f4')
            
            # Ensure it's the right dtype
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            
            print(f"[WS] Received audio chunk: {len(audio_array)} samples, range: [{audio_array.min():.3f}, {audio_array.max():.3f}]")
            
            self.audio_buffer.append(audio_array)
        except Exception as e:
            print(f"[WS] Error decoding audio: {e}")
            import traceback
            traceback.print_exc()
    
    def get_full_audio(self) -> np.ndarray:
        """Get concatenated audio from buffer."""
        if not self.audio_buffer:
            return np.array([], dtype=np.float32)
        full_audio = np.concatenate(self.audio_buffer)
        self.audio_buffer = []  # Clear buffer
        return full_audio
    
    def update_customer_mood(self, emotion: str, salesperson_quality: str):
        """Update customer mood based on emotion and salesperson performance."""
        mood_changes = {
            "good": 5,
            "neutral": 0,
            "bad": -10
        }
        
        sensitivity_multiplier = (self.persona.emotion_sensitivity or 50) / 50
        change = mood_changes.get(salesperson_quality, 0) * sensitivity_multiplier
        self.customer_mood = max(0, min(100, self.customer_mood + change))
        
        return self.customer_mood
    
    def get_emotion_state(self, emotion: str, quality: str) -> EmotionState:
        """Get current emotion state for frontend."""
        mood = self.update_customer_mood(emotion, quality)
        
        # Determine risk level
        if mood < 30:
            risk_level = "high"
        elif mood < 50:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        # Determine trend
        if len(self.emotion_history) >= 2:
            recent_avg = sum(self.emotion_history[-3:]) / len(self.emotion_history[-3:])
            if mood > recent_avg + 5:
                trend = "improving"
            elif mood < recent_avg - 5:
                trend = "worsening"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        self.emotion_history.append(mood)
        
        tip = self._generate_tip(emotion, risk_level, quality)
        
        return EmotionState(
            customer_emotion=emotion,
            customer_mood_score=int(mood * 2 - 100),
            risk_level=risk_level,
            emotion_trend=trend,
            tip=tip
        )
    
    def _generate_tip(self, emotion: str, risk_level: str, quality: str) -> Optional[str]:
        """Generate a helpful tip for the salesperson."""
        tips = {
            ("angry", "high"): "⚠️ العميل زعلان! حاول تهديه وتسمعله",
            ("angry", "medium"): "العميل مش مبسوط، خد بالك من لهجتك",
            ("frustrated", "high"): "العميل محبط، حاول تفهم مشكلته",
            ("sad", "medium"): "العميل قلقان، طمنه وادي له معلومات واضحة",
            ("neutral", "low"): None,
            ("happy", "low"): "👍 كمل كده، العميل مبسوط!",
        }
        
        tip = tips.get((emotion, risk_level))
        if tip:
            return tip
        
        if quality == "bad":
            return "💡 حاول تكون أكثر احترافية في ردك"
        
        return None
    
    def end_orchestration_session(self):
        """End the orchestration agent session."""
        try:
            if self.agent:
                summary = self.agent.end_session()
                print(f"[WS] Orchestration session ended: {summary}")
                return summary
        except Exception as e:
            print(f"[WS] Error ending orchestration session: {e}")
        return {}
    

async def process_turn_streaming(handler, db, websocket) -> dict:
    """
    Process turn with streaming: LLM sentences -> TTS chunks -> send immediately.
    User hears first sentence ~2s faster than waiting for full response.
    """
    import time as _time
    import asyncio
    import base64
    import numpy as np
    from uuid import UUID

    results = {
        "transcription": "",
        "emotion": "neutral",
        "response": "",
        "audio_base64": "",
        "evaluation": {"quality": "neutral", "reason": "", "suggestion": ""},
        "emotion_state": None
    }

    try:
        # ══════════════════════════════════════════════════════════════════
        # 1. GET AND VALIDATE AUDIO
        # ══════════════════════════════════════════════════════════════════
        audio = handler.get_full_audio()

        if len(audio) > 0 and np.abs(audio).max() > 0:
            max_val = np.abs(audio).max()
            if max_val < 0.1:
                target_level = 0.3
                audio = audio * (target_level / max_val)
                print(f"[WS] Audio normalized: boosted by {target_level/max_val:.1f}x")

        print(f"[WS] Audio stats: {len(audio)} samples, dtype: {audio.dtype}, range: [{audio.min():.3f}, {audio.max():.3f}]")

        if len(audio) < 8000:
            results["transcription"] = "[صوت قصير جداً]"
            return results

        if np.all(audio == 0) or np.abs(audio).max() < 0.001:
            results["transcription"] = "[صوت صامت أو تالف]"
            return results

        # ══════════════════════════════════════════════════════════════════
        # 2. PREPARE STATE (same way agent.process_turn does it)
        # ══════════════════════════════════════════════════════════════════
        print(f"[WS] 🚀 Processing turn (STREAMING mode)...")

        agent = handler.agent
        config = agent.config

        # Reset turn state exactly like agent.process_turn() does
        agent.state = reset_turn_state(agent.state)
        agent.state["audio_input"] = audio
        state = agent.state

        if config.verbose:
            print(f"\n{'='*60}")
            print(f"[AGENT] Processing turn {state['turn_count'] + 1} (streaming)")
            print(f"{'='*60}")

        turn_start = _time.time()

        # ══════════════════════════════════════════════════════════════════
        # 3. RUN PRE-LLM NODES (same order as graph)
        # ══════════════════════════════════════════════════════════════════
        from orchestration.nodes import (
            stt_node,
            emotion_node,
            rag_node,
            memory_load_node,
            memory_save_node,
        )
        from orchestration.nodes.llm_node import llm_node_streaming
        from orchestration.nodes.tts_node import tts_chunk

        state = memory_load_node(state, config)
        state = stt_node(state, config)
        state = emotion_node(state, config)
        state = rag_node(state, config)

        results["transcription"] = state.get("transcription", "")

        # Send transcription immediately (user sees what they said fast)
        await websocket.send_json({
            "type": "transcription",
            "data": {"text": results["transcription"]}
        })

        emotion_result = state.get("emotion") or {"primary_emotion": "neutral", "confidence": 0.5}
        results["emotion"] = emotion_result.get("primary_emotion", "neutral")

        # ══════════════════════════════════════════════════════════════════
        # 4. STREAMING LLM -> TTS PIPELINE
        # ══════════════════════════════════════════════════════════════════
        all_audio_chunks = []
        full_response = ""
        chunk_count = 0
        llm_start = _time.time()

        for sentence, updated_state in llm_node_streaming(state, config):
            chunk_count += 1
            full_response += (" " + sentence if full_response else sentence)

            # Generate TTS for this sentence in thread pool (non-blocking)
            tts_start = _time.time()
            audio_chunk = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda s=sentence: tts_chunk(s, state, config)
            )
            tts_elapsed = _time.time() - tts_start

            if audio_chunk is not None and len(audio_chunk) > 0:
                all_audio_chunks.append(audio_chunk)

                # Send audio chunk to frontend IMMEDIATELY
                audio_bytes = audio_chunk.astype(np.float32).tobytes()
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

                await websocket.send_json({
                    "type": "audio_chunk",
                    "data": {
                        "audio_base64": audio_b64,
                        "sample_rate": 24000,
                        "chunk_index": chunk_count,
                        "text": sentence,
                        "is_final": False
                    }
                })

                print(f"[WS] 🔊 Chunk {chunk_count}: '{sentence[:30]}...' ({tts_elapsed:.2f}s TTS)")

        llm_tts_elapsed = _time.time() - llm_start

        # Store complete response
        state["llm_response"] = full_response.strip()
        state["phase"] = "idle"
        results["response"] = full_response.strip()

        # Combine all audio for state
        if all_audio_chunks:
            combined_audio = np.concatenate(all_audio_chunks)
            state["audio_output"] = combined_audio
            audio_bytes = combined_audio.astype(np.float32).tobytes()
            results["audio_base64"] = base64.b64encode(audio_bytes).decode('utf-8')

        # Signal end of streaming
        await websocket.send_json({
            "type": "audio_chunk",
            "data": {"is_final": True, "total_chunks": chunk_count}
        })

        print(f"[WS] ✅ Streaming done: {chunk_count} chunks in {llm_tts_elapsed:.2f}s")

        # ══════════════════════════════════════════════════════════════════
        # 5. MEMORY SAVE
        # ══════════════════════════════════════════════════════════════════
        state = memory_save_node(state, config)

        # Update agent state so next turn has correct history
        agent.state = state

        # Calculate total time
        total_time = _time.time() - turn_start
        state["node_timings"]["total"] = total_time

        # Print turn summary
        if config.verbose:
            transcription = state.get('transcription', 'N/A')
            if transcription and len(transcription) > 50:
                transcription = transcription[:50] + "..."
            em = state.get('emotion', {})
            em_label = em.get('primary_emotion', 'N/A') if isinstance(em, dict) else em
            response = state.get('llm_response', 'N/A')
            if response and len(response) > 50:
                response = response[:50] + "..."

            print(f"\n[TURN SUMMARY]")
            print(f"  Input: '{transcription}'")
            print(f"  Emotion: {em_label}")
            print(f"  Output: '{response}'")
            print(f"\n[TIMINGS]")
            for node, t in state["node_timings"].items():
                print(f"  {node}: {t:.3f}s")

        # ══════════════════════════════════════════════════════════════════
        # 6. EVALUATE + EMOTION STATE
        # ══════════════════════════════════════════════════════════════════
        evaluation = evaluate_salesperson_turn(results["transcription"], handler.persona)
        results["evaluation"] = evaluation

        emotion_state = handler.get_emotion_state(results["emotion"], evaluation["quality"])
        results["emotion_state"] = emotion_state

        handler.turn_count = state.get("turn_count", handler.turn_count + 1)

        try:
            add_emotion_log(db, UUID(handler.session_id), None, emotion_state)
            handler.training_session.turn_count = handler.turn_count
            db.commit()
        except Exception as db_error:
            print(f"[WS] DB save warning: {db_error}")

    except Exception as e:
        print(f"[WS] Streaming error: {e}")
        import traceback
        traceback.print_exc()

        # Fallback: try non-streaming
        try:
            print(f"[WS] Falling back to non-streaming...")
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: process_turn_with_orchestration_sync(handler, db)
            )
        except Exception as fallback_error:
            print(f"[WS] Fallback also failed: {fallback_error}")
            results["response"] = "عذراً، حصل مشكلة. ممكن تعيد الكلام؟"

    return results


async def process_turn_with_orchestration(
    handler: ConversationHandler,
    db: Session
) -> dict:
    """
    Process a complete turn using the LangGraph Orchestration Agent.
    
    Flow (via LangGraph):
        memory_load → stt → emotion → rag → llm → tts → memory_save
    
    This automatically:
    - Loads memory context (checkpoints + recent messages)
    - Transcribes audio (real STT)
    - Detects emotion (mock for now)
    - Retrieves RAG context (mock for now)
    - Generates response (real LLM)
    - Synthesizes audio (mock for now)
    - Saves messages and creates checkpoints (real Memory Agent)
    """
    results = {
        "transcription": "",
        "emotion": "neutral",
        "response": "",
        "audio_base64": "",
        "evaluation": {"quality": "neutral", "reason": "", "suggestion": ""},
        "emotion_state": None
    }
    
    try:
        # ══════════════════════════════════════════════════════════════════
        # 1. GET AND VALIDATE AUDIO
        # ══════════════════════════════════════════════════════════════════
        audio = handler.get_full_audio()
        
        # Normalize audio to boost low volume
        if len(audio) > 0 and np.abs(audio).max() > 0:
            max_val = np.abs(audio).max()
            if max_val < 0.1:
                target_level = 0.3
                audio = audio * (target_level / max_val)
                print(f"[WS] Audio normalized: boosted by {target_level/max_val:.1f}x")
        
        print(f"[WS] Audio stats: {len(audio)} samples, dtype: {audio.dtype}, range: [{audio.min():.3f}, {audio.max():.3f}]")
        
        if len(audio) < 8000:  # Less than 0.5 seconds
            results["transcription"] = "[صوت قصير جداً]"
            return results
        
        if np.all(audio == 0) or np.abs(audio).max() < 0.001:
            print(f"[WS] WARNING: Audio appears to be silent or corrupted")
            results["transcription"] = "[صوت صامت أو تالف]"
            return results
        
        # ══════════════════════════════════════════════════════════════════
        # 2. PROCESS TURN THROUGH LANGGRAPH ORCHESTRATION
        # ══════════════════════════════════════════════════════════════════
        print(f"[WS] 🚀 Processing turn through LangGraph Orchestration...")
        
        # This runs the full pipeline:
        # memory_load → stt → emotion → rag → llm → tts → memory_save
        state = handler.agent.process_turn(audio)
        
        # Extract results from state
        results["transcription"] = state.get("transcription", "")
        results["response"] = state.get("llm_response", "")
        
        # Get emotion from state
        emotion_result = state.get("emotion") or {"primary_emotion": "neutral", "confidence": 0.5}
        results["emotion"] = emotion_result.get("primary_emotion", "neutral")
        
        # Get audio output
        audio_output = state.get("audio_output")
        if audio_output is not None:
            audio_bytes = audio_output.astype(np.float32).tobytes()
            results["audio_base64"] = base64.b64encode(audio_bytes).decode('utf-8')
        
        print(f"[WS] Transcription: {results['transcription'][:50]}...")
        print(f"[WS] Response: {results['response'][:50]}...")
        print(f"[WS] Emotion: {results['emotion']}")
        
        # ══════════════════════════════════════════════════════════════════
        # 3. EVALUATE SALESPERSON (Rule-based for now)
        # ══════════════════════════════════════════════════════════════════
        evaluation = evaluate_salesperson_turn(results["transcription"], handler.persona)
        results["evaluation"] = evaluation
        
        # ══════════════════════════════════════════════════════════════════
        # 4. GET EMOTION STATE FOR FRONTEND
        # ══════════════════════════════════════════════════════════════════
        emotion_state = handler.get_emotion_state(results["emotion"], evaluation["quality"])
        results["emotion_state"] = emotion_state
        
        # ══════════════════════════════════════════════════════════════════
        # 5. UPDATE HANDLER TURN COUNT
        # ══════════════════════════════════════════════════════════════════
        handler.turn_count = state.get("turn_count", handler.turn_count + 1)
        
        # ══════════════════════════════════════════════════════════════════
        # 6. SAVE TO DATABASE (Additional DB records beyond memory agent)
        # ══════════════════════════════════════════════════════════════════
        # Note: Messages are already saved by memory_save_node in orchestration
        # Here we save additional info like emotion_log with evaluation
        
        try:
            # Save emotion log with evaluation info
            add_emotion_log(db, UUID(handler.session_id), None, emotion_state)
            
            # Update session turn count
            handler.training_session.turn_count = handler.turn_count
            db.commit()
        except Exception as db_error:
            print(f"[WS] DB save warning (non-critical): {db_error}")
        
        # ══════════════════════════════════════════════════════════════════
        # 7. LOG CHECKPOINT INFO
        # ══════════════════════════════════════════════════════════════════
        memory = state.get("memory", {})
        checkpoints = memory.get("checkpoints", [])
        if checkpoints:
            print(f"[WS] 📌 Session has {len(checkpoints)} checkpoint(s)")
        
        # Check if checkpoint was just created (every 5 turns)
        if handler.turn_count > 0 and handler.turn_count % 5 == 0:
            print(f"[WS] 📌 Checkpoint should have been created at turn {handler.turn_count}")
        
    except Exception as e:
        print(f"[WS] Error processing turn: {e}")
        import traceback
        traceback.print_exc()
        results["response"] = "عذراً، حصل مشكلة. ممكن تعيد الكلام؟"
    
    return results


def evaluate_salesperson_turn(text: str, persona: Persona) -> dict:
    """
    Simple rule-based evaluation of salesperson response.
    In production, this would use LLM-as-Judge.
    """
    text_lower = text.lower()
    
    bad_patterns = [
        "مش فاهم", "مش عارف", "مش شغلي",
        "روح", "امشي", "سيبني",
        "غبي", "احمق", "بايخ"
    ]
    
    good_patterns = [
        "أهلا", "مرحبا", "اتفضل",
        "أكيد", "طبعا", "بكل سرور",
        "فاهم", "معاك حق", "نقطة مهمة",
        "اسمحلي", "خليني اشرح", "ممكن اساعدك"
    ]
    
    for pattern in bad_patterns:
        if pattern in text_lower:
            return {
                "quality": "bad",
                "reason": "استخدام كلمات أو لهجة غير مناسبة",
                "suggestion": "حاول تكون أكثر احترافية ولطافة مع العميل"
            }
    
    good_count = sum(1 for pattern in good_patterns if pattern in text_lower)
    
    if good_count >= 2:
        return {
            "quality": "good",
            "reason": "رد مهذب ومحترف",
            "suggestion": None
        }
    elif good_count >= 1:
        return {
            "quality": "neutral",
            "reason": "رد مقبول",
            "suggestion": "ممكن تكون أكثر ودية مع العميل"
        }
    else:
        return {
            "quality": "neutral",
            "reason": "رد محايد",
            "suggestion": "حاول تضيف تحية أو كلمات إيجابية"
        }


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    token: str = Query(...)
):
    """
    WebSocket endpoint for real-time conversation.
    
    NOW USING: LangGraph Orchestration Agent!
    
    Connect: ws://localhost:8000/ws/{session_id}?token={jwt_token}
    
    Messages from client:
    - {"type": "audio", "data": {"audio_base64": "..."}}
    - {"type": "audio_complete", "data": {"audio_base64": "...", "format": "webm"}}
    - {"type": "end_speaking"}
    - {"type": "end_session"}
    - {"type": "ping"}
    
    Messages to client:
    - {"type": "connected", "data": {...}}
    - {"type": "processing", "data": {"status": "started|completed"}}
    - {"type": "transcription", "data": {"text": "..."}}
    - {"type": "emotion", "data": {"emotion": "...", "mood_score": ..., ...}}
    - {"type": "response", "data": {"text": "..."}}
    - {"type": "audio", "data": {"audio_base64": "...", "sample_rate": 24000}}
    - {"type": "evaluation", "data": {"quality": "...", ...}}
    - {"type": "session_ended", "data": {...}}
    - {"type": "error", "data": {"message": "..."}}
    - {"type": "pong"}
    """
    
    # Validate token
    token_data = decode_token(token)
    if not token_data:
        await websocket.close(code=4001, reason="Invalid token")
        return
    
    # Get database session
    db = SessionLocal()
    handler = None
    
    try:
        # Validate session
        training_session = get_session(db, UUID(session_id))
        
        # Verify user owns this session
        if str(training_session.user_id) != token_data.user_id:
            await websocket.close(code=4003, reason="Access denied")
            return
        
        # Verify session is active
        if training_session.status != "active":
            await websocket.close(code=4004, reason="Session not active")
            return
        
        # Get persona
        persona = db.query(Persona).filter(Persona.id == training_session.persona_id).first()
        
        # ══════════════════════════════════════════════════════════════════
        # CREATE HANDLER WITH ORCHESTRATION AGENT
        # ══════════════════════════════════════════════════════════════════
        handler = ConversationHandler(
            session_id=session_id,
            user_id=token_data.user_id,
            training_session=training_session,
            persona=persona
        )
        
        # Connect WebSocket
        await manager.connect(session_id, websocket)
        
        # Send welcome message
        await websocket.send_json({
            "type": "connected",
            "data": {
                "session_id": session_id,
                "persona": {
                    "id": persona.id,
                    "name_ar": persona.name_ar,
                    "name_en": persona.name_en,
                    "difficulty": persona.difficulty
                },
                "message": f"متصل مع {persona.name_ar}",
                "orchestration": "LangGraph"  # Indicate we're using orchestration
            }
        })
        
        # ══════════════════════════════════════════════════════════════════
        # MAIN MESSAGE LOOP
        # ══════════════════════════════════════════════════════════════════
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            print(f"[WS] Received message type: {msg_type}")  # ← ADD THIS

            # ──────────────────────────────────────────────────────────────
            # AUDIO CHUNK (streaming)
            # ──────────────────────────────────────────────────────────────
            if msg_type == "audio":
                audio_data = data.get("data", {}).get("audio_base64", "")
                handler.add_audio_chunk(audio_data)
            
            # ──────────────────────────────────────────────────────────────
            # AUDIO COMPLETE (WebM from browser)
            # ──────────────────────────────────────────────────────────────
            elif msg_type == "audio_complete":
                audio_data = data.get("data", {}).get("audio_base64", "")
                audio_format = data.get("data", {}).get("format", "webm")
                
                if audio_data:
                    try:
                        import tempfile
                        import subprocess
                        import wave
                        
                        # Decode base64
                        audio_bytes = base64.b64decode(audio_data)
                        
                        # Save to temp file
                        with tempfile.NamedTemporaryFile(suffix=f'.{audio_format}', delete=False) as f:
                            f.write(audio_bytes)
                            temp_input = f.name
                        
                        # Convert to WAV using ffmpeg
                        temp_output = temp_input.replace(f'.{audio_format}', '.wav')
                        subprocess.run([
                            'ffmpeg', '-y', '-i', temp_input,
                            '-ar', '16000', '-ac', '1', '-f', 'wav', temp_output
                        ], capture_output=True)
                        
                        # Read WAV file
                        with wave.open(temp_output, 'rb') as wav:
                            frames = wav.readframes(wav.getnframes())
                            audio_array = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                        
                        # Clean up temp files
                        os.unlink(temp_input)
                        os.unlink(temp_output)
                        
                        # Add to buffer
                        handler.audio_buffer = [audio_array]
                        
                    except Exception as e:
                        print(f"[WS] Error converting audio: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "data": {"message": "Audio conversion failed"}
                        })
                        continue
                
                # Process the turn
                if handler.is_processing:
                    continue
                
                handler.is_processing = True
                
                try:
                    await websocket.send_json({
                        "type": "processing",
                        "data": {"status": "started"}
                    })
                    
                    # ══════════════════════════════════════════════════════
                    # PROCESS WITH LANGGRAPH ORCHESTRATION
                    # ══════════════════════════════════════════════════════
                    if handler.agent.config.enable_streaming:
                        results = await process_turn_streaming(handler, db, websocket)
                    else:
                        results = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: process_turn_with_orchestration_sync(handler, db)
                        )
                    
                    # Send results to client
                    await _send_turn_results(websocket, results)
                    
                    await websocket.send_json({
                        "type": "processing",
                        "data": {"status": "completed"}
                    })
                    
                finally:
                    handler.is_processing = False
                
                continue
            
            # ──────────────────────────────────────────────────────────────
            # END SPEAKING (process buffered audio)
            # ──────────────────────────────────────────────────────────────
            elif msg_type == "end_speaking":
                if handler.is_processing:
                    continue
                
                handler.is_processing = True
                
                try:
                    await websocket.send_json({
                        "type": "processing",
                        "data": {"status": "started"}
                    })
                    
                    # ══════════════════════════════════════════════════════
                    # PROCESS WITH LANGGRAPH ORCHESTRATION
                    # ══════════════════════════════════════════════════════
                    if handler.agent.config.enable_streaming:
                        results = await process_turn_streaming(handler, db, websocket)
                    else:
                        results = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: process_turn_with_orchestration_sync(handler, db)
                        ) 
                    
                    
                    # Send results to client
                    await _send_turn_results(websocket, results)
                    
                    await websocket.send_json({
                        "type": "processing",
                        "data": {"status": "completed"}
                    })
                    
                finally:
                    handler.is_processing = False
            
            # ──────────────────────────────────────────────────────────────
            # END SESSION
            # ──────────────────────────────────────────────────────────────
            elif msg_type == "end_session":
                # End orchestration session
                orchestration_summary = handler.end_orchestration_session()
                
                # End database session
                end_session(db, UUID(session_id))
                
                await websocket.send_json({
                    "type": "session_ended",
                    "data": {
                        "total_turns": handler.turn_count,
                        "final_mood": handler.customer_mood,
                        "message": "تم إنهاء الجلسة",
                        "orchestration_summary": orchestration_summary
                    }
                })
                break
            
            # ──────────────────────────────────────────────────────────────
            # PING/PONG
            # ──────────────────────────────────────────────────────────────
            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})
    
    except WebSocketDisconnect:
        print(f"[WS] Client disconnected: {session_id}")
    
    except Exception as e:
        print(f"[WS] Error: {e}")
        import traceback
        traceback.print_exc()
        
        try:
            await websocket.send_json({
                "type": "error",
                "data": {"message": str(e)}
            })
        except:
            pass
    
    finally:
        # Clean up
        if handler:
            handler.end_orchestration_session()
        manager.disconnect(session_id)
        db.close()


def process_turn_with_orchestration_sync(handler: ConversationHandler, db: Session) -> dict:
    """
    Synchronous wrapper for process_turn_with_orchestration.
    Used with run_in_executor for async compatibility.
    """
    import asyncio
    
    # Create a new event loop for this thread if needed
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Run the async function
    return asyncio.run(process_turn_with_orchestration(handler, db))


# ══════════════════════════════════════════════════════════════════════════════
# UPDATED _send_turn_results (replace existing)
# Skips transcription since streaming already sent it
# ══════════════════════════════════════════════════════════════════════════════

# PATCH for backend/routers/websocket.py
# Replace the existing _send_turn_results function with this one

async def _send_turn_results(websocket, results: dict):
    """Send turn results to frontend."""
    
    # ══════════════════════════════════════════════════════════════════
    # 1. SEND TRANSCRIPTION (what the salesperson said)
    # ══════════════════════════════════════════════════════════════════
    if results.get("transcription"):
        await websocket.send_json({
            "type": "transcription",
            "data": {"text": results["transcription"]}
        })

    # ══════════════════════════════════════════════════════════════════
    # 2. SEND EMOTION STATE
    # ══════════════════════════════════════════════════════════════════
    if results.get("emotion_state"):
        await websocket.send_json({
            "type": "emotion",
            "data": {
                "emotion": results["emotion_state"].customer_emotion,
                "mood_score": results["emotion_state"].customer_mood_score,
                "risk_level": results["emotion_state"].risk_level,
                "trend": results["emotion_state"].emotion_trend,
                "tip": results["emotion_state"].tip
            }
        })

    # ══════════════════════════════════════════════════════════════════
    # 3. SEND EVALUATION
    # ══════════════════════════════════════════════════════════════════
    await websocket.send_json({
        "type": "evaluation",
        "data": results["evaluation"]
    })

    # ══════════════════════════════════════════════════════════════════
    # 4. SEND RESPONSE TEXT (what the customer said)
    # ══════════════════════════════════════════════════════════════════
    await websocket.send_json({
        "type": "response",
        "data": {"text": results["response"]}
    })

    # ══════════════════════════════════════════════════════════════════
    # 5. SEND AUDIO
    # ══════════════════════════════════════════════════════════════════
    if results.get("audio_base64"):
        await websocket.send_json({
            "type": "audio",
            "data": {
                "audio_base64": results["audio_base64"],
                "sample_rate": 24000
            }
        })