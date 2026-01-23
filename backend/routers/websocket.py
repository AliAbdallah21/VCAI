# backend/routers/websocket.py
"""
WebSocket endpoint for real-time conversation.

This handles:
- Audio streaming from salesperson
- Real-time transcription
- Emotion detection & updates
- AI response generation
- TTS audio streaming back
- Real-time evaluation tips
"""

import os

from backend.app.utils import audio
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
from backend.services import get_session, add_message, add_emotion_log, decode_token
from backend.schemas import MessageCreate, EmotionState
from backend.models import Session as TrainingSession, Persona

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
    """Handles a single conversation session."""
    
    def __init__(self, session_id: str, training_session: TrainingSession, persona: Persona):
        self.session_id = session_id
        self.training_session = training_session
        self.persona = persona
        self.turn_count = training_session.turn_count or 0
        self.audio_buffer = []
        self.is_processing = False
        
        # Emotion tracking
        self.customer_mood = 50  # Start neutral (0-100 scale, 50 = neutral)
        self.emotion_history = []
    
    # def add_audio_chunk(self, audio_base64: str):
    #     """Add audio chunk to buffer."""
    #     try:
    #         audio_bytes = base64.b64decode(audio_base64)
    #         audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
    #         self.audio_buffer.append(audio_array)
    #     except Exception as e:
    #         print(f"[WS] Error decoding audio: {e}")
    def add_audio_chunk(self, audio_base64: str):
        try:
            # Decode base64
            audio_bytes = base64.b64decode(audio_base64)
            
            # Convert bytes to numpy array (Little Endian Float32)
            audio_array = np.frombuffer(audio_bytes, dtype='<f4')  # '<f4' = little-endian float32
            
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
        # Mood change based on salesperson quality
        mood_changes = {
            "good": 5,      # Good response improves mood
            "neutral": 0,   # Neutral keeps same
            "bad": -10      # Bad response worsens mood
        }
        
        # Adjust by persona sensitivity
        sensitivity_multiplier = self.persona.emotion_sensitivity / 50  # 1.0 at 50
        
        change = mood_changes.get(salesperson_quality, 0) * sensitivity_multiplier
        
        # Apply change with bounds
        self.customer_mood = max(0, min(100, self.customer_mood + change))
        
        return self.customer_mood
    
    def get_emotion_state(self, emotion: str, quality: str) -> EmotionState:
        """Get current emotion state for frontend."""
        # Update mood
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
        
        # Generate tip based on situation
        tip = self._generate_tip(emotion, risk_level, quality)
        
        return EmotionState(
            customer_emotion=emotion,
            customer_mood_score=int(mood * 2 - 100),  # Convert 0-100 to -100 to +100
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
            ("neutral", "low"): None,  # No tip needed
            ("happy", "low"): "👍 كمل كده، العميل مبسوط!",
        }
        
        # Check specific combinations
        tip = tips.get((emotion, risk_level))
        if tip:
            return tip
        
        # Quality-based tips
        if quality == "bad":
            return "💡 حاول تكون أكثر احترافية في ردك"
        
        return None


async def process_turn(
    handler: ConversationHandler,
    db: Session
) -> dict:
    """
    Process a complete turn:
    1. STT - Transcribe audio
    2. Emotion - Detect emotion
    3. LLM - Generate response
    4. Evaluate - Rate the salesperson
    5. TTS - Generate audio
    """
    from backend.config import get_settings
    settings = get_settings()
    
    results = {
        "transcription": "",
        "emotion": "neutral",
        "response": "",
        "audio_base64": "",
        "evaluation": {"quality": "neutral", "reason": "", "suggestion": ""},
        "emotion_state": None
    }
    
    try:
        # Get audio
        audio = handler.get_full_audio()
        
        # Normalize audio to boost low volume
        if len(audio) > 0 and np.abs(audio).max() > 0:
            max_val = np.abs(audio).max()
            if max_val < 0.1:  # If audio is quiet
                target_level = 0.3
                audio = audio * (target_level / max_val)
                print(f"[WS] Audio normalized: boosted by {target_level/max_val:.1f}x")
        
        # Add this validation:
        print(f"[WS] Audio stats: {len(audio)} samples, dtype: {audio.dtype}, range: [{audio.min():.3f}, {audio.max():.3f}]")


        
        if len(audio) < 8000:  # Less than 0.5 seconds
            results["transcription"] = "[صوت قصير جداً]"
            return results
        
        # Also check if audio is all zeros or garbage
        if np.all(audio == 0) or np.abs(audio).max() < 0.001:
            print(f"[WS] WARNING: Audio appears to be silent or corrupted")
            results["transcription"] = "[صوت صامت أو تالف]"
            return results

        # ════════════════════════════════════════════════════════════════
        # 1. STT - Transcribe
        # ══════════════════════════════════════════════════════════════════
        print(f"[WS] Processing STT...")
        
        if settings.use_mocks:
            from orchestration.mocks import mock_llm
            results["transcription"] = "هذا نص تجريبي من الميكروفون"
        else:
            from stt.realtime_stt import transcribe_audio
            results["transcription"] = transcribe_audio(audio)
        
        print(f"[WS] Transcription: {results['transcription'][:50]}...")
        
        # ══════════════════════════════════════════════════════════════════
        # 2. Emotion Detection
        # ══════════════════════════════════════════════════════════════════
        print(f"[WS] Detecting emotion...")
        
        if settings.use_mocks:
            from orchestration.mocks import detect_emotion
            emotion_result = detect_emotion(results["transcription"], audio)
        else:
            # Use mock for now until emotion module is ready
            from orchestration.mocks import detect_emotion
            emotion_result = detect_emotion(results["transcription"], audio)
        
        results["emotion"] = emotion_result["primary_emotion"]
        print(f"[WS] Emotion: {results['emotion']}")
        
        # ══════════════════════════════════════════════════════════════════
        # 3. Real-time Evaluation (simple for now)
        # ══════════════════════════════════════════════════════════════════
        print(f"[WS] Evaluating response...")
        
        evaluation = evaluate_salesperson_turn(
            results["transcription"],
            handler.persona
        )
        results["evaluation"] = evaluation
        print(f"[WS] Quality: {evaluation['quality']}")
        
        # ══════════════════════════════════════════════════════════════════
        # 4. Get Emotion State for Frontend
        # ══════════════════════════════════════════════════════════════════
        emotion_state = handler.get_emotion_state(
            results["emotion"],
            evaluation["quality"]
        )
        results["emotion_state"] = emotion_state
        
        # ══════════════════════════════════════════════════════════════════
        # 5. LLM - Generate Response
        # ══════════════════════════════════════════════════════════════════
        print(f"[WS] Generating response...")
        
        if settings.use_mocks:
            from orchestration.mocks import generate_response, retrieve_context, get_session_memory
            
            memory = get_session_memory(str(handler.session_id))
            rag_context = retrieve_context(results["transcription"])
            
            results["response"] = generate_response(
                customer_text=results["transcription"],
                emotion=emotion_result,
                emotional_context={
                    "current": emotion_result,
                    "trend": emotion_state.emotion_trend,
                    "recommendation": "be_professional",
                    "risk_level": emotion_state.risk_level
                },
                persona={
                    "id": handler.persona.id,
                    "name": handler.persona.name_ar,
                    "personality_prompt": handler.persona.personality_prompt
                },
                memory=memory,
                rag_context=rag_context
            )
        else:
            # Use orchestration agent when ready
            from llm.agent import generate_response
            from orchestration.mocks import retrieve_context
            from memory.agent import get_session_memory
            
            
            memory = get_session_memory(str(handler.session_id))
            rag_context = retrieve_context(results["transcription"])
            
            results["response"] = generate_response(
                customer_text=results["transcription"],
                emotion=emotion_result,
                emotional_context={
                    "current": emotion_result,
                    "trend": emotion_state.emotion_trend,
                    "recommendation": "be_professional",
                    "risk_level": emotion_state.risk_level
                },
                persona={
                    "id": handler.persona.id,
                    "name": handler.persona.name_ar,
                    "personality_prompt": handler.persona.personality_prompt
                },
                memory=memory,
                rag_context=rag_context
            )
        
        print(f"[WS] Response: {results['response'][:50]}...")
        
        # ══════════════════════════════════════════════════════════════════
        # 6. TTS - Generate Audio
        # ══════════════════════════════════════════════════════════════════
        print(f"[WS] Generating TTS...")
        
        if settings.use_mocks:
            from orchestration.mocks import text_to_speech
            audio_output = text_to_speech(
                results["response"],
                handler.persona.voice_id or "egyptian_male_01",
                results["emotion"]
            )
        else:
            # Use mock for now until TTS module is ready
            from orchestration.mocks import text_to_speech
            audio_output = text_to_speech(
                results["response"],
                handler.persona.voice_id or "egyptian_male_01",
                results["emotion"]
            )
        
        # Convert to base64
        audio_bytes = audio_output.astype(np.float32).tobytes()
        results["audio_base64"] = base64.b64encode(audio_bytes).decode('utf-8')
        
        print(f"[WS] TTS complete: {len(audio_output)} samples")
        
        # ══════════════════════════════════════════════════════════════════
        # 7. Save to Database
        # ══════════════════════════════════════════════════════════════════
        handler.turn_count += 1
        
        # Save salesperson message
        salesperson_msg = MessageCreate(
            turn_number=handler.turn_count,
            speaker="salesperson",
            text=results["transcription"],
            detected_emotion=results["emotion"],
            emotion_confidence=emotion_result.get("confidence", 0.5),
            response_quality=evaluation["quality"],
            quality_reason=evaluation["reason"],
            suggestion=evaluation["suggestion"]
        )
        add_message(db, UUID(handler.session_id), salesperson_msg)
        
        # Save customer message
        customer_msg = MessageCreate(
            turn_number=handler.turn_count,
            speaker="customer",
            text=results["response"]
        )
        add_message(db, UUID(handler.session_id), customer_msg)
        
        # Save emotion log
        add_emotion_log(db, UUID(handler.session_id), None, emotion_state)
        
        db.commit()
        
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
    
    # Bad patterns
    bad_patterns = [
        "مش فاهم", "مش عارف", "مش شغلي",
        "روح", "امشي", "سيبني",
        "غبي", "احمق", "بايخ"
    ]
    
    # Good patterns
    good_patterns = [
        "أهلا", "مرحبا", "اتفضل",
        "أكيد", "طبعا", "بكل سرور",
        "فاهم", "معاك حق", "نقطة مهمة",
        "اسمحلي", "خليني اشرح", "ممكن اساعدك"
    ]
    
    # Check for bad patterns
    for pattern in bad_patterns:
        if pattern in text_lower:
            return {
                "quality": "bad",
                "reason": "استخدام كلمات أو لهجة غير مناسبة",
                "suggestion": "حاول تكون أكثر احترافية ولطافة مع العميل"
            }
    
    # Check for good patterns
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
    
    Connect: ws://localhost:8000/ws/{session_id}?token={jwt_token}
    
    Messages from client:
    - {"type": "audio", "data": {"audio_base64": "..."}}
    - {"type": "end_speaking"}
    - {"type": "end_session"}
    
    Messages to client:
    - {"type": "transcription", "data": {"text": "..."}}
    - {"type": "emotion", "data": {"emotion": "...", "mood_score": ..., "risk_level": "...", "tip": "..."}}
    - {"type": "response", "data": {"text": "..."}}
    - {"type": "audio", "data": {"audio_base64": "...", "sample_rate": 22050}}
    - {"type": "evaluation", "data": {"quality": "...", "reason": "...", "suggestion": "..."}}
    - {"type": "error", "data": {"message": "..."}}
    """
    
    # Validate token
    token_data = decode_token(token)
    if not token_data:
        await websocket.close(code=4001, reason="Invalid token")
        return
    
    # Get database session
    db = SessionLocal()
    
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
        
        # Create handler
        handler = ConversationHandler(session_id, training_session, persona)
        
        # Connect
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
                "message": f"متصل مع {persona.name_ar}"
            }
        })
        
        # Main loop
        while True:
            # Receive message
            data = await websocket.receive_json()
            msg_type = data.get("type")
            
            if msg_type == "audio":
                # Add audio chunk to buffer
                audio_data = data.get("data", {}).get("audio_base64", "")
                handler.add_audio_chunk(audio_data)
            
            elif msg_type == "audio_complete":
                # Handle complete audio file (WebM format from browser)
                audio_data = data.get("data", {}).get("audio_base64", "")
                audio_format = data.get("data", {}).get("format", "webm")
                
                if audio_data:
                    try:
                        import tempfile
                        import subprocess
                        
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
                        import wave
                        with wave.open(temp_output, 'rb') as wav:
                            frames = wav.readframes(wav.getnframes())
                            audio_array = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                        
                        # Clean up temp files
                        import os
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
                
                # Now process the turn (fall through to end_speaking logic)
                if handler.is_processing:
                    continue
                
                handler.is_processing = True
                
                try:
                    await websocket.send_json({
                        "type": "processing",
                        "data": {"status": "started"}
                    })
                    
                    results = await process_turn(handler, db)
                    
                    await websocket.send_json({
                        "type": "transcription",
                        "data": {"text": results["transcription"]}
                    })
                    
                    if results["emotion_state"]:
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
                    
                    await websocket.send_json({
                        "type": "evaluation",
                        "data": results["evaluation"]
                    })
                    
                    await websocket.send_json({
                        "type": "response",
                        "data": {"text": results["response"]}
                    })
                    
                    await websocket.send_json({
                        "type": "audio",
                        "data": {
                            "audio_base64": results["audio_base64"],
                            "sample_rate": 22050
                        }
                    })
                    
                    await websocket.send_json({
                        "type": "processing",
                        "data": {"status": "completed"}
                    })
                    
                finally:
                    handler.is_processing = False
                
                continue

            elif msg_type == "end_speaking":
                # Process the complete turn
                if handler.is_processing:
                    continue
                
                handler.is_processing = True
                
                try:
                    # Send processing indicator
                    await websocket.send_json({
                        "type": "processing",
                        "data": {"status": "started"}
                    })
                    
                    # Process turn
                    results = await process_turn(handler, db)
                    
                    # Send transcription
                    await websocket.send_json({
                        "type": "transcription",
                        "data": {"text": results["transcription"]}
                    })
                    
                    # Send emotion state
                    if results["emotion_state"]:
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
                    
                    # Send evaluation
                    await websocket.send_json({
                        "type": "evaluation",
                        "data": results["evaluation"]
                    })
                    
                    # Send response text
                    await websocket.send_json({
                        "type": "response",
                        "data": {"text": results["response"]}
                    })
                    
                    # Send audio
                    await websocket.send_json({
                        "type": "audio",
                        "data": {
                            "audio_base64": results["audio_base64"],
                            "sample_rate": 22050
                        }
                    })
                    
                    # Send processing complete
                    await websocket.send_json({
                        "type": "processing",
                        "data": {"status": "completed"}
                    })
                    
                finally:
                    handler.is_processing = False
            
            elif msg_type == "end_session":
                # End the session
                from backend.services import end_session
                end_session(db, UUID(session_id))
                
                await websocket.send_json({
                    "type": "session_ended",
                    "data": {
                        "total_turns": handler.turn_count,
                        "final_mood": handler.customer_mood,
                        "message": "تم إنهاء الجلسة"
                    }
                })
                break
            
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
        manager.disconnect(session_id)
        db.close()

