# backend/routers/ws_handler.py
"""
ConversationHandler — per-session stateful object wrapping the OrchestrationAgent.
"""

import numpy as np
from typing import Optional

from backend.schemas import EmotionState
from backend.models import Session as TrainingSession, Persona


class ConversationHandler:
    """
    Holds all mutable state for one active WebSocket session:
    audio buffer, turn count, customer mood, and the OrchestrationAgent.
    """

    def __init__(
        self,
        session_id: str,
        user_id: str,
        training_session: TrainingSession,
        persona: Persona,
    ):
        self.session_id = session_id
        self.user_id = user_id
        self.training_session = training_session
        self.persona = persona
        self.turn_count = training_session.turn_count or 0
        self.audio_buffer: list[np.ndarray] = []
        self.is_processing = False

        # Raw audio of the salesperson's latest turn (for replay storage).
        # Cleared each turn by ws_pipeline once written to disk.
        self.pending_salesperson_audio_bytes: bytes | None = None
        self.pending_salesperson_audio_format: str = "webm"

        # Customer mood tracking (0–100, 50 = neutral)
        self.customer_mood = 50
        self.emotion_history: list[float] = []

        self._init_agent()
        self._prewarm_tts()

    # ──────────────────────────────────────────────────────────────────────
    # Agent lifecycle
    # ──────────────────────────────────────────────────────────────────────

    def _init_agent(self) -> None:
        from llm.agent import USE_OPENROUTER as _USE_OPENROUTER
        from orchestration import OrchestrationAgent

        self.agent = OrchestrationAgent(
            use_mocks=False,
            verbose=True,
            enable_streaming=_USE_OPENROUTER,
        )

        # difficulty is the SESSION's chosen difficulty (set at session setup),
        # not the persona's inherent one — this is what makes any persona
        # playable at any difficulty. Fall back to the persona's difficulty if
        # the session somehow has none.
        session_difficulty = (
            getattr(self.training_session, "difficulty", None)
            or self.persona.difficulty
            or "medium"
        )

        persona_dict = {
            "id":                  self.persona.id,
            "name":                self.persona.name_ar,
            "name_en":             self.persona.name_en,
            "description":         getattr(self.persona, "description_ar", ""),
            "personality_prompt":  self.persona.personality_prompt,
            "voice_id":            self.persona.voice_id,
            "default_emotion":     "neutral",
            "difficulty":          session_difficulty,
            "traits":              self.persona.traits or [],
            "avatar_url":          getattr(self.persona, "avatar_url", None),
            # Buyer scenario for this session (budget / timeline / must-haves /
            # deal-breakers). Rides on the persona dict so it reaches the LLM
            # layer without extra plumbing. Phase 4 makes the prompt act on it.
            "scenario":            getattr(self.training_session, "scenario", None),
        }

        self.agent.start_session(
            session_id=self.session_id,
            user_id=self.user_id,
            persona_id=self.persona.id,
            persona_dict=persona_dict,
        )
        print(f"[WS] Orchestration session started for {self.session_id}")

    def _prewarm_tts(self) -> None:
        """
        Fire a dummy TTS synthesis in a background thread so CUDA kernels are
        hot before the user's first message arrives.  First real TTS call will
        then be ~1s instead of ~3s (no cold-start overhead).
        """
        from threading import Thread

        def _warm():
            try:
                from tts.agent import text_to_speech
                text_to_speech("مرحبا", voice_id="default", emotion="neutral")
                print("[WS] TTS pre-warm complete")
            except Exception as e:
                print(f"[WS] TTS pre-warm failed (non-fatal): {e}")

        Thread(target=_warm, daemon=True).start()
        print("[WS] TTS pre-warm started in background")

    def end_orchestration_session(self) -> dict:
        try:
            if self.agent:
                summary = self.agent.end_session()
                print(f"[WS] Orchestration session ended: {summary}")
                return summary or {}
        except Exception as e:
            print(f"[WS] Error ending orchestration session: {e}")
        return {}

    # ──────────────────────────────────────────────────────────────────────
    # Audio buffer
    # ──────────────────────────────────────────────────────────────────────

    def add_audio_chunk(self, audio_base64: str) -> None:
        import base64
        try:
            audio_bytes = base64.b64decode(audio_base64)
            audio_array = np.frombuffer(audio_bytes, dtype="<f4")
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            self.audio_buffer.append(audio_array)
        except Exception as e:
            print(f"[WS] Error decoding audio chunk: {e}")

    def get_full_audio(self) -> np.ndarray:
        if not self.audio_buffer:
            return np.array([], dtype=np.float32)
        full = np.concatenate(self.audio_buffer)
        self.audio_buffer = []
        return full

    # ──────────────────────────────────────────────────────────────────────
    # Emotion / mood
    # ──────────────────────────────────────────────────────────────────────

    def update_customer_mood(self, emotion: str, quality: str) -> float:
        mood_delta = {"good": 5, "neutral": 0, "bad": -10}
        sensitivity = (self.persona.emotion_sensitivity or 50) / 50
        self.customer_mood = max(
            0, min(100, self.customer_mood + mood_delta.get(quality, 0) * sensitivity)
        )
        return self.customer_mood

    def get_emotion_state(self, emotion: str, quality: str) -> EmotionState:
        mood = self.update_customer_mood(emotion, quality)

        if mood < 30:
            risk = "high"
        elif mood < 50:
            risk = "medium"
        else:
            risk = "low"

        if len(self.emotion_history) >= 2:
            avg = sum(self.emotion_history[-3:]) / len(self.emotion_history[-3:])
            trend = "improving" if mood > avg + 5 else "worsening" if mood < avg - 5 else "stable"
        else:
            trend = "stable"

        self.emotion_history.append(mood)
        tip = self._make_tip(emotion, risk, quality)

        return EmotionState(
            customer_emotion=emotion,
            customer_mood_score=int(mood * 2 - 100),
            risk_level=risk,
            emotion_trend=trend,
            tip=tip,
        )

    def _make_tip(self, emotion: str, risk: str, quality: str) -> Optional[str]:
        tips = {
            ("angry",     "high"):   "⚠️ العميل زعلان! حاول تهديه وتسمعله",
            ("angry",     "medium"): "العميل مش مبسوط، خد بالك من لهجتك",
            ("frustrated","high"):   "العميل محبط، حاول تفهم مشكلته",
            ("sad",       "medium"): "العميل قلقان، طمنه وادي له معلومات واضحة",
            ("neutral",   "low"):    None,
            ("happy",     "low"):    "👍 كمل كده، العميل مبسوط!",
        }
        tip = tips.get((emotion, risk))
        if tip:
            return tip
        if quality == "bad":
            return "💡 حاول تكون أكثر احترافية في ردك"
        return None
