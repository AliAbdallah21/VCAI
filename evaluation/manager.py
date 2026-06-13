# evaluation/manager.py
"""
Evaluation Manager - Main Orchestrator
Author: Menna Khaled

This is the main entry point for the VCAI Evaluation System.
It orchestrates the evaluation process by calling Ismail's LangGraph pipeline.
"""

from typing import Optional
import logging
from datetime import datetime

from evaluation.state import (
    EvaluationState,
    create_initial_state,
    mark_completed,
    mark_failed
)
from evaluation.schemas.report_schema import FinalReport, QuickStats
from evaluation.config import settings


logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# LLM Wrapper for Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

import os
import requests


class EvaluationLLMError(RuntimeError):
    """Retryable LLM failure (empty content / no choices) — triggers model fallback."""


class EvaluationLLMWrapper:
    """
    Wrapper using OpenRouter for evaluation.
    """
    
    # Models tried in order when the primary fails with 4xx.
    # All confirmed available on OpenRouter and strong enough for evaluation.
    FALLBACK_MODELS = [
        "google/gemini-2.5-pro",           # Best: strong JSON, Arabic, ~$1-2.50/M in
        "deepseek/deepseek-chat",           # V3: excellent JSON, very cheap
        "google/gemini-2.5-flash",          # Fast, cheap, decent quality
        "google/gemini-2.0-flash-exp:free", # Free tier last resort
    ]

    def __init__(self):
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is required for evaluation. "
                "Set it in your .env file or shell environment."
            )
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        # EVAL_LLM_MODEL overrides OPENROUTER_MODEL specifically for evaluation.
        # Default: anthropic/claude-3.5-sonnet (confirmed working on OpenRouter).
        self.model = (
            os.environ.get("EVAL_LLM_MODEL")
            or os.environ.get("OPENROUTER_MODEL")
            or "anthropic/claude-3.5-sonnet"
        )
        # Output token cap. The synthesizer must emit a full FinalReport
        # (all skills + 6 checkpoints + turn-by-turn feedback) in Arabic,
        # which easily exceeds 8k tokens. The old 8192 cap caused silent
        # truncation: the LLM stopped mid-JSON, json_repair "recovered" it
        # by closing brackets, and every field after the cut fell back to
        # schema defaults — yielding a report with an empty executive
        # summary, no strengths/improvements, and only partial checkpoints.
        try:
            self.max_tokens = int(os.environ.get("EVAL_MAX_TOKENS", "32000"))
        except ValueError:
            self.max_tokens = 32000
        print(f"[LLM] Evaluation LLM initialised — model: {self.model}  max_tokens: {self.max_tokens}")

    def _call_openrouter(self, model: str, system_prompt: str, user_prompt: str) -> str:
        """Make a single blocking call to OpenRouter and return the text content."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "temperature": 0.3,
            "max_tokens": self.max_tokens,
            # Cap reasoning effort. Gemini 2.5 Pro is a thinking model; on the
            # large synthesizer prompt its reasoning tokens can consume the
            # whole output budget, so OpenRouter returns empty content with
            # finish_reason=length. Synthesis is structured JSON formatting,
            # not deep reasoning — low effort is appropriate. Non-reasoning
            # models (e.g. DeepSeek V3) ignore this field.
            "reasoning": {"effort": "low"},
        }
        print(f"[LLM] → POST {self.base_url}  model={model}  prompt_chars={len(user_prompt)}")
        response = requests.post(
            self.base_url, json=payload, headers=headers,
            timeout=(15, 180),
        )
        if not response.ok:
            try:
                err_body = response.json()
            except Exception:
                err_body = response.text[:600]
            print(f"[LLM] ✗ HTTP {response.status_code} from OpenRouter: {err_body}")
            response.raise_for_status()

        data = response.json()
        if not data.get("choices"):
            # HTTP 200 but an error envelope / no choices — retryable.
            raise EvaluationLLMError(
                f"OpenRouter returned no choices (model={model}): {str(data)[:300]}"
            )
        choice = data["choices"][0]
        message = choice.get("message") or {}
        content = message.get("content")
        finish_reason = choice.get("finish_reason") or choice.get("native_finish_reason")

        # Content can be null/empty even on HTTP 200 — a reasoning model that
        # spent its whole budget thinking, or a refusal. Guard before len()
        # and raise a retryable error so generate() falls back to another
        # model instead of crashing downstream on len(None).
        if not content or not str(content).strip():
            reasoning = message.get("reasoning") or ""
            print(f"[LLM] ✗ empty content  model={model}  finish={finish_reason}  "
                  f"reasoning_chars={len(reasoning)}")
            raise EvaluationLLMError(
                f"OpenRouter returned empty content "
                f"(model={model}, finish_reason={finish_reason})"
            )

        print(f"[LLM] ← response  model={model}  chars={len(content)}  finish={finish_reason}")
        if finish_reason == "length":
            # Output hit the token cap and was cut off mid-JSON. Returning it
            # would let json_repair silently backfill the missing half of the
            # report with empty schema defaults. Fail loudly instead so the
            # truncation is visible rather than producing a hollow report.
            raise RuntimeError(
                f"OpenRouter response truncated (finish_reason=length) at "
                f"max_tokens={self.max_tokens}. Raise EVAL_MAX_TOKENS."
            )
        return content

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        """Generate response using OpenRouter, with model fallback on 4xx."""
        models_to_try = [self.model] + [m for m in self.FALLBACK_MODELS if m != self.model]
        last_exc = None
        for model in models_to_try:
            try:
                return self._call_openrouter(model, system_prompt, user_prompt)
            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else 0
                if status in (404, 422, 400):
                    print(f"[LLM] Model {model!r} rejected (HTTP {status}) — trying next fallback")
                    last_exc = exc
                    continue
                raise  # non-retryable error (401, 429, 500, …)
            except EvaluationLLMError as exc:
                # Empty content / no choices — another model may succeed.
                print(f"[LLM] Model {model!r} returned no usable content — trying next fallback")
                last_exc = exc
                continue
        raise RuntimeError(
            f"All OpenRouter models failed. Last error: {last_exc}"
        ) from last_exc


# Singleton LLM wrapper
_llm_wrapper = None

def get_evaluation_llm() -> EvaluationLLMWrapper:
    """Get or create the evaluation LLM wrapper."""
    global _llm_wrapper
    if _llm_wrapper is None:
        _llm_wrapper = EvaluationLLMWrapper()
    return _llm_wrapper


# ═══════════════════════════════════════════════════════════════════════════════
# Data Gathering Functions
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_duration(session, messages: list) -> int:
    """Return actual conversation time (first msg → last msg), not idle wait time."""
    from datetime import timezone as _tz
    msgs_with_ts = [m for m in messages if getattr(m, 'created_at', None)]
    if len(msgs_with_ts) >= 2:
        t0 = msgs_with_ts[0].created_at
        t1 = msgs_with_ts[-1].created_at
        if t0.tzinfo is None:
            t0 = t0.replace(tzinfo=_tz.utc)
        if t1.tzinfo is None:
            t1 = t1.replace(tzinfo=_tz.utc)
        return max(0, int((t1 - t0).total_seconds()))
    if session.duration_seconds:
        return session.duration_seconds
    if session.started_at and session.ended_at:
        return max(0, int((session.ended_at - session.started_at).total_seconds()))
    return 0


def gather_evaluation_inputs_db_only(session_id: str) -> dict:
    """
    Gather ONLY database data (transcript, emotions, session_info).
    Does NOT run the fact-check. Call _gather_fact_check_context separately.
    Used by run_evaluation_background to emit granular progress stages.
    """
    from backend.database import get_db_context
    from backend.models import Session as TrainingSession, Message, EmotionLog
    from backend.models.persona import Persona as PersonaModel
    from uuid import UUID

    logger.info(f"[MANAGER] DB-only gather for session {session_id}")

    with get_db_context() as db:
        session = db.query(TrainingSession).filter(
            TrainingSession.id == UUID(session_id)
        ).first()
        if not session:
            raise ValueError(f"Session {session_id} not found")

        messages = db.query(Message).filter(
            Message.session_id == UUID(session_id)
        ).order_by(Message.turn_number.asc()).all()

        transcript = [
            {
                "turn_number": msg.turn_number,
                "speaker": msg.speaker,
                "text": msg.text,
                "timestamp": msg.created_at.isoformat() if msg.created_at else None,
                "detected_emotion": getattr(msg, 'detected_emotion', None),
                "emotion_confidence": None,
            }
            for msg in messages
        ]

        emotion_logs = db.query(EmotionLog).filter(
            EmotionLog.session_id == UUID(session_id)
        ).order_by(EmotionLog.id.asc()).all()

        emotion_log = [
            {
                "turn_number": i + 1,
                "emotion": log.customer_emotion,
                "confidence": 1.0,
                "mood_score": log.customer_mood_score,
                "risk_level": log.risk_level,
                "trend": log.emotion_trend,
            }
            for i, log in enumerate(emotion_logs)
        ]

        duration_seconds = _compute_duration(session, messages)

        persona = db.query(PersonaModel).filter(
            PersonaModel.id == session.persona_id
        ).first()
        persona_name = (persona.name_ar or persona.name_en) if persona else session.persona_id
        persona_difficulty = persona.difficulty if persona else "medium"

        session_info = {
            "session_id": str(session.id),
            "user_id": str(session.user_id),
            "persona_id": session.persona_id,
            "persona_name": persona_name,
            "persona_difficulty": persona_difficulty,
            "difficulty": getattr(session, "difficulty", None) or persona_difficulty,
            "scenario": getattr(session, "scenario", None),
            "started_at": session.started_at.isoformat() if session.started_at else None,
            "ended_at": session.ended_at.isoformat() if session.ended_at else None,
            "duration_seconds": duration_seconds,
        }

    logger.info(f"[MANAGER] DB gather done: {len(transcript)} messages, {len(emotion_log)} emotions")
    return {
        "transcript": transcript,
        "emotion_log": emotion_log,
        "session_info": session_info,
    }


def gather_evaluation_inputs(session_id: str) -> dict:
    """
    Gather all inputs needed for evaluation from the database.
    
    Returns:
        dict with: transcript, emotion_log, rag_context, session_info
    """
    from backend.database import get_db_context
    from backend.models import Session as TrainingSession, Message, EmotionLog
    from backend.models.persona import Persona as PersonaModel
    from uuid import UUID
    
    logger.info(f"[MANAGER] Gathering inputs for session {session_id}")
    
    with get_db_context() as db:
        # Get session
        session = db.query(TrainingSession).filter(
            TrainingSession.id == UUID(session_id)
        ).first()
        
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Get messages (transcript)
        messages = db.query(Message).filter(
            Message.session_id == UUID(session_id)
        ).order_by(Message.turn_number.asc()).all()
        
        transcript = []
        for msg in messages:
            transcript.append({
                "turn_number": msg.turn_number,
                "speaker": msg.speaker,
                "text": msg.text,
                "timestamp": msg.created_at.isoformat() if msg.created_at else None,
                "detected_emotion": getattr(msg, 'detected_emotion', None),
                "emotion_confidence": None,
            })
        
        # Get emotion logs
        emotion_logs = db.query(EmotionLog).filter(
            EmotionLog.session_id == UUID(session_id)
        ).order_by(EmotionLog.id.asc()).all()
        
        emotion_log = []
        for i, log in enumerate(emotion_logs):
            emotion_log.append({
                "turn_number": i + 1,
                "emotion": log.customer_emotion,
                "confidence": 1.0,  # We don't store confidence, assume 1.0
                "mood_score": log.customer_mood_score,
                "risk_level": log.risk_level,
                "trend": log.emotion_trend,
            })
        
        duration_seconds = _compute_duration(session, messages)

        # Look up persona for name and difficulty
        persona = db.query(PersonaModel).filter(
            PersonaModel.id == session.persona_id
        ).first()
        persona_name = (persona.name_ar or persona.name_en) if persona else session.persona_id
        persona_difficulty = persona.difficulty if persona else "medium"

        # Session info
        session_info = {
            "session_id": str(session.id),
            "user_id": str(session.user_id),
            "persona_id": session.persona_id,
            "persona_name": persona_name,
            "persona_difficulty": persona_difficulty,
            # The session's chosen difficulty (what the trainee actually faced).
            # The evaluator uses this to calibrate scoring — a hard customer
            # shouldn't be graded on the same closing expectations as an easy one.
            "difficulty": getattr(session, "difficulty", None) or persona_difficulty,
            # The buyer scenario — lets the evaluator score whether the
            # salesperson discovered the customer's real needs and respected
            # their budget/timeline.
            "scenario": getattr(session, "scenario", None),
            "started_at": session.started_at.isoformat() if session.started_at else None,
            "ended_at": session.ended_at.isoformat() if session.ended_at else None,
            "duration_seconds": duration_seconds,
        }
        
        logger.info(f"[MANAGER] Gathered: {len(transcript)} messages, {len(emotion_log)} emotions")

    # Run structured fact-checking outside the DB session
    structured_fact_check = _gather_fact_check_context(transcript)

    return {
        "transcript": transcript,
        "emotion_log": emotion_log,
        "rag_context": [],
        "structured_fact_check": structured_fact_check,
        "session_info": session_info,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RAG Fact-Checking Context Gathering
# ═══════════════════════════════════════════════════════════════════════════════

def _gather_fact_check_context(transcript: list) -> dict:
    """
    Run structured fact-checking via hybrid RAG pipeline (claim extraction + KB comparison).

    This is called AFTER the session ends. The result is pre-computed and passed directly
    to the analyzer prompt, replacing unreliable unstructured RAG text chunks.

    Args:
        transcript: List of TranscriptMessage dicts from the session

    Returns:
        Structured fact-check result dict from rag.agent.fact_check_transcript, or {}
        on failure (evaluation still runs, just without fact-checking).
    """
    try:
        from rag.agent import fact_check_transcript
    except ImportError:
        logger.warning("[MANAGER] RAG module not available — fact-checking will be skipped")
        return {}

    try:
        result = fact_check_transcript(transcript)
        errors = result.get("errors", [])
        claims_checked = result.get("claims_checked", 0)
        logger.info(
            "[Evaluation] Fact-check complete: %d claims checked, %d errors found, accuracy_rate=%.1f%%",
            claims_checked,
            len(errors),
            result.get("accuracy_rate", 1.0) * 100,
        )
        # Debug print so it's visible in terminal during evaluation
        print(f"\n{'='*60}")
        print(f"[FACT-CHECK DEBUG] claims_checked={claims_checked}")
        print(f"[FACT-CHECK DEBUG] accurate={result.get('accurate_count', 0)}")
        print(f"[FACT-CHECK DEBUG] inaccurate={result.get('inaccurate_count', 0)}")
        print(f"[FACT-CHECK DEBUG] unverifiable={len(result.get('unverifiable_claims', []))}")
        print(f"[FACT-CHECK DEBUG] accuracy_rate={result.get('accuracy_rate', 1.0)*100:.1f}%")
        if errors:
            print(f"[FACT-CHECK DEBUG] ERRORS:")
            for err in errors:
                print(f"  [{err.get('severity','?').upper()}] turn={err.get('turn_number')} "
                      f"type={err.get('claim_type')} property={err.get('property_name')} "
                      f"claimed={err.get('claimed_value')} correct={err.get('correct_value')}")
                print(f"    {err.get('explanation_ar','')}")
        if result.get("unverifiable_claims"):
            print(f"[FACT-CHECK DEBUG] UNVERIFIABLE claims (no KB match):")
            for uv in result.get("unverifiable_claims", []):
                print(f"  turn={uv.get('turn_number')} type={uv.get('claim_type')} "
                      f"value={uv.get('claimed_value')} hint={uv.get('property_hint')} "
                      f"reason={uv.get('reason')}")
        print(f"{'='*60}\n")
        return result
    except Exception as exc:
        logger.warning("[MANAGER] Fact-check failed: %s — evaluation continues without it", exc)
        return {}


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation Manager Class
# ═══════════════════════════════════════════════════════════════════════════════

class EvaluationManager:
    """
    Main Orchestrator - Menna's Core Implementation
    
    This class is the entry point for evaluation. It:
    1. Gathers data from database
    2. Creates the initial state with LLM injected
    3. Calls Ismail's LangGraph pipeline
    4. Returns the final report
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the evaluation manager
        
        Args:
            verbose: Whether to print progress messages
        """
        self.verbose = verbose
        self.settings = settings
        
    def evaluate(
        self,
        session_id: str,
        mode: str = "training"
    ) -> FinalReport:
        """
        Run full evaluation and return report.
        
        This is what Bakr calls from the API.
        
        Args:
            session_id: Session to evaluate
            mode: "training" or "testing"
            
        Returns:
            FinalReport with complete evaluation results
            
        Raises:
            ValueError: If session not found or invalid mode
            RuntimeError: If evaluation pipeline fails
        """
        self._log(f" Starting evaluation for session {session_id}")
        self._log(f" Mode: {mode}")
        
        try:
            # Step 1: Gather inputs from database
            self._log(" Gathering inputs from database...")
            inputs = gather_evaluation_inputs(session_id)
            
            # Step 2: Create initial state
            state = create_initial_state(session_id, mode)
            self._log(" Initial state created")

            # Step 3: Inject LLM FIRST (CRITICAL!)
            self._log(" Injecting LLM into state...")
            state["llm"] = get_evaluation_llm()

            # Step 4: Populate state with gathered data
            state["transcript"] = inputs["transcript"]
            state["emotion_log"] = inputs["emotion_log"]
            state["rag_context"] = inputs["rag_context"]
            state["structured_fact_check"] = inputs["structured_fact_check"]
            state["session_info"] = inputs["session_info"]

            # Verify LLM is still there
            if not state.get("llm"):
                raise RuntimeError("LLM injection failed!")
            
            # Step 5: Run Ismail's LangGraph pipeline
            from evaluation.graphs.evaluation_graph import run_evaluation
            
            self._log(" Running LangGraph evaluation pipeline...")
            final_state = run_evaluation(state)
            
            # Step 6: Check for errors
            if final_state.get("errors"):
                error_msg = "; ".join(final_state["errors"])
                raise RuntimeError(f"Evaluation failed: {error_msg}")
            
            # Step 7: Extract final report
            final_report = final_state.get("final_report")
            if not final_report:
                raise RuntimeError("Pipeline completed but no final report generated")
            
            self._log(f" Evaluation complete! Score: {final_report.scores.overall_score}/100")

            # NOTE: Saving the FinalReport to the EvaluationReport DB table is the
            # responsibility of the CALLER (backend/services/evaluation_service.py →
            # run_evaluation_background). This method intentionally returns the report
            # without persisting it, so it remains testable without a DB dependency.

            return final_report
            
        except ImportError as e:
            raise RuntimeError(
                f"Could not import evaluation graph. "
                f"Make sure Ismail has created graphs/evaluation_graph.py. Error: {e}"
            )
        except Exception as e:
            logger.error(f"Evaluation failed for session {session_id}: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Evaluation failed: {e}")
    
    async def evaluate_async(
        self,
        session_id: str,
        mode: str = "training"
    ) -> FinalReport:
        """
        Async version for background tasks.
        
        Args:
            session_id: Session to evaluate
            mode: "training" or "testing"
            
        Returns:
            FinalReport with complete evaluation results
        """
        self._log(f" Starting async evaluation for session {session_id}")
        
        try:
            # Gather inputs (this is sync, but fast)
            inputs = gather_evaluation_inputs(session_id)
            
            # Create and populate state
            state = create_initial_state(session_id, mode)
            state["transcript"] = inputs["transcript"]
            state["emotion_log"] = inputs["emotion_log"]
            state["rag_context"] = inputs["rag_context"]
            state["structured_fact_check"] = inputs["structured_fact_check"]
            state["session_info"] = inputs["session_info"]
            state["llm"] = get_evaluation_llm()
            
            # Run async pipeline
            from evaluation.graphs.evaluation_graph import run_evaluation_async
            final_state = await run_evaluation_async(state)
            
            # Check for errors
            if final_state.get("errors"):
                error_msg = "; ".join(final_state["errors"])
                raise RuntimeError(f"Evaluation failed: {error_msg}")
            
            final_report = final_state.get("final_report")
            if not final_report:
                raise RuntimeError("Pipeline completed but no final report generated")
            
            self._log(f" Async evaluation complete! Score: {final_report.overall_score}/100")
            
            return final_report
            
        except ImportError:
            # Fallback to sync version if async not implemented
            logger.warning("Async evaluation not available, falling back to sync")
            return self.evaluate(session_id, mode)
        except Exception as e:
            logger.error(f"Async evaluation failed for session {session_id}: {e}")
            raise RuntimeError(f"Async evaluation failed: {e}")
    
    def get_statistics(self) -> dict:
        """
        Get statistics about the evaluation system.
        
        Returns:
            Dictionary with system statistics
        """
        from evaluation.config import SKILL_CONFIGS, CHECKPOINT_CONFIGS
        
        return {
            'total_skills': len(SKILL_CONFIGS),
            'total_checkpoints': len(CHECKPOINT_CONFIGS),
            'skill_keys': list(SKILL_CONFIGS.keys()),
            'checkpoint_keys': list(CHECKPOINT_CONFIGS.keys()),
            'pass_threshold': self.settings.thresholds.overall_pass,
            'evaluation_timeout': self.settings.evaluation_timeout_seconds,
            'features': {
                'turn_feedback': self.settings.enable_turn_feedback,
                'fact_verification': self.settings.enable_fact_verification,
            }
        }
    
    def _log(self, message: str) -> None:
        """
        Log a message if verbose mode is enabled
        
        Args:
            message: Message to log
        """
        if self.verbose:
            print(message)
            logger.info(message)


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience Functions
# ═══════════════════════════════════════════════════════════════════════════════

def create_manager(verbose: bool = True) -> EvaluationManager:
    """
    Create an evaluation manager instance.
    
    Args:
        verbose: Whether to print progress messages
        
    Returns:
        EvaluationManager instance
        
    Example:
        >>> from evaluation.manager import create_manager
        >>> manager = create_manager(verbose=True)
        >>> report = manager.evaluate(session_id="123", mode="training")
    """
    return EvaluationManager(verbose=verbose)


def quick_evaluate(session_id: str, mode: str = "training") -> FinalReport:
    """
    Quick evaluation function - creates manager and evaluates in one call.
    
    Args:
        session_id: Session to evaluate
        mode: "training" or "testing"
        
    Returns:
        FinalReport
        
    Example:
        >>> from evaluation.manager import quick_evaluate
        >>> report = quick_evaluate("123", mode="testing")
        >>> print(f"Score: {report.overall_score}")
    """
    manager = create_manager(verbose=False)
    return manager.evaluate(session_id, mode)