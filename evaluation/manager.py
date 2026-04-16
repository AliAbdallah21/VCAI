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

class EvaluationLLMWrapper:
    """
    Wrapper using OpenRouter for evaluation.
    """
    
    def __init__(self):
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is required for evaluation. "
                "Set it in your .env file or shell environment."
            )
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        # Try these models in order:
        # 1. "anthropic/claude-3.5-sonnet" (best, costs credits)
        # 2. "google/gemini-2.0-flash-exp:free" (free, good)
        # 3. "meta-llama/llama-3.1-70b-instruct" (cheap, decent)
        self.model = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-3.7-sonnet")
    
    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        """Generate response using OpenRouter."""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 8192,  # Much higher than your local model
        }
        
        response = requests.post(self.base_url, json=payload, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["message"]["content"]


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
        
        # Calculate duration
        duration_seconds = 0
        if session.duration_seconds:
            duration_seconds = session.duration_seconds
        elif session.started_at and session.ended_at:
            duration_seconds = int((session.ended_at - session.started_at).total_seconds())
        
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
        logger.info(
            "[Evaluation] Fact-check complete: %d claims checked, %d errors found, accuracy_rate=%.1f%%",
            result.get("claims_checked", 0),
            len(errors),
            result.get("accuracy_rate", 1.0) * 100,
        )
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
        self._log(f"[MENNA] Starting evaluation for session {session_id}")
        self._log(f"[MENNA] Mode: {mode}")
        
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
        self._log(f"[MENNA] Starting async evaluation for session {session_id}")
        
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
            
            self._log(f"[MENNA] Async evaluation complete! Score: {final_report.overall_score}/100")
            
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