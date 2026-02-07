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


class EvaluationManager:
    """
    Main Orchestrator - Menna's Core Implementation
    
    This class is the entry point for evaluation. It:
    1. Creates the initial state
    2. Calls Ismail's LangGraph pipeline
    3. Returns the final report
    
    The actual evaluation logic is in Ismail's pipeline (graphs/evaluation_graph.py).
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
            # Import here to avoid circular dependencies
            from evaluation.graphs.evaluation_graph import run_evaluation
            
            # Create initial state
            state = create_initial_state(session_id, mode)
            self._log("[MENNA] Initial state created")
            
            # Run Ismail's LangGraph pipeline
            self._log("[MENNA] Running LangGraph evaluation pipeline...")
            final_state = run_evaluation(state)
            
            # Check for errors
            if final_state.get("errors"):
                error_msg = "; ".join(final_state["errors"])
                raise RuntimeError(f"Evaluation failed: {error_msg}")
            
            # Extract final report
            final_report = final_state.get("final_report")
            if not final_report:
                raise RuntimeError("Pipeline completed but no final report generated")
            
            self._log(f"[MENNA] Evaluation complete! Score: {final_report.overall_score}/100")
            
            return final_report
            
        except ImportError as e:
            raise RuntimeError(
                f"Could not import evaluation graph. "
                f"Make sure Ismail has created graphs/evaluation_graph.py. Error: {e}"
            )
        except Exception as e:
            logger.error(f"Evaluation failed for session {session_id}: {e}")
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
            from evaluation.graphs.evaluation_graph import run_evaluation_async
            
            # Create initial state
            state = create_initial_state(session_id, mode)
            
            # Run async pipeline
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


# Convenience function for quick access
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