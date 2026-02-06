from __future__ import annotations

import time
from typing import Protocol

from evaluation.state import (
    EvaluationState,
    EvaluationProgress,
    update_state_status,
    record_node_timing,
    add_error,
    mark_failed,
)
from evaluation.schemas import AnalysisReport
from evaluation.prompts import (
    ANALYZER_SYSTEM_PROMPT,
    build_analyzer_prompt,
    validate_analysis_json,
)
from evaluation.config import SKILL_CONFIGS, CHECKPOINT_CONFIGS


class EvaluationLLM(Protocol):
    """
    Minimal LLM interface for evaluation.

    Your project can satisfy this with LangChain, Ollama, OpenAI, etc.
    As long as the object in state["llm"] implements this method.
    """
    def generate(self, *, system_prompt: str, user_prompt: str) -> str: ...


def analyzer_node(state: EvaluationState) -> EvaluationState:
    """
    Node 2 (LLM): Analyzer pass.

    Reads:
      - transcript
      - emotion_log (optional)
      - rag_context (optional)

    Writes:
      - analysis_report (AnalysisReport)
      - analysis_raw_response (str)
      - status/progress/node_timings
    """
    t0 = time.perf_counter()
    state = update_state_status(state, "analyzing", EvaluationProgress.ANALYZING)

    try:
        # Validate required inputs (doc requirement: transcript must exist)
        transcript = state.get("transcript", [])
        if not transcript:
            return mark_failed(state, "Analyzer: transcript is empty")

        emotion_log = state.get("emotion_log", [])
        rag_context = state.get("rag_context", [])

        # Build analyzer prompt (doc requirement: use skills + checkpoints)
        prompt = build_analyzer_prompt(
            transcript=transcript,
            emotion_log=emotion_log,
            rag_context=rag_context,
            skill_configs=[cfg.model_dump() for cfg in SKILL_CONFIGS.values()],
            checkpoint_configs=[cfg.model_dump() for cfg in CHECKPOINT_CONFIGS.values()],
            AnalysisReport=AnalysisReport,
        )

        # LLM call (doc requires LLM here; interface provided by infra)
        llm = state.get("llm")  # injected by manager/service
        if llm is None:
            return mark_failed(
                state,
                "Analyzer: missing LLM client in state['llm'] (inject it before running the graph).",
            )

        llm_response_text = llm.generate(
            system_prompt=ANALYZER_SYSTEM_PROMPT,
            user_prompt=prompt,
        )

        # Validate JSON against AnalysisReport schema (doc requirement)
        analysis, error = validate_analysis_json(llm_response_text, AnalysisReport)
        if error or analysis is None:
            return mark_failed(state, f"Analyzer JSON validation failed: {error}")

        # Write outputs to state
        state["analysis_report"] = analysis
        state["analysis_raw_response"] = llm_response_text

        state = update_state_status(state, "analyzing", EvaluationProgress.ANALYSIS_COMPLETE)

    except Exception as e:
        state = add_error(state, f"Analyzer exception: {str(e)}")
        return mark_failed(state, str(e))

    finally:
        state = record_node_timing(state, "analyzer_node", time.perf_counter() - t0)

    return state
