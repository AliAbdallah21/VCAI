# evaluation/pipeline/synthesizer.py

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
from evaluation.schemas import FinalReport
from evaluation.prompts import (
    build_synthesizer_prompt,
    validate_report_json,
)


class EvaluationLLM(Protocol):
    """
    Minimal LLM interface for evaluation.

    Your project can satisfy this with LangChain, Ollama, OpenAI, etc.
    As long as the object in state["llm"] implements this method.
    """
    def generate(self, *, system_prompt: str, user_prompt: str) -> str: ...


def synthesizer_node(state: EvaluationState) -> EvaluationState:
    """
    Node 3 (LLM): Synthesizer pass.

    Reads:
      - analysis_report (required)
      - quick_stats (required)
      - mode (required)

    Writes:
      - final_report (FinalReport)
      - synthesis_raw_response (str)
      - status/progress/node_timings
    """
    t0 = time.perf_counter()
    state = update_state_status(state, "synthesizing", EvaluationProgress.SYNTHESIZING)

    try:
        analysis = state.get("analysis_report")
        if analysis is None:
            return mark_failed(state, "Synthesizer: missing analysis_report")

        quick_stats = state.get("quick_stats")
        if quick_stats is None:
            return mark_failed(state, "Synthesizer: missing quick_stats")

        mode = state.get("mode")
        if mode is None:
            return mark_failed(state, "Synthesizer: missing mode")

        # Build prompts (system + user) based on mode (training/testing)
        system_prompt, user_prompt = build_synthesizer_prompt(
            analysis_report=analysis.model_dump(),
            quick_stats=quick_stats.model_dump(),
            mode=mode.value if hasattr(mode, "value") else str(mode),
            FinalReport=FinalReport,
        )

        # LLM call 
        llm = state.get("llm")  # injected by manager/service
        if llm is None:
            return mark_failed(
                state,
                "Synthesizer: missing LLM client in state['llm'] (inject it before running the graph).",
            )

        llm_response_text = llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        # ADD THIS DEBUG LOGGING:
        print("=" * 80)
        print("[DEBUG] SYNTHESIZER RAW LLM OUTPUT:")
        print(llm_response_text[:2000])  # First 2000 chars
        print("=" * 80)

        # Validate JSON against FinalReport schema
        report, error = validate_report_json(llm_response_text, FinalReport)
        if error or report is None:
            return mark_failed(state, f"Synthesizer JSON validation failed: {error}")

        state["final_report"] = report
        state["synthesis_raw_response"] = llm_response_text

        state = update_state_status(state, "synthesizing", EvaluationProgress.SYNTHESIS_COMPLETE)

    except Exception as e:
        state = add_error(state, f"Synthesizer exception: {str(e)}")
        return mark_failed(state, str(e))

    finally:
        state = record_node_timing(state, "synthesizer_node", time.perf_counter() - t0)

    return state
