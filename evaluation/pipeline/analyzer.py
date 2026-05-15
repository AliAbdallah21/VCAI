from __future__ import annotations

import time
from typing import Protocol

from evaluation.state import (
    EvaluationState,
    EvaluationProgress,
    update_state_status,
    record_node_timing,
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
        structured_fact_check = state.get("structured_fact_check", {})

        # ended_by_user: defaults to True because the overwhelmingly common
        # end-path is the user clicking "End Session". The session_info dict
        # can override this for sessions auto-ended by inactivity/timeout.
        session_info = state.get("session_info") or {}
        ended_by_user = session_info.get("ended_by_user", True)
        # difficulty: the session's chosen difficulty — calibrates scoring so
        # a hard customer isn't graded on easy-customer closing expectations.
        difficulty = session_info.get("difficulty", "medium")
        # scenario: the buyer's real situation — lets the analyzer judge
        # needs-discovery and budget-fit objectively.
        scenario = session_info.get("scenario")

        # Build analyzer prompt (doc requirement: use skills + checkpoints)
        prompt = build_analyzer_prompt(
            transcript=transcript,
            emotion_log=emotion_log,
            structured_fact_check=structured_fact_check,
            skill_configs=[cfg.model_dump() for cfg in SKILL_CONFIGS.values()],
            checkpoint_configs=[cfg.model_dump() for cfg in CHECKPOINT_CONFIGS.values()],
            AnalysisReport=AnalysisReport,
            ended_by_user=ended_by_user,
            difficulty=difficulty,
            scenario=scenario,
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

        # Validate JSON against AnalysisReport schema
        analysis, error = validate_analysis_json(llm_response_text, AnalysisReport)

        # One repair retry on JSON failure (Gemini occasionally truncates /
        # produces stray commas in long JSON outputs).
        if error or analysis is None:
            print(f"[ANALYZER] First JSON parse failed: {error}. Retrying with repair prompt...")
            repair_system = (
                "You are a strict JSON repairer. You will receive a JSON document "
                "that failed parsing and the parser error. Return ONLY the corrected "
                "JSON document — no prose, no markdown fences, no explanation. "
                "Preserve every field and value; only fix syntax."
            )
            repair_user = (
                f"PARSER ERROR:\n{error}\n\n"
                f"BROKEN JSON (return a fixed version of this, nothing else):\n"
                f"{llm_response_text}"
            )
            try:
                repaired_text = llm.generate(
                    system_prompt=repair_system,
                    user_prompt=repair_user,
                )
                print(f"[ANALYZER] Repair response received ({len(repaired_text)} chars), revalidating...")
                analysis, error = validate_analysis_json(repaired_text, AnalysisReport)
                if not (error or analysis is None):
                    llm_response_text = repaired_text
                    print("[ANALYZER] JSON repair SUCCEEDED.")
            except Exception as repair_exc:
                print(f"[ANALYZER] Repair call itself failed: {repair_exc}")

        if error or analysis is None:
            return mark_failed(state, f"Analyzer JSON validation failed (after 1 repair retry): {error}")

        # Write outputs to state
        state["analysis_report"] = analysis
        state["analysis_raw_response"] = llm_response_text

        state = update_state_status(state, "analyzing", EvaluationProgress.ANALYSIS_COMPLETE)

    except Exception as e:
        return mark_failed(state, f"Analyzer exception: {str(e)}")

    finally:
        state = record_node_timing(state, "analyzer_node", time.perf_counter() - t0)

    return state
