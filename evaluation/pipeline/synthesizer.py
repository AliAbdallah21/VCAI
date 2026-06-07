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
from evaluation.schemas import FinalReport, FactCheckSummary, FactCheckError
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
        print("=" * 80)
        print("[DEBUG] SYNTHESIZER RAW LLM OUTPUT (first 2000 chars):")
        print(llm_response_text[:2000])
        print("=" * 80)

        # Validate JSON against FinalReport schema
        report, error = validate_report_json(llm_response_text, FinalReport)

        # If validation failed, retry ONCE with a repair prompt — Gemini
        # occasionally truncates / produces stray commas in long JSON outputs,
        # and a focused retry usually fixes it without re-running the whole
        # analysis. This turned a one-shot failure into a self-healing path.
        if error or report is None:
            print(f"[SYNTHESIZER] First JSON parse failed: {error}. Retrying with repair prompt...")
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
                print(f"[SYNTHESIZER] Repair response received ({len(repaired_text)} chars), revalidating...")
                report, error = validate_report_json(repaired_text, FinalReport)
                if not (error or report is None):
                    llm_response_text = repaired_text
                    print("[SYNTHESIZER] JSON repair SUCCEEDED.")
            except Exception as repair_exc:
                print(f"[SYNTHESIZER] Repair call itself failed: {repair_exc}")

        if error or report is None:
            return mark_failed(state, f"Synthesizer JSON validation failed (after 1 repair retry): {error}")

        # Overwrite LLM-hallucinated metadata with real values from state
        session_info = state.get("session_info") or {}
        if session_info:
            from datetime import datetime as _dt
            real_session_id  = str(session_info.get("session_id",  report.session_id))
            real_user_id     = str(session_info.get("user_id",     report.user_id))
            real_persona_id  = str(session_info.get("persona_id",  report.persona_id))
            real_persona_name = str(session_info.get("persona_name", report.persona_name))
            real_report_id   = f"RPT-{_dt.utcnow().strftime('%Y-%m-%d-%H-%M-%S')}"
            report = report.model_copy(update={
                "report_id":    real_report_id,
                "session_id":   real_session_id,
                "user_id":      real_user_id,
                "persona_id":   real_persona_id,
                "persona_name": real_persona_name,
                # The LLM hallucinates a stale created_at (e.g. "2024-05-21");
                # stamp the real generation time so the report isn't dated wrong.
                "created_at":   _dt.utcnow(),
            })
            print(f"[SYNTHESIZER] Filled metadata: session={real_session_id[:8]}... user={real_user_id[:8]}... persona={real_persona_id}")

        # Inject deterministic fact-check results (NOT generated by the LLM).
        # If the salesperson made claims that contradict the KB, surface them
        # as their own section AND cap product_knowledge_score so the LLM can't
        # praise inaccurate product knowledge.
        fact_check_raw = state.get("structured_fact_check") or {}
        if fact_check_raw:
            try:
                errors = []
                for e in (fact_check_raw.get("errors") or []):
                    errors.append(FactCheckError(
                        turn_number=e.get("turn_number"),
                        claim_type=str(e.get("claim_type", "")),
                        claimed_value=str(e.get("claimed_value", "")),
                        correct_value=str(e.get("correct_value", "")),
                        severity=e.get("severity") if e.get("severity") in ("critical", "minor") else "minor",
                        explanation_ar=e.get("explanation_ar"),
                        property_name=e.get("property_name"),
                    ))
                properties = []
                for pm in (fact_check_raw.get("property_mentions") or []):
                    nm = pm.get("name") or pm.get("property_id")
                    if nm and nm not in properties:
                        properties.append(str(nm))

                summary = FactCheckSummary(
                    claims_checked=int(fact_check_raw.get("claims_checked", 0) or 0),
                    accurate_count=int(fact_check_raw.get("accurate_count", 0) or 0),
                    inaccurate_count=int(fact_check_raw.get("inaccurate_count", 0) or 0),
                    unverifiable_count=int(fact_check_raw.get("unverifiable_count", 0) or 0),
                    accuracy_rate=float(fact_check_raw.get("accuracy_rate", 0.0) or 0.0),
                    errors=errors,
                    properties_discussed=properties,
                )
                report = report.model_copy(update={"fact_check": summary})

                # Cap product_knowledge_score by accuracy. The LLM tends to praise
                # "good product knowledge" even when prices/sizes were wrong —
                # this enforces the floor.
                if summary.claims_checked > 0 and report.scores and report.scores.skills:
                    max_pk = int(round(summary.accuracy_rate * 100))
                    new_skills = []
                    capped = False
                    for s in report.scores.skills:
                        if s.skill_key == "product_knowledge" and s.score > max_pk:
                            new_skills.append(s.model_copy(update={
                                "score": max_pk,
                                "weighted_contribution": round(max_pk * s.weight, 3),
                                "summary": f"{s.summary} (Capped by fact-check accuracy {int(summary.accuracy_rate*100)}%)".strip(),
                            }))
                            capped = True
                        else:
                            new_skills.append(s)
                    if capped:
                        report = report.model_copy(update={
                            "scores": report.scores.model_copy(update={"skills": new_skills})
                        })
                        print(f"[SYNTHESIZER] Capped product_knowledge_score → {max_pk} (accuracy={summary.accuracy_rate:.0%})")

                print(f"[SYNTHESIZER] Fact-check surfaced: {summary.accurate_count}/{summary.claims_checked} accurate ({len(summary.errors)} errors)")
            except Exception as fc_err:
                print(f"[SYNTHESIZER] Fact-check inject failed (non-fatal): {fc_err}")

        # ── Reconcile quick_stats checkpoints (deterministic, not LLM) ────────
        # calculate_quick_stats() runs before the analyzer, so it never knows
        # the checkpoints — quick_stats.checkpoints_achieved stays 0 and the
        # list empty even when the analyzer found several. Derive the real
        # counts from the analyzer's checkpoints array.
        try:
            if report.checkpoints:
                achieved_n = sum(1 for c in report.checkpoints if c.achieved)
                cp_list = [f"{c.icon} {c.name}" for c in report.checkpoints]
                report = report.model_copy(update={
                    "quick_stats": report.quick_stats.model_copy(update={
                        "checkpoints_achieved": achieved_n,
                        "checkpoints_total": len(report.checkpoints),
                        "checkpoint_list": cp_list,
                    })
                })
                print(f"[SYNTHESIZER] Reconciled quick_stats checkpoints: "
                      f"{achieved_n}/{len(report.checkpoints)}")
        except Exception as cp_err:
            print(f"[SYNTHESIZER] Checkpoint reconcile failed (non-fatal): {cp_err}")

        # ── Restore verbatim turn_feedback text (deterministic, not LLM) ──────
        # The synthesizer LLM sometimes rewrites a turn's words — e.g. emitting
        # a templated "[اسم]" greeting for turn 1 instead of the real line. A
        # turn quote must be exact, so overwrite each turn_feedback.text with
        # the real transcript line, keyed by (turn_number, speaker).
        try:
            transcript = state.get("transcript") or []
            if transcript and report.turn_feedback:
                text_by_turn = {
                    (t.get("turn_number"), t.get("speaker")): (t.get("text") or "")
                    for t in transcript
                }
                new_tf, fixed = [], 0
                for tf in report.turn_feedback:
                    real = text_by_turn.get((tf.turn_number, tf.speaker))
                    if real and real.strip() and real.strip() != (tf.text or "").strip():
                        new_tf.append(tf.model_copy(update={"text": real.strip()}))
                        fixed += 1
                    else:
                        new_tf.append(tf)
                if fixed:
                    report = report.model_copy(update={"turn_feedback": new_tf})
                    print(f"[SYNTHESIZER] Corrected {fixed} turn_feedback text(s) from transcript")
        except Exception as tf_err:
            print(f"[SYNTHESIZER] Turn-text restore failed (non-fatal): {tf_err}")

        state["final_report"] = report
        state["synthesis_raw_response"] = llm_response_text

        state = update_state_status(state, "synthesizing", EvaluationProgress.SYNTHESIS_COMPLETE)

    except Exception as e:
        state = add_error(state, f"Synthesizer exception: {str(e)}")
        return mark_failed(state, str(e))

    finally:
        state = record_node_timing(state, "synthesizer_node", time.perf_counter() - t0)

    return state
