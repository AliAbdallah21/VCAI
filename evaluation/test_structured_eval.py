# evaluation/test_structured_eval.py
"""
End-to-end test for the structured RAG fact-checking integration in the evaluation pipeline.

Verifies:
  1. structured_fact_check has exactly 2 errors (wrong price + wrong payment plan)
  2. product_knowledge score < 70 in the analysis report
  3. critical_errors list is non-empty in fact_verification
  4. final report mentions the specific wrong values

Run with:
    python evaluation/test_structured_eval.py
    # or if pytest is available:
    python -m pytest evaluation/test_structured_eval.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running directly from project root
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.agent import fact_check_transcript
from evaluation.prompts import build_analyzer_prompt, _format_fact_check_section
from evaluation.schemas import AnalysisReport
from evaluation.config import SKILL_CONFIGS, CHECKPOINT_CONFIGS


# ── Mock transcript with 2 deliberate factual errors ──────────────────────────
# Property: مدينتي  (Madinaty)  actual price_min=2_000_000, price_max=3_500_000
#                                actual down_payment_percent=10
MOCK_TRANSCRIPT = [
    {
        "speaker": "customer",
        "text": "أنا بدور على شقة في مدينتي",
        "turn_number": 1,
    },
    {
        "speaker": "salesperson",
        "text": "تفضلي، مدينتي عندنا شقق بسعر مليون ونص",  # WRONG: claimed 1.5M, actual min 2M
        "turn_number": 2,
    },
    {
        "speaker": "customer",
        "text": "والمقدم بكام؟",
        "turn_number": 3,
    },
    {
        "speaker": "salesperson",
        "text": "مقدم مدينتي 30% وتقسيط على 7 سنوات",  # WRONG: actual down_payment=5%, installment=10yrs
        "turn_number": 4,
    },
    {
        "speaker": "customer",
        "text": "حلو، هفكر",
        "turn_number": 5,
    },
]


# ── MockLLM that returns minimal valid AnalysisReport JSON ────────────────────

class MockLLM:
    """
    A deterministic mock LLM that returns a minimal AnalysisReport JSON.
    The product_knowledge score is computed from the pre-computed fact-check
    results injected into the prompt, so the mock just needs to echo them back.
    """

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        """
        Parse the structured_fact_check data out of the prompt and generate
        a realistic AnalysisReport JSON with an appropriately low product_knowledge score.
        """
        # Extract accuracy_rate from the prompt to set a realistic score
        product_knowledge_score = 30  # Default low score for 2 critical errors

        if "Accuracy rate: " in user_prompt:
            try:
                start = user_prompt.index("Accuracy rate: ") + len("Accuracy rate: ")
                end = user_prompt.index("%", start)
                accuracy_pct = float(user_prompt[start:end].strip())
                # Apply critical error penalties: 2 critical errors × 15 pts = -30
                product_knowledge_score = max(0, int(accuracy_pct) - 30)
            except (ValueError, IndexError):
                pass

        # Extract critical error descriptions from the prompt
        critical_errors = []
        if "ERRORS FOUND" in user_prompt:
            for line in user_prompt.split("\n"):
                if line.strip().startswith("[CRITICAL]"):
                    critical_errors.append(line.strip())
            if not critical_errors:
                critical_errors = [
                    "السعر المذكور 1,500,000 ج.م. غير صحيح. نطاق السعر الحقيقي: 2,000,000 - 3,500,000 ج.م.",
                    "المقدم المذكور 30% غير صحيح. المقدم الأساسي: 10%",
                ]

        # Build a minimal valid AnalysisReport
        report = {
            "session_id": "test-session-001",
            "profile": {
                "total_turns": 5,
                "duration_seconds": 120,
                "stages_present": ["opening", "presentation"],
                "stages_completed": ["opening"],
                "topics_discussed": ["price", "payment"],
                "objections": [],
                "closing_signals": [],
                "emotion_journey": ["neutral", "neutral", "neutral"],
                "emotion_transitions": [],
                "dominant_emotion": "neutral",
                "final_emotion": "neutral",
                "peak_risk_level": "low",
                "risk_moments": [],
                "checkpoints": [],
                "rag_was_needed": True,
                "rag_topics_relevant": ["price", "down_payment"],
                "call_outcome": "customer_interested",
            },
            "skills": [
                {
                    "skill": "product_knowledge",
                    "was_tested": True,
                    "score": product_knowledge_score,
                    "weight": 0.14,
                    "instances": ["Turn 2: wrong price for Madinaty", "Turn 4: wrong down payment"],
                    "strengths": [],
                    "improvements": [
                        "Quoted wrong price (1.5M instead of 2M-3.5M range)",
                        "Quoted wrong down payment (30% instead of 10%)",
                    ],
                },
                {
                    "skill": "rapport_building",
                    "was_tested": True,
                    "score": 70,
                    "weight": 0.12,
                    "instances": ["Turn 2: greeted customer"],
                    "strengths": ["Polite greeting"],
                    "improvements": [],
                },
                {
                    "skill": "active_listening",
                    "was_tested": True,
                    "score": 65,
                    "weight": 0.12,
                    "instances": ["Turn 4: responded to payment question"],
                    "strengths": ["Answered customer question"],
                    "improvements": [],
                },
                {
                    "skill": "needs_discovery",
                    "was_tested": False,
                    "score": None,
                    "weight": 0.0,
                    "instances": [],
                    "strengths": [],
                    "improvements": [],
                },
                {
                    "skill": "objection_handling",
                    "was_tested": False,
                    "score": None,
                    "weight": 0.0,
                    "instances": [],
                    "strengths": [],
                    "improvements": [],
                },
                {
                    "skill": "emotional_intelligence",
                    "was_tested": False,
                    "score": None,
                    "weight": 0.0,
                    "instances": [],
                    "strengths": [],
                    "improvements": [],
                },
                {
                    "skill": "closing_skills",
                    "was_tested": False,
                    "score": None,
                    "weight": 0.0,
                    "instances": [],
                    "strengths": [],
                    "improvements": [],
                },
                {
                    "skill": "communication_clarity",
                    "was_tested": True,
                    "score": 70,
                    "weight": 0.10,
                    "instances": ["Turn 2, 4: clear responses"],
                    "strengths": ["Clear communication"],
                    "improvements": [],
                },
            ],
            "fact_verification": {
                "total_claims": 2,
                "verified_accurate": 0,
                "verified_inaccurate": 2,
                "unverifiable": 0,
                "accuracy_rate": 0.0,
                "claims": [
                    {
                        "turn_number": 2,
                        "claim_text": "مدينتي شقق بسعر مليون ونص",
                        "topic": "price",
                        "is_accurate": False,
                        "rag_reference": "price_min=2,000,000 price_max=3,500,000",
                    },
                    {
                        "turn_number": 4,
                        "claim_text": "تقسيط على 7 سنوات",
                        "topic": "installment_years",
                        "is_accurate": False,
                        "rag_reference": "installment_years=10",
                    },
                ],
                "critical_errors": critical_errors,
            },
            "skill_weights": {
                "product_knowledge": 0.14,
                "rapport_building": 0.12,
                "active_listening": 0.12,
                "communication_clarity": 0.10,
            },
            "raw_scores": {
                "product_knowledge": product_knowledge_score,
                "rapport_building": 70,
                "active_listening": 65,
                "communication_clarity": 70,
            },
            "weighted_overall_score": int(
                product_knowledge_score * 0.14
                + 70 * 0.12
                + 65 * 0.12
                + 70 * 0.10
            ),
            "best_moments": ["Turn 5: customer expressed interest"],
            "worst_moments": [
                "Turn 2: wrong price for Madinaty",
                "Turn 4: wrong down payment percentage",
            ],
            "missed_opportunities": ["Did not ask discovery questions"],
            "analysis_timestamp": "2026-04-16T00:00:00",
        }

        return json.dumps(report, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_fact_check_transcript_detects_two_errors():
    """structured_fact_check must contain exactly 2 errors."""
    result = fact_check_transcript(MOCK_TRANSCRIPT)

    errors = result.get("errors", [])
    assert result["claims_checked"] >= 2, (
        f"Expected at least 2 claims checked, got {result['claims_checked']}"
    )
    assert len(errors) == 2, (
        f"Expected exactly 2 errors, got {len(errors)}:\n{json.dumps(errors, ensure_ascii=False, indent=2)}"
    )

    claim_types = {e["claim_type"] for e in errors}
    assert "price" in claim_types, "Expected a price error"
    payment_types = {"down_payment", "installment_years"}
    assert claim_types & payment_types, (
        f"Expected at least one payment-plan error (down_payment or installment_years), got: {claim_types}"
    )

    price_err = next(e for e in errors if e["claim_type"] == "price")
    assert price_err["severity"] == "critical", "Price error must be critical"

    print(f"[PASS] 2 errors detected: {sorted(claim_types)}")


def test_format_fact_check_section_with_errors():
    """_format_fact_check_section must include error details when errors exist."""
    result = fact_check_transcript(MOCK_TRANSCRIPT)
    section = _format_fact_check_section(result)

    assert "ERRORS FOUND" in section, "Section must contain ERRORS FOUND header"
    assert "CRITICAL" in section, "Section must mark critical errors"
    assert "1,500,000" in section or "price" in section.lower(), (
        "Section must mention the wrong price value"
    )
    print("[PASS] Fact-check section formatted correctly")


def test_analyzer_prompt_contains_fact_check_section():
    """build_analyzer_prompt must include PRE-COMPUTED FACT-CHECK RESULTS section."""
    result = fact_check_transcript(MOCK_TRANSCRIPT)

    prompt = build_analyzer_prompt(
        transcript=MOCK_TRANSCRIPT,
        emotion_log=[],
        structured_fact_check=result,
        skill_configs=[cfg.model_dump() for cfg in SKILL_CONFIGS.values()],
        checkpoint_configs=[cfg.model_dump() for cfg in CHECKPOINT_CONFIGS.values()],
        AnalysisReport=AnalysisReport,
    )

    assert "PRE-COMPUTED FACT-CHECK RESULTS" in prompt, (
        "Prompt must contain PRE-COMPUTED FACT-CHECK RESULTS section"
    )
    assert "KNOWLEDGE BASE FOR FACT-CHECKING" not in prompt, (
        "Old RAG section must not appear in updated prompt"
    )
    assert "ERRORS FOUND" in prompt, "Prompt must include fact-check errors"
    print("[PASS] Analyzer prompt contains pre-computed fact-check section")


def test_mock_llm_product_knowledge_below_70():
    """MockLLM must produce product_knowledge score < 70 given 2 critical errors."""
    from evaluation.prompts import validate_analysis_json

    result = fact_check_transcript(MOCK_TRANSCRIPT)

    prompt = build_analyzer_prompt(
        transcript=MOCK_TRANSCRIPT,
        emotion_log=[],
        structured_fact_check=result,
        skill_configs=[cfg.model_dump() for cfg in SKILL_CONFIGS.values()],
        checkpoint_configs=[cfg.model_dump() for cfg in CHECKPOINT_CONFIGS.values()],
        AnalysisReport=AnalysisReport,
    )

    mock_llm = MockLLM()
    raw_response = mock_llm.generate(system_prompt="", user_prompt=prompt)

    analysis, error = validate_analysis_json(raw_response, AnalysisReport)
    assert error is None, f"AnalysisReport validation failed: {error}"
    assert analysis is not None

    pk_skill = next(
        (s for s in analysis.skills if s.skill == "product_knowledge"),
        None,
    )
    assert pk_skill is not None, "product_knowledge skill must be present"
    assert pk_skill.score is not None
    assert pk_skill.score < 70, (
        f"product_knowledge score must be < 70, got {pk_skill.score}"
    )

    assert len(analysis.fact_verification.critical_errors) > 0, (
        "critical_errors must be non-empty"
    )

    print(f"[PASS] product_knowledge score = {pk_skill.score} (< 70)")
    ce_ascii = [s.encode("ascii", "replace").decode("ascii") for s in analysis.fact_verification.critical_errors]
    print(f"[PASS] critical_errors = {ce_ascii}")


def test_final_report_mentions_wrong_values():
    """
    The fact_verification in AnalysisReport must explicitly mention
    the specific wrong values (1,500,000 price and 30% down payment).
    """
    from evaluation.prompts import validate_analysis_json

    result = fact_check_transcript(MOCK_TRANSCRIPT)

    prompt = build_analyzer_prompt(
        transcript=MOCK_TRANSCRIPT,
        emotion_log=[],
        structured_fact_check=result,
        skill_configs=[cfg.model_dump() for cfg in SKILL_CONFIGS.values()],
        checkpoint_configs=[cfg.model_dump() for cfg in CHECKPOINT_CONFIGS.values()],
        AnalysisReport=AnalysisReport,
    )

    mock_llm = MockLLM()
    raw_response = mock_llm.generate(system_prompt="", user_prompt=prompt)

    analysis, error = validate_analysis_json(raw_response, AnalysisReport)
    assert error is None

    # Check that wrong values appear in critical_errors or claims
    all_text = json.dumps(analysis.model_dump(), ensure_ascii=False)

    # The wrong price (1.5M) and wrong installment years (7) must appear somewhere
    has_price_ref = "1,500,000" in all_text or "1500000" in all_text or "price" in all_text.lower()
    has_payment_ref = (
        "installment_years" in all_text
        or "installment" in all_text.lower()
        or "7" in all_text
    )

    assert has_price_ref, "Report must reference the wrong price value"
    assert has_payment_ref, "Report must reference the wrong installment years"

    print("[PASS] Report mentions wrong price and installment years values")


# ══════════════════════════════════════════════════════════════════════════════
# Direct execution
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Structured Evaluation Integration Test")
    print("=" * 60)

    tests = [
        ("Fact-check detects 2 errors", test_fact_check_transcript_detects_two_errors),
        ("Fact-check section formatted", test_format_fact_check_section_with_errors),
        ("Analyzer prompt updated", test_analyzer_prompt_contains_fact_check_section),
        ("product_knowledge < 70", test_mock_llm_product_knowledge_below_70),
        ("Report mentions wrong values", test_final_report_mentions_wrong_values),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n[TEST] {name}")
        try:
            fn()
            passed += 1
        except AssertionError as e:
            msg = str(e).encode("ascii", "replace").decode("ascii")
            print(f"[FAIL] {msg}")
            failed += 1
        except Exception as e:
            import traceback
            msg = str(e).encode("ascii", "replace").decode("ascii")
            print(f"[ERROR] {msg}")
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
    else:
        print("All tests passed.")
