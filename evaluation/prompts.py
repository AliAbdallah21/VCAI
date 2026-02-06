# evaluation/prompts.py
"""
VCAI Evaluation Module - LLM Prompts + JSON validation helpers (Pydantic v2)

This module provides:
- Pydantic models for AnalysisReport + FinalReport
- JSON parsing/validation helpers for LLM outputs
- Prompt builders for:
  - Analyzer: conversation -> AnalysisReport JSON
  - Synthesizer: analysis + mode -> FinalReport JSON

Modes:
- training: encouraging tone + suggestions + turn-by-turn feedback
- testing: professional/objective + pass/fail using 75% threshold
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, TypeVar

import json
import re

from pydantic import BaseModel, Field, ValidationError


# =============================================================================
# Modes
# =============================================================================

class EvaluationMode(str, Enum):
    training = "training"
    testing = "testing"


PASS_THRESHOLD: float = 75.0


# =============================================================================
# Pydantic Models (v2)
# =============================================================================

class ConversationTurn(BaseModel):
    """
    Minimal turn representation expected by the evaluator.
    You can pass richer turn objects upstream; the evaluator only needs these fields.
    """
    turn_index: int = Field(..., description="0-based index of the turn in the conversation.")
    speaker: Literal["customer", "agent", "system"] = Field(
        ..., description="Who produced this turn."
    )
    text: str = Field(..., description="Raw text content of the turn.")


class RubricDimensionScore(BaseModel):
    name: str = Field(..., description="Dimension name, e.g., 'Empathy', 'Accuracy', 'Policy Compliance'.")
    score: float = Field(..., ge=0, le=100, description="0-100 score for this dimension.")
    weight: float = Field(..., ge=0, le=1, description="Weight of the dimension (0-1).")
    justification: str = Field(..., description="Concise justification referencing observable evidence.")


class PolicyFlags(BaseModel):
    safety_or_policy_violation: bool = Field(
        ..., description="True if agent content violates policy/safety or disallowed behavior."
    )
    privacy_risk: bool = Field(
        ..., description="True if agent requests/shares sensitive personal data unnecessarily."
    )
    hallucination_risk: bool = Field(
        ..., description="True if agent makes claims not supported by conversation/context."
    )
    other_flags: List[str] = Field(
        default_factory=list, description="Any additional flags, short strings."
    )


class EvidenceItem(BaseModel):
    """
    Evidence grounded in the conversation.
    Keep quotes short.
    """
    turn_index: int = Field(..., description="Turn index where evidence appears.")
    speaker: Literal["customer", "agent", "system"] = Field(..., description="Speaker of the evidence turn.")
    quote: str = Field(
        ..., description="Short direct quote (<= 25 words) that supports the evaluator claim."
    )
    note: str = Field(..., description="Why this quote matters.")


class TurnByTurnFeedback(BaseModel):
    turn_index: int = Field(..., description="Turn index being assessed.")
    agent_action_quality: Literal["excellent", "good", "okay", "poor"] = Field(
        ..., description="Qualitative rating of the agent's action at this turn."
    )
    what_went_well: List[str] = Field(
        default_factory=list, description="Positive observations for this specific turn."
    )
    what_to_improve: List[str] = Field(
        default_factory=list, description="Improvements for this specific turn."
    )
    suggested_rewrite: Optional[str] = Field(
        default=None, description="Optional improved rewrite of the agent message for this turn."
    )


class AnalysisReport(BaseModel):
    """
    Output of the Analyzer prompt.

    This is an intermediate structured analysis that the Synthesizer will use to
    generate the mode-specific FinalReport.
    """
    conversation_summary: str = Field(
        ..., description="Brief factual summary of the customer issue and agent actions."
    )
    customer_intent: str = Field(
        ..., description="Best-guess of the customer's intent / goal."
    )
    outcome: Literal["resolved", "partially_resolved", "unresolved", "escalated"] = Field(
        ..., description="Outcome status by end of conversation."
    )

    dimension_scores: List[RubricDimensionScore] = Field(
        ..., description="Per-dimension scores (0-100) with weights summing ~1."
    )
    overall_score: float = Field(..., ge=0, le=100, description="Weighted overall score (0-100).")

    strengths: List[str] = Field(
        default_factory=list, description="Top strengths based on evidence."
    )
    improvement_areas: List[str] = Field(
        default_factory=list, description="Top improvement areas based on evidence."
    )

    policy_flags: PolicyFlags = Field(
        ..., description="Any safety/policy risks detected."
    )
    evidence: List[EvidenceItem] = Field(
        default_factory=list, description="Evidence items grounding the evaluation."
    )

    turn_feedback: List[TurnByTurnFeedback] = Field(
        default_factory=list,
        description="Turn-by-turn feedback. In training mode this will be used heavily."
    )

    missing_info_questions: List[str] = Field(
        default_factory=list,
        description="Questions the agent should have asked to resolve the issue faster."
    )


class FinalReport(BaseModel):
    """
    Output of the Synthesizer prompt.

    In training mode: encouraging, actionable coaching + turn-by-turn feedback.
    In testing mode: objective assessment with pass/fail at 75% threshold.
    """
    mode: EvaluationMode = Field(..., description="Evaluation mode used to produce this report.")
    overall_score: float = Field(..., ge=0, le=100, description="Overall score (0-100).")

    passed: Optional[bool] = Field(
        default=None,
        description="Testing mode only: pass/fail based on 75% threshold. Null in training mode."
    )
    pass_threshold: Optional[float] = Field(
        default=None,
        description="Testing mode only: the threshold used (typically 75). Null in training mode."
    )

    headline: str = Field(
        ..., description="One-line summary of performance."
    )
    strengths: List[str] = Field(
        default_factory=list, description="Key strengths (concise bullets)."
    )
    improvements: List[str] = Field(
        default_factory=list, description="Key improvements (concise bullets)."
    )

    coaching_plan: Optional[List[str]] = Field(
        default=None,
        description="Training mode only: step-by-step improvement plan. Null in testing mode."
    )
    turn_by_turn_feedback: Optional[List[TurnByTurnFeedback]] = Field(
        default=None,
        description="Training mode only: detailed turn-by-turn feedback. Null in testing mode."
    )

    testing_notes: Optional[List[str]] = Field(
        default=None,
        description="Testing mode only: objective notes, rubric alignment, and reasons for pass/fail."
    )

    policy_flags: PolicyFlags = Field(
        ..., description="Same policy flags, carried forward from analysis."
    )


# =============================================================================
# JSON Validation Helpers
# =============================================================================

T = TypeVar("T", bound=BaseModel)


def json_schema_for_model(model: Type[BaseModel]) -> Dict[str, Any]:
    """
    Returns the JSON schema for a Pydantic v2 model (as a dict).
    Useful for embedding in prompts.
    """
    return model.model_json_schema()


def _extract_json_candidate(text: str) -> str:
    """
    Best-effort extraction:
    - Removes ```json fences
    - Extracts the first top-level JSON object or array found
    """
    if not isinstance(text, str):
        raise TypeError("LLM output must be a string.")

    # Remove code fences
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", text.strip(), flags=re.IGNORECASE)

    # Try to find a JSON object/array by locating the first '{' or '[' and last matching '}' or ']'
    obj_start = cleaned.find("{")
    arr_start = cleaned.find("[")
    if obj_start == -1 and arr_start == -1:
        return cleaned  # let JSON parser raise a meaningful error

    if obj_start == -1 or (arr_start != -1 and arr_start < obj_start):
        start = arr_start
        end = cleaned.rfind("]")
    else:
        start = obj_start
        end = cleaned.rfind("}")

    if end == -1 or end <= start:
        return cleaned

    return cleaned[start : end + 1].strip()


def parse_json(text: str) -> Any:
    """
    Parses JSON from LLM output, with best-effort extraction.
    Raises json.JSONDecodeError if parsing fails.
    """
    candidate = _extract_json_candidate(text)
    return json.loads(candidate)


def validate_llm_json(text: str, model: Type[T]) -> Tuple[Optional[T], Optional[str]]:
    """
    Validate LLM output JSON against a Pydantic model.

    Returns: (instance, error_message)
    - instance is None on failure
    - error_message is None on success
    """
    try:
        data = parse_json(text)
        instance = model.model_validate(data)
        return instance, None
    except (json.JSONDecodeError, ValidationError, TypeError) as e:
        return None, str(e)


def strict_dump(instance: BaseModel) -> str:
    """
    Dumps a Pydantic model to compact JSON (no trailing commentary).
    """
    return instance.model_dump_json(indent=None)


# =============================================================================
# Prompt Builders
# =============================================================================

def _system_style_guardrails() -> str:
    """System instructions to ensure clean JSON output from the LLM."""
    return (
        "You are an evaluation engine for a Virtual Customer AI training platform (VCAI).\n"
        "You MUST follow these rules:\n"
        "1) Output MUST be valid JSON only (no markdown, no code fences, no extra text).\n"
        "2) JSON MUST match the provided schema exactly (all required keys, correct types).\n"
        "3) Use only evidence from the provided conversation.\n"
        "4) Quotes must be short (<= 25 words) and attributable to a specific turn.\n"
        "5) If unsure, be conservative and note uncertainty in justifications.\n"
        "6) Do not add any preamble or postamble - just output the JSON.\n"
    )


def build_analyzer_prompt(rubric: List[Dict[str, Any]]) -> str:
    """
    Analyzer Prompt:
    Input: conversation turns + rubric
    Output: AnalysisReport JSON
    
    Args:
        rubric: List of rubric dimensions with name, weight, and description
    
    Returns:
        System prompt string for the analyzer
    """
    schema = json.dumps(json_schema_for_model(AnalysisReport), ensure_ascii=False, indent=2)

    rubric_text = json.dumps(
        rubric,
        ensure_ascii=False,
        indent=2,
    )

    return (
        f"{_system_style_guardrails()}\n\n"
        "TASK: Analyze the conversation and produce an AnalysisReport JSON.\n\n"
        "RUBRIC:\n"
        f"{rubric_text}\n\n"
        "SCHEMA (AnalysisReport JSON Schema):\n"
        f"{schema}\n\n"
        "SCORING INSTRUCTIONS:\n"
        "- Score each rubric dimension 0-100 based on observable evidence.\n"
        "- Use the given weights (0-1). Weights should sum to 1.0 (or very close).\n"
        "- overall_score MUST be the weighted average: sum(score * weight) for all dimensions.\n"
        "- Set policy_flags appropriately if any safety/policy/privacy risks are present.\n"
        "- Provide evidence items grounded in specific turns with short quotes.\n"
        "- Provide turn_feedback for each agent turn (one TurnByTurnFeedback per agent turn).\n"
        "- List missing_info_questions that could have helped resolve the issue faster.\n\n"
        "INPUT FORMAT YOU WILL RECEIVE:\n"
        "{\n"
        '  "conversation": [\n'
        '    { "turn_index": 0, "speaker": "customer|agent|system", "text": "..." },\n'
        '    ...\n'
        '  ]\n'
        "}\n\n"
        "OUTPUT:\n"
        "- Return ONLY the AnalysisReport as valid JSON.\n"
        "- No preamble, no code fences, no explanations.\n"
    )


def build_synthesizer_prompt() -> str:
    """
    Synthesizer Prompt:
    Input: AnalysisReport + mode
    Output: FinalReport JSON (tone and fields differ by mode)
    
    Returns:
        System prompt string for the synthesizer
    """
    schema = json.dumps(json_schema_for_model(FinalReport), ensure_ascii=False, indent=2)

    return (
        f"{_system_style_guardrails()}\n\n"
        "TASK: Produce a FinalReport JSON using the given AnalysisReport and evaluation mode.\n\n"
        "SCHEMA (FinalReport JSON Schema):\n"
        f"{schema}\n\n"
        "MODE RULES:\n\n"
        "A) TRAINING MODE:\n"
        "- Tone: encouraging, coaching, supportive, and constructive.\n"
        "- MUST include: coaching_plan (list of actionable improvement steps).\n"
        "- MUST include: turn_by_turn_feedback (detailed feedback from AnalysisReport).\n"
        "- MUST set: passed = null, pass_threshold = null.\n"
        "- Focus on growth and learning opportunities.\n\n"
        "B) TESTING MODE:\n"
        "- Tone: professional, objective, rubric-aligned, factual.\n"
        "- MUST include: passed (boolean) using 75% threshold.\n"
        "- MUST include: pass_threshold = 75.0.\n"
        "- MUST include: testing_notes (objective reasons for the score and pass/fail decision).\n"
        "- MUST set: coaching_plan = null, turn_by_turn_feedback = null.\n"
        "- Focus on objective assessment against standards.\n\n"
        "PASS/FAIL LOGIC:\n"
        f"- PASS_THRESHOLD = {PASS_THRESHOLD}\n"
        f"- passed = true if overall_score >= {PASS_THRESHOLD}, else false\n"
        "- This applies ONLY in testing mode.\n\n"
        "INPUT FORMAT YOU WILL RECEIVE:\n"
        "{\n"
        '  "mode": "training" or "testing",\n'
        '  "analysis": { ...AnalysisReport JSON... }\n'
        "}\n\n"
        "OUTPUT:\n"
        "- Return ONLY the FinalReport as valid JSON.\n"
        "- No preamble, no code fences, no explanations.\n"
    )


# =============================================================================
# Convenience: Prompt Constants
# =============================================================================

# NOTE: Analyzer prompt depends on rubric; use function-based builder.
SYNTHESIZER_PROMPT: str = build_synthesizer_prompt()


# =============================================================================
# Default rubric template (example)
# =============================================================================

DEFAULT_RUBRIC: List[Dict[str, Any]] = [
    {
        "name": "Problem Understanding",
        "weight": 0.20,
        "description": "Accurately understands the customer's intent, asks clarifying questions when needed, and avoids making assumptions."
    },
    {
        "name": "Accuracy & Helpfulness",
        "weight": 0.25,
        "description": "Provides correct, relevant information and clear next steps that address the customer's needs."
    },
    {
        "name": "Empathy & Tone",
        "weight": 0.20,
        "description": "Communicates in a polite, empathetic manner with tone appropriate for the situation and customer emotions."
    },
    {
        "name": "Efficiency",
        "weight": 0.15,
        "description": "Resolves issues efficiently, avoids unnecessary back-and-forth, and keeps responses concise and on-point."
    },
    {
        "name": "Policy & Safety",
        "weight": 0.20,
        "description": "Avoids disallowed content, protects customer privacy, follows safety guidelines, and adheres to company policies."
    },
]


# Validate that default rubric weights sum to 1.0
_total_weight = sum(dim["weight"] for dim in DEFAULT_RUBRIC)
assert abs(_total_weight - 1.0) < 0.01, f"DEFAULT_RUBRIC weights must sum to 1.0, got {_total_weight}"