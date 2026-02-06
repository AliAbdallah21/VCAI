# evaluation/prompts.py
"""
VCAI Evaluation Module - LLM Prompts + JSON validation helpers (Pydantic v2)

This module provides:
- JSON parsing/validation helpers for LLM outputs  
- Prompt builders for:
  - Analyzer: conversation -> AnalysisReport JSON (from schemas/analysis_schema.py)
  - Synthesizer: analysis + mode -> FinalReport JSON (from schemas/report_schema.py)

NOTE: This file does NOT define the Pydantic models (AnalysisReport, FinalReport).
Those are defined in:
- evaluation/schemas/analysis_schema.py (by Ali)
- evaluation/schemas/report_schema.py (by Ali)

This separation allows Ismail to use the prompts without circular dependencies.

Modes:
- training: encouraging tone + suggestions + turn-by-turn feedback
- testing: professional/objective + pass/fail using 75% threshold
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

import json
import re

from pydantic import BaseModel, ValidationError


# =============================================================================
# Constants
# =============================================================================

PASS_THRESHOLD: float = 75.0


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


def extract_json_from_response(text: str) -> Any:
    """
    Alias for parse_json() - used by pipeline nodes.
    Extracts and parses JSON from LLM response.
    """
    return parse_json(text)


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


def validate_analysis_json(text: str, AnalysisReport: Type[T]) -> Tuple[Optional[T], Optional[str]]:
    """
    Validate analyzer output against AnalysisReport model.
    Convenience wrapper for validate_llm_json.
    """
    return validate_llm_json(text, AnalysisReport)


def validate_report_json(text: str, FinalReport: Type[T]) -> Tuple[Optional[T], Optional[str]]:
    """
    Validate synthesizer output against FinalReport model.
    Convenience wrapper for validate_llm_json.
    """
    return validate_llm_json(text, FinalReport)


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
        "You are an evaluation engine for VCAI (Virtual Customer AI) - a sales training platform.\n"
        "You evaluate real estate salespeople on their conversations with virtual customers.\n\n"
        "OUTPUT RULES - CRITICAL:\n"
        "1) Output MUST be valid JSON only (no markdown, no code fences, no extra text).\n"
        "2) JSON MUST match the provided schema exactly (all required keys, correct types).\n"
        "3) Use only evidence from the provided conversation.\n"
        "4) Quotes must be short (<= 25 words) and attributable to a specific turn.\n"
        "5) If unsure, be conservative and note uncertainty in justifications.\n"
        "6) Do not add any preamble or postamble - just output the JSON.\n"
        "7) All Arabic text must use proper UTF-8 encoding.\n"
    )


def build_analyzer_prompt(
    transcript: List[Dict[str, Any]],
    emotion_log: List[Dict[str, Any]],
    rag_context: List[Dict[str, Any]],
    skill_configs: List[Dict[str, Any]],
    checkpoint_configs: List[Dict[str, Any]],
    AnalysisReport: Type[BaseModel]
) -> str:
    """
    Build the complete analyzer prompt.
    
    The analyzer performs Pass 1: deep analysis of what happened in the conversation.
    
    Args:
        transcript: List of conversation turns with turn_number, speaker, text
        emotion_log: List of emotion detections with turn_number, emotion, confidence
        rag_context: List of RAG retrievals with query, documents, sources
        skill_configs: The 8 skills with name, name_ar, weight, description
        checkpoint_configs: The 6 checkpoints with name, name_ar, criteria
        AnalysisReport: Pydantic model class for schema generation
    
    Returns:
        Complete system + user prompt for the analyzer
    """
    schema = json.dumps(json_schema_for_model(AnalysisReport), ensure_ascii=False, indent=2)
    
    transcript_json = json.dumps(transcript, ensure_ascii=False, indent=2)
    emotion_json = json.dumps(emotion_log, ensure_ascii=False, indent=2)
    rag_json = json.dumps(rag_context, ensure_ascii=False, indent=2)
    skills_json = json.dumps(skill_configs, ensure_ascii=False, indent=2)
    checkpoints_json = json.dumps(checkpoint_configs, ensure_ascii=False, indent=2)

    return (
        f"{_system_style_guardrails()}\n\n"
        "═══════════════════════════════════════════════════════════════════\n"
        "TASK: CONVERSATION ANALYSIS (Pass 1)\n"
        "═══════════════════════════════════════════════════════════════════\n\n"
        "You will analyze a real estate sales conversation and produce a detailed AnalysisReport.\n"
        "This is Pass 1 of a 2-pass evaluation system.\n\n"
        
        "CONTEXT - The VCAI System:\n"
        "- This is a training platform for real estate salespeople\n"
        "- Salespeople practice with AI virtual customers (personas)\n"
        "- Conversations are in Arabic (Egyptian dialect) or Arabic/English mix\n"
        "- Your job: analyze what happened and score the salesperson's performance\n\n"
        
        "═══════════════════════════════════════════════════════════════════\n"
        "THE 8 SALES SKILLS (What You're Evaluating):\n"
        "═══════════════════════════════════════════════════════════════════\n\n"
        f"{skills_json}\n\n"
        
        "═══════════════════════════════════════════════════════════════════\n"
        "THE 6 CHECKPOINTS (Milestones to Track):\n"
        "═══════════════════════════════════════════════════════════════════\n\n"
        f"{checkpoints_json}\n\n"
        
        "═══════════════════════════════════════════════════════════════════\n"
        "CONVERSATION DATA:\n"
        "═══════════════════════════════════════════════════════════════════\n\n"
        
        "TRANSCRIPT (Complete conversation):\n"
        f"{transcript_json}\n\n"
        
        "EMOTION LOG (Customer emotions detected at each turn):\n"
        f"{emotion_json}\n\n"
        
        "RAG CONTEXT (Knowledge base documents retrieved during conversation):\n"
        f"{rag_json}\n\n"
        
        "═══════════════════════════════════════════════════════════════════\n"
        "YOUR ANALYSIS TASKS:\n"
        "═══════════════════════════════════════════════════════════════════\n\n"
        
        "1. UNDERSTAND THE CONVERSATION:\n"
        "   - What was the customer looking for?\n"
        "   - What stages did the conversation go through? (greeting, discovery, presentation, objection, closing)\n"
        "   - What topics were discussed? (price, location, features, payment, etc.)\n"
        "   - Were there objections? What were they? How were they handled?\n"
        "   - Were there closing signals? Were they recognized?\n"
        "   - How did emotions change throughout?\n\n"
        
        "2. EVALUATE EACH SKILL (0-100):\n"
        "   - Score ONLY skills that were actually tested in this conversation\n"
        "   - If a skill wasn't tested (e.g., no objections = can't test objection_handling), note this\n"
        "   - Use specific evidence from turns to justify scores\n"
        "   - Be fair: don't penalize for situations that didn't arise\n\n"
        
        "3. CHECK FACTS:\n"
        "   - Did the salesperson make claims about the property?\n"
        "   - Compare claims against the RAG context (knowledge base)\n"
        "   - Flag any inaccuracies (CRITICAL: wrong info can lose sales)\n\n"
        
        "4. TRACK CHECKPOINTS:\n"
        "   - Which of the 6 milestones were achieved?\n"
        "   - Provide evidence (which turn, what happened)\n\n"
        
        "5. PROVIDE TURN-BY-TURN FEEDBACK:\n"
        "   - For each salesperson turn, assess quality (excellent/good/okay/poor)\n"
        "   - Note what went well and what could improve\n"
        "   - Suggest rewrites for poor turns\n\n"
        
        "═══════════════════════════════════════════════════════════════════\n"
        "SCORING GUIDELINES:\n"
        "═══════════════════════════════════════════════════════════════════\n\n"
        
        "- Scores are 0-100 for each skill\n"
        "- Use the weights from skill_configs (they sum to 1.0)\n"
        "- overall_score = weighted average: Σ(skill_score × skill_weight)\n"
        "- Be evidence-based: every score must be justified with specific quotes/observations\n"
        "- Short quotes only: <= 25 words\n\n"
        
        "DYNAMIC EVALUATION (Important!):\n"
        "- Only evaluate what was ACTUALLY TESTED\n"
        "- If no objections arose, set objection_handling weight to 0 (or very low)\n"
        "- If no closing opportunity, set closing_skills weight to 0 (or very low)\n"
        "- This makes evaluation FAIR\n\n"
        
        "═══════════════════════════════════════════════════════════════════\n"
        "OUTPUT SCHEMA:\n"
        "═══════════════════════════════════════════════════════════════════\n\n"
        f"{schema}\n\n"
        
        "═══════════════════════════════════════════════════════════════════\n"
        "CRITICAL REMINDERS:\n"
        "═══════════════════════════════════════════════════════════════════\n\n"
        "✓ Output ONLY valid JSON (no preamble, no code fences)\n"
        "✓ Match the schema exactly\n"
        "✓ Use evidence from actual turns\n"
        "✓ Be fair: only evaluate what was tested\n"
        "✓ Verify facts against RAG context\n"
        "✓ Track all 6 checkpoints\n"
        "✓ Provide turn-by-turn feedback\n\n"
        
        "BEGIN YOUR ANALYSIS NOW.\n"
    )


# Separate system prompts for training vs testing mode
ANALYZER_SYSTEM_PROMPT = _system_style_guardrails()

SYNTHESIZER_SYSTEM_PROMPT_TRAINING = (
    f"{_system_style_guardrails()}\n\n"
    "═══════════════════════════════════════════════════════════════════\n"
    "MODE: TRAINING (Coaching & Development)\n"
    "═══════════════════════════════════════════════════════════════════\n\n"
    "Your tone must be:\n"
    "✓ Encouraging and supportive\n"
    "✓ Focused on growth and learning\n"
    "✓ Constructive (highlight strengths BEFORE improvements)\n"
    "✓ Actionable (give specific tips, not vague advice)\n"
    "✓ Patient (this is training, not testing)\n\n"
    
    "Example Training Feedback:\n"
    "\"Great job building rapport in turns 1-2! Your friendly greeting made the customer comfortable.\n"
    "When the customer mentioned budget concerns in turn 6, you could strengthen your response by \n"
    "acknowledging their concern first before offering solutions. Try: 'أنا فاهم إن الميزانية مهمة...' \n"
    "This shows empathy before problem-solving.\"\n\n"
)

SYNTHESIZER_SYSTEM_PROMPT_TESTING = (
    f"{_system_style_guardrails()}\n\n"
    "═══════════════════════════════════════════════════════════════════\n"
    "MODE: TESTING (Professional Assessment)\n"
    "═══════════════════════════════════════════════════════════════════\n\n"
    "Your tone must be:\n"
    "✓ Professional and objective\n"
    "✓ Fact-based and rubric-aligned\n"
    "✓ Clear about pass/fail criteria (75% threshold)\n"
    "✓ Specific about strengths and weaknesses\n"
    "✓ No coaching suggestions (just assessment)\n\n"
    
    "Example Testing Feedback:\n"
    "\"Overall Score: 72/100 — FAILED (below 75% threshold)\n\n"
    "Strengths:\n"
    "- Rapport building: 85% - Excellent friendly greeting and tone\n"
    "- Product knowledge: 90% - Accurate information provided\n\n"
    "Weaknesses:\n"
    "- Closing skills: 60% - Missed closing signal in turn 8 when customer asked about availability\n"
    "- Objection handling: 65% - Addressed concern but delayed response reduced effectiveness\n\n"
    "Result: Not ready for certification. Retake after practicing closing signal recognition.\"\n\n"
)


def build_synthesizer_prompt(
    analysis_report: Dict[str, Any],
    quick_stats: Dict[str, Any],
    mode: str,
    FinalReport: Type[BaseModel]
) -> Tuple[str, str]:
    """
    Build the complete synthesizer prompt.
    
    The synthesizer performs Pass 2: generate the final report based on analysis.
    
    Args:
        analysis_report: The AnalysisReport from Pass 1 (as dict)
        quick_stats: Quick statistics (duration, turns, final emotion)
        mode: "training" or "testing"
        FinalReport: Pydantic model class for schema generation
    
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    schema = json.dumps(json_schema_for_model(FinalReport), ensure_ascii=False, indent=2)
    
    analysis_json = json.dumps(analysis_report, ensure_ascii=False, indent=2)
    stats_json = json.dumps(quick_stats, ensure_ascii=False, indent=2)
    
    # Choose system prompt based on mode
    if mode == "training":
        system_prompt = SYNTHESIZER_SYSTEM_PROMPT_TRAINING
    else:
        system_prompt = SYNTHESIZER_SYSTEM_PROMPT_TESTING
    
    user_prompt = (
        "═══════════════════════════════════════════════════════════════════\n"
        "TASK: GENERATE FINAL REPORT (Pass 2)\n"
        "═══════════════════════════════════════════════════════════════════\n\n"
        
        f"EVALUATION MODE: {mode.upper()}\n\n"
        
        "You have received a detailed AnalysisReport from Pass 1.\n"
        "Your job: Generate a mode-appropriate FinalReport.\n\n"
        
        "═══════════════════════════════════════════════════════════════════\n"
        "INPUT DATA:\n"
        "═══════════════════════════════════════════════════════════════════\n\n"
        
        "ANALYSIS REPORT (from Pass 1):\n"
        f"{analysis_json}\n\n"
        
        "QUICK STATS:\n"
        f"{stats_json}\n\n"
        
        "═══════════════════════════════════════════════════════════════════\n"
        "MODE-SPECIFIC REQUIREMENTS:\n"
        "═══════════════════════════════════════════════════════════════════\n\n"
    )
    
    if mode == "training":
        user_prompt += (
            "TRAINING MODE Requirements:\n\n"
            "MUST INCLUDE:\n"
            "✓ coaching_plan: List of 3-5 actionable improvement steps\n"
            "✓ turn_by_turn_feedback: Detailed feedback for each agent turn (from analysis)\n"
            "✓ headline: Encouraging summary (e.g., 'Strong start with room to grow!')\n"
            "✓ strengths: Top 3 things done well\n"
            "✓ improvements: Top 3 areas to practice\n\n"
            
            "MUST SET TO NULL:\n"
            "✗ passed: null (no pass/fail in training)\n"
            "✗ pass_threshold: null\n"
            "✗ testing_notes: null\n\n"
            
            "TONE: Encouraging, coaching, growth-focused\n\n"
        )
    else:  # testing
        user_prompt += (
            "TESTING MODE Requirements:\n\n"
            "MUST INCLUDE:\n"
            f"✓ passed: boolean (true if overall_score >= {PASS_THRESHOLD}, else false)\n"
            f"✓ pass_threshold: {PASS_THRESHOLD}\n"
            "✓ testing_notes: List of objective reasons for score and pass/fail decision\n"
            "✓ headline: Professional summary (e.g., 'Competent performance with minor gaps')\n"
            "✓ strengths: Top 3 objective strengths\n"
            "✓ improvements: Top 3 objective weaknesses\n\n"
            
            "MUST SET TO NULL:\n"
            "✗ coaching_plan: null (no coaching in testing)\n"
            "✗ turn_by_turn_feedback: null\n\n"
            
            "TONE: Professional, objective, rubric-aligned\n\n"
            
            "PASS/FAIL LOGIC:\n"
            f"- If overall_score >= {PASS_THRESHOLD}: PASSED (ready for certification)\n"
            f"- If overall_score < {PASS_THRESHOLD}: FAILED (needs more practice)\n\n"
        )
    
    user_prompt += (
        "═══════════════════════════════════════════════════════════════════\n"
        "OUTPUT SCHEMA:\n"
        "═══════════════════════════════════════════════════════════════════\n\n"
        f"{schema}\n\n"
        
        "═══════════════════════════════════════════════════════════════════\n"
        "CRITICAL REMINDERS:\n"
        "═══════════════════════════════════════════════════════════════════\n\n"
        "✓ Output ONLY valid JSON (no preamble, no code fences)\n"
        "✓ Match the schema exactly for this mode\n"
        "✓ Use appropriate tone for the mode\n"
        "✓ Carry forward policy_flags from analysis\n"
        "✓ Reference the analysis but synthesize (don't just copy)\n\n"
        
        "BEGIN GENERATING THE FINAL REPORT NOW.\n"
    )
    
    return system_prompt, user_prompt


# =============================================================================
# Export constants
# =============================================================================

__all__ = [
    "PASS_THRESHOLD",
    "extract_json_from_response",
    "parse_json",
    "validate_llm_json",
    "validate_analysis_json",
    "validate_report_json",
    "strict_dump",
    "build_analyzer_prompt",
    "build_synthesizer_prompt",
    "ANALYZER_SYSTEM_PROMPT",
    "SYNTHESIZER_SYSTEM_PROMPT_TRAINING",
    "SYNTHESIZER_SYSTEM_PROMPT_TESTING",
]