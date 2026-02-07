# evaluation/config.py
"""
Evaluation Configuration - Settings, thresholds, weights, and checkpoint definitions.

This file contains all configurable parameters for the evaluation system.
Adjust these values to tune evaluation behavior without changing code.
"""

from typing import Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════════
# Skill Configuration
# ═══════════════════════════════════════════════════════════════════════════════

class SkillConfig(BaseModel):
    """Configuration for a single skill category."""
    
    key: str = Field(..., description="Skill identifier")
    name_en: str = Field(..., description="Display name (English)")
    name_ar: str = Field(..., description="Display name (Arabic)")
    
    # Default weight when skill is tested
    # These get adjusted dynamically based on conversation profile
    default_weight: float = Field(
        default=0.125,
        ge=0.0, le=1.0,
        description="Default weight (8 skills = 0.125 each)"
    )
    
    # Minimum score to "pass" this skill
    pass_threshold: int = Field(
        default=60,
        ge=0, le=100,
        description="Minimum score to pass this skill"
    )
    
    # What triggers this skill being "tested"
    triggers: list[str] = Field(
        default_factory=list,
        description="Conversation events that trigger this skill evaluation"
    )
    
    # Description for prompts
    description: str = Field(
        default="",
        description="What this skill measures (for LLM prompts)"
    )


# Default skill configurations
SKILL_CONFIGS: dict[str, SkillConfig] = {
    "rapport_building": SkillConfig(
        key="rapport_building",
        name_en="Rapport Building",
        name_ar="بناء العلاقة",
        default_weight=0.12,
        pass_threshold=60,
        triggers=["opening_stage", "greeting", "small_talk"],
        description="Ability to establish connection and trust with the customer"
    ),
    "active_listening": SkillConfig(
        key="active_listening",
        name_en="Active Listening",
        name_ar="الاستماع الفعال",
        default_weight=0.12,
        pass_threshold=60,
        triggers=["customer_shares_info", "customer_expresses_concern"],
        description="Demonstrating understanding of customer's words and concerns"
    ),
    "needs_discovery": SkillConfig(
        key="needs_discovery",
        name_en="Needs Discovery",
        name_ar="اكتشاف الاحتياجات",
        default_weight=0.14,
        pass_threshold=65,
        triggers=["discovery_stage", "questions_asked"],
        description="Asking the right questions to understand customer needs"
    ),
    "product_knowledge": SkillConfig(
        key="product_knowledge",
        name_en="Product Knowledge",
        name_ar="المعرفة بالمنتج",
        default_weight=0.14,
        pass_threshold=70,
        triggers=["factual_claim", "feature_discussion", "rag_needed"],
        description="Accuracy and depth of property/product information"
    ),
    "objection_handling": SkillConfig(
        key="objection_handling",
        name_en="Objection Handling",
        name_ar="التعامل مع الاعتراضات",
        default_weight=0.14,
        pass_threshold=65,
        triggers=["objection_raised", "customer_hesitant", "price_concern"],
        description="Addressing customer concerns effectively"
    ),
    "emotional_intelligence": SkillConfig(
        key="emotional_intelligence",
        name_en="Emotional Intelligence",
        name_ar="الذكاء العاطفي",
        default_weight=0.12,
        pass_threshold=60,
        triggers=["emotion_change", "customer_frustrated", "customer_happy"],
        description="Reading and responding to customer emotions appropriately"
    ),
    "closing_skills": SkillConfig(
        key="closing_skills",
        name_en="Closing Skills",
        name_ar="مهارات الإغلاق",
        default_weight=0.12,
        pass_threshold=65,
        triggers=["closing_signal", "closing_stage", "commitment_request"],
        description="Recognizing opportunities and moving toward commitment"
    ),
    "communication_clarity": SkillConfig(
        key="communication_clarity",
        name_en="Communication Clarity",
        name_ar="وضوح التواصل",
        default_weight=0.10,
        pass_threshold=60,
        triggers=["always"],  # Always evaluated
        description="Clear, professional, and effective communication"
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint Configuration
# ═══════════════════════════════════════════════════════════════════════════════

class CheckpointConfig(BaseModel):
    """Configuration for a conversation checkpoint/milestone."""
    
    key: str = Field(..., description="Checkpoint identifier")
    name_en: str = Field(..., description="Display name (English)")
    name_ar: str = Field(..., description="Display name (Arabic)")
    
    description: str = Field(..., description="What achieving this means")
    
    # Detection criteria
    detection_criteria: str = Field(
        ...,
        description="How to detect if checkpoint was achieved (for LLM)"
    )
    
    # Importance
    is_critical: bool = Field(
        default=False,
        description="Is this a critical checkpoint (affects pass/fail significantly)"
    )
    
    # Order (for display)
    order: int = Field(default=0, description="Display order")
    
    # Icon
    icon_achieved: str = Field(default="✅", description="Icon when achieved")
    icon_missed: str = Field(default="❌", description="Icon when missed")


# Default checkpoint configurations
CHECKPOINT_CONFIGS: dict[str, CheckpointConfig] = {
    "rapport_established": CheckpointConfig(
        key="rapport_established",
        name_en="Rapport Established",
        name_ar="تم بناء العلاقة",
        description="Customer moved from neutral/skeptical to comfortable",
        detection_criteria="Customer shows positive engagement, uses friendly language, or explicitly expresses comfort",
        is_critical=False,
        order=1
    ),
    "needs_identified": CheckpointConfig(
        key="needs_identified",
        name_en="Needs Identified",
        name_ar="تم تحديد الاحتياجات",
        description="Salesperson discovered customer's key requirements",
        detection_criteria="Salesperson asked discovery questions AND customer revealed budget, timeline, preferences, or must-haves",
        is_critical=True,
        order=2
    ),
    "value_demonstrated": CheckpointConfig(
        key="value_demonstrated",
        name_en="Value Demonstrated",
        name_ar="تم عرض القيمة",
        description="Customer's interest increased after product presentation",
        detection_criteria="Customer emotion improved or showed increased interest after salesperson presented features/benefits",
        is_critical=False,
        order=3
    ),
    "objection_handled": CheckpointConfig(
        key="objection_handled",
        name_en="Objection Handled",
        name_ar="تم معالجة الاعتراض",
        description="Customer concern was addressed and interest recovered",
        detection_criteria="Customer raised objection/concern AND salesperson addressed it AND customer's interest didn't decrease significantly",
        is_critical=True,
        order=4
    ),
    "closing_signal_recognized": CheckpointConfig(
        key="closing_signal_recognized",
        name_en="Closing Signal Recognized",
        name_ar="تم التعرف على إشارة الإغلاق",
        description="Salesperson recognized and acted on customer's buying signal",
        detection_criteria="Customer gave closing signal (asked about next steps, availability, payment) AND salesperson responded appropriately within 1-2 turns",
        is_critical=True,
        order=5
    ),
    "commitment_achieved": CheckpointConfig(
        key="commitment_achieved",
        name_en="Commitment Achieved",
        name_ar="تم الحصول على الالتزام",
        description="Customer agreed to next step (viewing, reservation, purchase)",
        detection_criteria="Customer explicitly agreed to schedule viewing, make reservation, proceed with purchase, or other concrete next step",
        is_critical=False,
        order=6
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Dynamic Weight Multipliers
# ═══════════════════════════════════════════════════════════════════════════════

class WeightMultipliers(BaseModel):
    """
    Multipliers for adjusting skill weights based on conversation profile.
    
    Example: If objections were raised, objection_handling weight increases.
    """
    
    # If objections were raised, boost objection_handling weight
    objection_raised_boost: float = Field(
        default=1.5,
        description="Multiply objection_handling weight by this if objections occurred"
    )
    
    # If closing signals given, boost closing_skills weight
    closing_signal_boost: float = Field(
        default=1.8,
        description="Multiply closing_skills weight by this if closing signals occurred"
    )
    
    # If RAG topics discussed, boost product_knowledge weight
    rag_needed_boost: float = Field(
        default=1.4,
        description="Multiply product_knowledge weight by this if factual claims were made"
    )
    
    # If significant emotion changes, boost emotional_intelligence weight
    emotion_volatility_boost: float = Field(
        default=1.3,
        description="Multiply emotional_intelligence weight if multiple emotion transitions"
    )
    
    # If conversation was short (< 5 turns), reduce weight for complex skills
    short_conversation_penalty: float = Field(
        default=0.7,
        description="Multiply complex skills weight by this for short conversations"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation Thresholds
# ═══════════════════════════════════════════════════════════════════════════════

class EvaluationThresholds(BaseModel):
    """Score thresholds for pass/fail and assessments."""
    
    # Overall pass threshold
    overall_pass: int = Field(
        default=75,
        ge=0, le=100,
        description="Overall score needed to pass (testing mode)"
    )
    
    # Borderline range (close to threshold)
    borderline_range: int = Field(
        default=5,
        description="Points from threshold to be considered 'borderline'"
    )
    
    # Score interpretation
    excellent_threshold: int = Field(default=90, description="Score >= this is excellent")
    good_threshold: int = Field(default=75, description="Score >= this is good")
    adequate_threshold: int = Field(default=60, description="Score >= this is adequate")
    needs_improvement_threshold: int = Field(default=40, description="Score >= this needs improvement")
    # Below needs_improvement = poor
    
    # Minimum skills that must pass
    min_skills_passed: int = Field(
        default=5,
        description="Minimum number of tested skills that must pass threshold"
    )
    
    # Critical skill failure = automatic fail
    critical_skill_failure_threshold: int = Field(
        default=40,
        description="If any critical skill scores below this, automatic fail"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# LLM Configuration for Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

class EvaluationLLMConfig(BaseModel):
    """LLM settings for evaluation pipeline."""
    
    # Model selection
    use_same_model_as_conversation: bool = Field(
        default=True,
        description="Use the same Qwen model loaded for conversation"
    )
    
    # If not using same model, specify which to load
    analyzer_model: str = Field(
        default="Qwen/Qwen2.5-7B-Instruct",
        description="Model for analysis pass"
    )
    synthesizer_model: str = Field(
        default="Qwen/Qwen2.5-7B-Instruct",
        description="Model for synthesis pass"
    )
    
    # Generation parameters
    max_tokens_analysis: int = Field(
        default=4096,
        description="Max tokens for analysis generation"
    )
    max_tokens_synthesis: int = Field(
        default=4096,
        description="Max tokens for synthesis generation"
    )
    
    temperature: float = Field(
        default=0.3,
        description="Temperature for evaluation (lower = more consistent)"
    )
    
    # Retry on parse failure
    max_retries: int = Field(
        default=2,
        description="Max retries if JSON parsing fails"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Main Settings Class
# ═══════════════════════════════════════════════════════════════════════════════

class EvaluationSettings(BaseSettings):
    """
    Main evaluation settings.
    Can be overridden via environment variables.
    """
    
    # Feature flags
    enable_turn_feedback: bool = Field(
        default=True,
        description="Generate turn-by-turn feedback (training mode)"
    )
    enable_fact_verification: bool = Field(
        default=True,
        description="Verify claims against RAG context"
    )
    include_raw_analysis: bool = Field(
        default=False,
        description="Include raw analysis in final report (for debugging)"
    )
    
    # Thresholds
    thresholds: EvaluationThresholds = Field(
        default_factory=EvaluationThresholds
    )
    
    # Weight multipliers
    weight_multipliers: WeightMultipliers = Field(
        default_factory=WeightMultipliers
    )
    
    # LLM config
    llm_config: EvaluationLLMConfig = Field(
        default_factory=EvaluationLLMConfig
    )
    
    # Timeouts
    evaluation_timeout_seconds: int = Field(
        default=120,
        description="Max time for full evaluation pipeline"
    )
    
    # Background task
    run_in_background: bool = Field(
        default=True,
        description="Run evaluation as background task (vs synchronous)"
    )
    
    model_config = {
        "env_prefix": "EVAL_",
        "env_file": ".env",
        "extra": "ignore"  # ← ADD THIS: Ignore extra env vars
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════════

def get_skill_config(skill_key: str) -> Optional[SkillConfig]:
    """Get configuration for a skill by key."""
    return SKILL_CONFIGS.get(skill_key)


def get_checkpoint_config(checkpoint_key: str) -> Optional[CheckpointConfig]:
    """Get configuration for a checkpoint by key."""
    return CHECKPOINT_CONFIGS.get(checkpoint_key)


def get_all_skill_keys() -> list[str]:
    """Get all skill keys."""
    return list(SKILL_CONFIGS.keys())


def get_all_checkpoint_keys() -> list[str]:
    """Get all checkpoint keys in order."""
    return sorted(
        CHECKPOINT_CONFIGS.keys(),
        key=lambda k: CHECKPOINT_CONFIGS[k].order
    )


def get_score_assessment(score: int, thresholds: EvaluationThresholds = None) -> str:
    """Convert score to assessment string."""
    if thresholds is None:
        thresholds = EvaluationThresholds()
    
    if score >= thresholds.excellent_threshold:
        return "excellent"
    elif score >= thresholds.good_threshold:
        return "good"
    elif score >= thresholds.adequate_threshold:
        return "adequate"
    elif score >= thresholds.needs_improvement_threshold:
        return "needs_improvement"
    else:
        return "poor"


def get_default_settings() -> EvaluationSettings:
    """Get default evaluation settings."""
    return EvaluationSettings()


# ═══════════════════════════════════════════════════════════════════════════════
# Singleton instance
# ═══════════════════════════════════════════════════════════════════════════════

# Use this in other modules:
# from evaluation.config import settings
settings = get_default_settings()