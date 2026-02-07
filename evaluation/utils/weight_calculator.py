# evaluation/utils/weight_calculator.py
"""
Dynamic Weight Calculator for VCAI Evaluation System
Author: Menna Khaled

This module calculates dynamic weights based on conversation analysis.
The weights are adjusted based on what actually happened in the conversation.
"""

from typing import Dict, Optional
import logging

from evaluation.schemas.analysis_schema import ConversationProfile
from evaluation.config import (
    SKILL_CONFIGS,
    WeightMultipliers,
    get_all_skill_keys
)


logger = logging.getLogger(__name__)


def calculate_dynamic_weights(
    profile: ConversationProfile,
    multipliers: Optional[WeightMultipliers] = None
) -> Dict[str, float]:
    """
    Calculate dynamic weights based on conversation profile.
    
    This is Menna's core responsibility: adjusting skill weights based on
    what was actually tested in the conversation.
    
    Args:
        profile: ConversationProfile from the analyzer
        multipliers: Weight multipliers (uses defaults if None)
        
    Returns:
        Dictionary mapping skill keys to weights (normalized to sum to 1.0)
        
    Example:
        >>> profile = ConversationProfile(
        ...     objections=[...],  # 2 objections
        ...     closing_signals=[...],  # 0 signals
        ... )
        >>> weights = calculate_dynamic_weights(profile)
        >>> # objection_handling will have high weight
        >>> # closing_skills will have low weight
    """
    if multipliers is None:
        from evaluation.config import settings
        multipliers = settings.weight_multipliers
    
    # Start with default weights from config
    weights = {}
    for skill_key, config in SKILL_CONFIGS.items():
        weights[skill_key] = config.default_weight
    
    logger.info(f"[WEIGHT_CALC] Starting with base weights")
    
    # Apply multipliers based on conversation profile
    
    # 1. Objection Handling
    if profile.objections and len(profile.objections) > 0:
        logger.info(f"[WEIGHT_CALC] Found {len(profile.objections)} objections - boosting objection_handling")
        weights["objection_handling"] *= multipliers.objection_raised_boost
    else:
        # Reduce weight if not tested
        logger.info(f"[WEIGHT_CALC] No objections - reducing objection_handling weight")
        weights["objection_handling"] *= multipliers.short_conversation_penalty
    
    # 2. Closing Skills
    if profile.closing_signals and len(profile.closing_signals) > 0:
        logger.info(f"[WEIGHT_CALC] Found {len(profile.closing_signals)} closing signals - boosting closing_skills")
        weights["closing_skills"] *= multipliers.closing_signal_boost
    else:
        logger.info(f"[WEIGHT_CALC] No closing signals - reducing closing_skills weight")
        weights["closing_skills"] *= multipliers.short_conversation_penalty
    
    # 3. Product Knowledge (based on RAG usage and factual claims)
    if profile.rag_was_needed or (profile.rag_topics_relevant and len(profile.rag_topics_relevant) > 0):

        logger.info(f"[WEIGHT_CALC] Found {len(profile.rag_retrievals)} RAG retrievals - boosting product_knowledge")
        weights["product_knowledge"] *= multipliers.rag_needed_boost
    
    # 4. Emotional Intelligence (based on emotion volatility)
    if profile.emotion_transitions and len(profile.emotion_transitions) > 3:
        logger.info(f"[WEIGHT_CALC] Found {len(profile.emotion_transitions)} emotion changes - boosting emotional_intelligence")
        weights["emotional_intelligence"] *= multipliers.emotion_volatility_boost
    
    # 5. Needs Discovery (based on discovery stage presence)
    from evaluation.schemas.analysis_schema import ConversationStage
    has_discovery = ConversationStage.NEEDS_DISCOVERY in (profile.stages_present or [])

    if not has_discovery:
        logger.info(f"[WEIGHT_CALC] No discovery stage - reducing needs_discovery weight")
        weights["needs_discovery"] *= multipliers.short_conversation_penalty
    
    # 6. Short conversations get reduced weights for complex skills
    if profile.total_turns < 5:
        logger.info(f"[WEIGHT_CALC] Short conversation ({profile.total_turns} turns) - reducing complex skills")
        weights["objection_handling"] *= multipliers.short_conversation_penalty
        weights["closing_skills"] *= multipliers.short_conversation_penalty
        weights["needs_discovery"] *= multipliers.short_conversation_penalty
    
    # Normalize weights to sum to 1.0
    normalized_weights = _normalize_weights(weights)
    
    logger.info(f"[WEIGHT_CALC] Final weights calculated and normalized")
    _log_weights(normalized_weights)
    
    return normalized_weights


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize weights so they sum to 1.0.
    
    Args:
        weights: Dictionary of weights
        
    Returns:
        Normalized weights
    """
    total = sum(weights.values())
    
    if total == 0:
        # Fallback to equal weights
        logger.warning("[WEIGHT_CALC] Total weight is 0, using equal weights")
        skill_keys = get_all_skill_keys()
        equal_weight = 1.0 / len(skill_keys)
        return {key: equal_weight for key in skill_keys}
    
    # Normalize
    normalized = {key: value / total for key, value in weights.items()}
    
    # Verify sum is close to 1.0
    final_sum = sum(normalized.values())
    if abs(final_sum - 1.0) > 0.01:
        logger.warning(f"[WEIGHT_CALC] Weight sum is {final_sum}, expected 1.0")
    
    return normalized


def _log_weights(weights: Dict[str, float]) -> None:
    """
    Log the calculated weights for debugging.
    
    Args:
        weights: Dictionary of weights
    """
    logger.info("[WEIGHT_CALC] Final weight distribution:")
    for skill_key, weight in sorted(weights.items(), key=lambda x: -x[1]):
        percentage = weight * 100
        logger.info(f"  {skill_key:30s}: {percentage:5.1f}%")


def get_weight_explanation(
    weights: Dict[str, float],
    profile: ConversationProfile
) -> Dict[str, str]:
    """
    Generate human-readable explanation for why weights were assigned.
    
    This is useful for training mode feedback.
    
    Args:
        weights: Calculated weights
        profile: ConversationProfile that was analyzed
        
    Returns:
        Dictionary mapping skill names to explanations
    """
    explanations = {}
    
    # Objection handling
    obj_count = len(profile.objections) if profile.objections else 0
    if obj_count > 0:
        explanations['objection_handling'] = (
            f"High weight ({weights['objection_handling']:.1%}) because "
            f"{obj_count} objection(s) were raised and needed to be handled"
        )
    else:
        explanations['objection_handling'] = (
            f"Low weight ({weights['objection_handling']:.1%}) because "
            "no objections were raised in this conversation"
        )
    
    # Closing skills
    closing_count = len(profile.closing_signals) if profile.closing_signals else 0
    if closing_count > 0:
        explanations['closing_skills'] = (
            f"High weight ({weights['closing_skills']:.1%}) because "
            f"{closing_count} closing signal(s) were detected"
        )
    else:
        explanations['closing_skills'] = (
            f"Low weight ({weights['closing_skills']:.1%}) because "
            "no closing signals were detected"
        )
    
    # Product knowledge
    rag_count = len(profile.rag_topics_relevant) if profile.rag_topics_relevant else 0
    if rag_count > 0:
        explanations['product_knowledge'] = (
            f"Weight ({weights['product_knowledge']:.1%}) based on "
            f"{rag_count} factual claims requiring product knowledge"
        )
    else:
        explanations['product_knowledge'] = (
            f"Standard weight ({weights['product_knowledge']:.1%}) - "
            "some product knowledge always required"
        )
    
    # Emotional intelligence
    emotion_count = len(profile.emotion_transitions) if profile.emotion_transitions else 0
    if emotion_count > 3:
        explanations['emotional_intelligence'] = (
            f"High weight ({weights['emotional_intelligence']:.1%}) because "
            f"{emotion_count} emotional transitions were detected"
        )
    else:
        explanations['emotional_intelligence'] = (
            f"Standard weight ({weights['emotional_intelligence']:.1%}) based on "
            f"{emotion_count} emotional transitions"
        )
    
    # Needs discovery
    from evaluation.schemas.analysis_schema import ConversationStage
    has_discovery = ConversationStage.NEEDS_DISCOVERY in (profile.stages_present or [])
    
    explanations['needs_discovery'] = (
        f"Weight ({weights['needs_discovery']:.1%}) - "
        f"{'discovery stage present' if has_discovery else 'minimal discovery detected'}"
    )
    
    # Rapport building
    explanations['rapport_building'] = (
        f"Weight ({weights['rapport_building']:.1%}) based on "
        f"{profile.total_turns} conversation turns"
    )
    
    # Always-evaluated skills
    explanations['active_listening'] = (
        f"Standard weight ({weights['active_listening']:.1%}) - "
        "always evaluated in every conversation"
    )
    explanations['communication_clarity'] = (
        f"Standard weight ({weights['communication_clarity']:.1%}) - "
        "always evaluated in every conversation"
    )
    
    return explanations


def validate_weights(weights: Dict[str, float]) -> bool:
    """
    Validate that weights are properly normalized.
    
    Args:
        weights: Dictionary of weights
        
    Returns:
        True if valid, False otherwise
    """
    # Check all skills are present
    skill_keys = get_all_skill_keys()
    if set(weights.keys()) != set(skill_keys):
        logger.error("[WEIGHT_CALC] Missing or extra skills in weights")
        return False
    
    # Check sum is close to 1.0
    total = sum(weights.values())
    if abs(total - 1.0) > 0.01:
        logger.error(f"[WEIGHT_CALC] Weights sum to {total}, expected 1.0")
        return False
    
    # Check all weights are positive
    if any(w < 0 for w in weights.values()):
        logger.error("[WEIGHT_CALC] Negative weights detected")
        return False
    
    return True