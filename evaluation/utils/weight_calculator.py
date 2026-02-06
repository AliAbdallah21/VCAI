"""
Weight Calculator for VCAI Evaluation System
Author: Mena Khaled

This module calculates dynamic weights based on conversation analysis.
The weights are adjusted based on what actually happened in the conversation.
"""

from typing import Dict, Any
from .models import DynamicWeights, ConversationAnalysis
from .config import EvaluationConfig


class WeightCalculator:
    """
    Calculates dynamic weights based on conversation analysis
    This is a key part of Menna's responsibility
    """
    
    def __init__(self, config: EvaluationConfig = None):
        """
        Initialize weight calculator
        
        Args:
            config: Configuration object (optional, uses defaults if None)
        """
        self.config = config or EvaluationConfig()
        self.base_weight = self.config.DEFAULT_BASE_WEIGHT
        
    def calculate_weights(self, conversation_analysis: Dict[str, Any]) -> DynamicWeights:
        """
        Calculate dynamic weights based on what actually happened in the conversation
        
        Args:
            conversation_analysis: Analysis from Ismail's AI pipeline containing:
                - topics_discussed: List of topics
                - objections_count: Number of objections raised
                - closing_signals_count: Number of closing signals
                - rapport_moments: Number of rapport-building moments
                - questions_asked: Number of discovery questions
                - factual_claims_count: Number of factual statements made
                - emotional_moments_count: Number of emotional moments
                - total_turns: Total conversation turns
                
        Returns:
            DynamicWeights object with adjusted weights
        """
        # Extract conversation characteristics
        topics = conversation_analysis.get('topics_discussed', [])
        objections_count = conversation_analysis.get('objections_count', 0)
        closing_signals = conversation_analysis.get('closing_signals_count', 0)
        questions_asked = conversation_analysis.get('questions_asked', 0)
        factual_claims = conversation_analysis.get('factual_claims_count', 0)
        emotional_moments = conversation_analysis.get('emotional_moments_count', 0)
        total_turns = conversation_analysis.get('total_turns', 0)
        
        # Initialize weights with base values
        weights = DynamicWeights()
        base = self.base_weight
        
        # Calculate each skill weight
        weights.objection_handling = self._calculate_objection_weight(
            objections_count, base
        )
        
        weights.closing_skills = self._calculate_closing_weight(
            closing_signals, base
        )
        
        weights.product_knowledge = self._calculate_knowledge_weight(
            factual_claims, base
        )
        
        weights.needs_discovery = self._calculate_discovery_weight(
            questions_asked, base
        )
        
        weights.emotional_intelligence = self._calculate_emotion_weight(
            emotional_moments, base
        )
        
        weights.rapport_building = self._calculate_rapport_weight(
            total_turns, base
        )
        
        # Active listening and communication clarity get standard weights
        # These are always evaluated
        weights.active_listening = base * self.config.ACTIVE_LISTENING_MULTIPLIER
        weights.communication_clarity = base * self.config.COMMUNICATION_CLARITY_MULTIPLIER
        
        # Normalize weights to sum to 1.0
        weights = self._normalize_weights(weights)
        
        return weights
    
    def _calculate_objection_weight(self, objections_count: int, base: float) -> float:
        """
        Calculate weight for objection handling skill
        
        Args:
            objections_count: Number of objections in conversation
            base: Base weight value
            
        Returns:
            Calculated weight for objection handling
        """
        if objections_count > 0:
            # High objections = important skill
            multiplier = min(
                1 + (objections_count * self.config.OBJECTION_BOOST_MULTIPLIER),
                self.config.OBJECTION_MAX_MULTIPLIER
            )
            return base * multiplier
        else:
            # No objections = minimal weight
            return base * self.config.OBJECTION_NO_OCCURRENCE_MULTIPLIER
    
    def _calculate_closing_weight(self, closing_signals: int, base: float) -> float:
        """
        Calculate weight for closing skills
        
        Args:
            closing_signals: Number of closing signals detected
            base: Base weight value
            
        Returns:
            Calculated weight for closing skills
        """
        if closing_signals > 0:
            # Closing signals present = important skill
            multiplier = min(
                1 + (closing_signals * self.config.CLOSING_BOOST_MULTIPLIER),
                self.config.CLOSING_MAX_MULTIPLIER
            )
            return base * multiplier
        else:
            # No signals = minimal weight
            return base * self.config.CLOSING_NO_OCCURRENCE_MULTIPLIER
    
    def _calculate_knowledge_weight(self, factual_claims: int, base: float) -> float:
        """
        Calculate weight for product knowledge skill
        
        Args:
            factual_claims: Number of factual claims made
            base: Base weight value
            
        Returns:
            Calculated weight for product knowledge
        """
        if factual_claims > 0:
            # Many factual claims = knowledge was important
            multiplier = min(
                1 + (factual_claims * self.config.KNOWLEDGE_BOOST_MULTIPLIER),
                self.config.KNOWLEDGE_MAX_MULTIPLIER
            )
            return base * multiplier
        else:
            return base * self.config.KNOWLEDGE_NO_OCCURRENCE_MULTIPLIER
    
    def _calculate_discovery_weight(self, questions_asked: int, base: float) -> float:
        """
        Calculate weight for needs discovery skill
        
        Args:
            questions_asked: Number of discovery questions asked
            base: Base weight value
            
        Returns:
            Calculated weight for needs discovery
        """
        if questions_asked > 0:
            # Many questions = discovery was important
            multiplier = min(
                1 + (questions_asked * self.config.DISCOVERY_BOOST_MULTIPLIER),
                self.config.DISCOVERY_MAX_MULTIPLIER
            )
            return base * multiplier
        else:
            return base * self.config.DISCOVERY_NO_OCCURRENCE_MULTIPLIER
    
    def _calculate_emotion_weight(self, emotional_moments: int, base: float) -> float:
        """
        Calculate weight for emotional intelligence skill
        
        Args:
            emotional_moments: Number of emotional moments detected
            base: Base weight value
            
        Returns:
            Calculated weight for emotional intelligence
        """
        if emotional_moments > 0:
            # Emotional moments present = EI was important
            multiplier = min(
                1 + (emotional_moments * self.config.EMOTION_BOOST_MULTIPLIER),
                self.config.EMOTION_MAX_MULTIPLIER
            )
            return base * multiplier
        else:
            return base * self.config.EMOTION_NO_OCCURRENCE_MULTIPLIER
    
    def _calculate_rapport_weight(self, total_turns: int, base: float) -> float:
        """
        Calculate weight for rapport building skill
        
        Args:
            total_turns: Total number of conversation turns
            base: Base weight value
            
        Returns:
            Calculated weight for rapport building
        """
        if total_turns < self.config.RAPPORT_SHORT_CONVERSATION_THRESHOLD:
            # Quick conversation = less rapport time
            return base * self.config.RAPPORT_SHORT_MULTIPLIER
        else:
            # Normal conversation = standard importance
            return base * self.config.RAPPORT_NORMAL_MULTIPLIER
    
    def _normalize_weights(self, weights: DynamicWeights) -> DynamicWeights:
        """
        Normalize weights so they sum to 1.0
        
        Args:
            weights: DynamicWeights object to normalize
            
        Returns:
            Normalized DynamicWeights object
        """
        total = sum([
            weights.rapport_building,
            weights.active_listening,
            weights.needs_discovery,
            weights.product_knowledge,
            weights.objection_handling,
            weights.emotional_intelligence,
            weights.closing_skills,
            weights.communication_clarity
        ])
        
        if total == 0:
            # Fallback to equal weights
            return DynamicWeights()
        
        # Normalize each weight
        weights.rapport_building /= total
        weights.active_listening /= total
        weights.needs_discovery /= total
        weights.product_knowledge /= total
        weights.objection_handling /= total
        weights.emotional_intelligence /= total
        weights.closing_skills /= total
        weights.communication_clarity /= total
        
        return weights
    
    def get_weight_explanation(
        self,
        weights: DynamicWeights,
        conversation_analysis: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate human-readable explanation for why weights were assigned
        
        Args:
            weights: Calculated weights
            conversation_analysis: Original conversation analysis
            
        Returns:
            Dictionary mapping skill names to explanations
        """
        explanations = {}
        
        # Objection handling
        obj_count = conversation_analysis.get('objections_count', 0)
        if obj_count > 0:
            explanations['objection_handling'] = (
                f"High weight ({weights.objection_handling:.1%}) because "
                f"{obj_count} objection(s) were raised"
            )
        else:
            explanations['objection_handling'] = (
                f"Low weight ({weights.objection_handling:.1%}) because "
                "no objections were raised"
            )
        
        # Closing skills
        closing_count = conversation_analysis.get('closing_signals_count', 0)
        if closing_count > 0:
            explanations['closing_skills'] = (
                f"High weight ({weights.closing_skills:.1%}) because "
                f"{closing_count} closing signal(s) detected"
            )
        else:
            explanations['closing_skills'] = (
                f"Low weight ({weights.closing_skills:.1%}) because "
                "no closing signals detected"
            )
        
        # Product knowledge
        claims_count = conversation_analysis.get('factual_claims_count', 0)
        explanations['product_knowledge'] = (
            f"Weight ({weights.product_knowledge:.1%}) based on "
            f"{claims_count} factual claim(s) made"
        )
        
        # Needs discovery
        questions_count = conversation_analysis.get('questions_asked', 0)
        explanations['needs_discovery'] = (
            f"Weight ({weights.needs_discovery:.1%}) based on "
            f"{questions_count} discovery question(s) asked"
        )
        
        # Emotional intelligence
        emotion_count = conversation_analysis.get('emotional_moments_count', 0)
        explanations['emotional_intelligence'] = (
            f"Weight ({weights.emotional_intelligence:.1%}) based on "
            f"{emotion_count} emotional moment(s)"
        )
        
        # Rapport building
        turns = conversation_analysis.get('total_turns', 0)
        explanations['rapport_building'] = (
            f"Weight ({weights.rapport_building:.1%}) based on "
            f"{turns} conversation turns"
        )
        
        # Standard weights
        explanations['active_listening'] = (
            f"Standard weight ({weights.active_listening:.1%}) - "
            "always evaluated"
        )
        explanations['communication_clarity'] = (
            f"Standard weight ({weights.communication_clarity:.1%}) - "
            "always evaluated"
        )
        
        return explanations
