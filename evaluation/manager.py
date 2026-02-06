# Evaluation Manager - Main entry point
"""
Evaluation Manager - Menna's Implementation
This is the main orchestrator for the VCAI Evaluation System
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json


class EvaluationMode(Enum):
    """Evaluation modes"""
    TRAINING = "training"  # Learning mode - encouraging feedback
    TESTING = "testing"    # Assessment mode - pass/fail


@dataclass
class ConversationData:
    """Data structure for conversation information"""
    session_id: str
    transcript: List[Dict[str, Any]]  # List of turns with speaker and message
    emotion_log: List[Dict[str, Any]]  # List of emotion states per turn
    rag_context: Dict[str, Any]  # Knowledge base facts
    customer_persona: str  # easy, medium, hard
    start_time: str
    end_time: str
    duration_seconds: int
    mode: EvaluationMode


@dataclass
class DynamicWeights:
    """Dynamic weights for skills based on conversation content"""
    rapport_building: float = 0.125  # Default: equal weight (1/8 = 12.5%)
    active_listening: float = 0.125
    needs_discovery: float = 0.125
    product_knowledge: float = 0.125
    objection_handling: float = 0.125
    emotional_intelligence: float = 0.125
    closing_skills: float = 0.125
    communication_clarity: float = 0.125
    
    def validate(self) -> bool:
        """Ensure weights sum to 1.0"""
        total = sum([
            self.rapport_building,
            self.active_listening,
            self.needs_discovery,
            self.product_knowledge,
            self.objection_handling,
            self.emotional_intelligence,
            self.closing_skills,
            self.communication_clarity
        ])
        return abs(total - 1.0) < 0.01  # Allow small floating point errors


@dataclass
class EvaluationReport:
    """Complete evaluation report structure"""
    session_id: str
    overall_score: float
    passed: bool
    mode: str
    
    # Skill scores
    skill_scores: Dict[str, float]
    
    # Checkpoints
    checkpoints: Dict[str, bool]
    
    # Strengths and improvements
    top_strengths: List[str]
    areas_to_improve: List[str]
    
    # Turn-by-turn feedback
    turn_feedback: List[Dict[str, Any]]
    
    # Recommendations
    recommended_practice: List[str]
    
    # Metadata
    timestamp: str
    duration_seconds: int
    total_turns: int
    final_emotion: str


class WeightCalculator:
    """
    Calculates dynamic weights based on conversation analysis
    This is a key part of Menna's responsibility
    """
    
    def __init__(self):
        self.base_weight = 0.125  # 1/8 for equal distribution
        
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
                - factual_claims: Number of factual statements made
                
        Returns:
            DynamicWeights object with adjusted weights
        """
        weights = DynamicWeights()
        
        # Extract conversation characteristics
        topics = conversation_analysis.get('topics_discussed', [])
        objections_count = conversation_analysis.get('objections_count', 0)
        closing_signals = conversation_analysis.get('closing_signals_count', 0)
        questions_asked = conversation_analysis.get('questions_asked', 0)
        factual_claims = conversation_analysis.get('factual_claims_count', 0)
        emotional_moments = conversation_analysis.get('emotional_moments_count', 0)
        
        # Start with base weights
        base = self.base_weight
        
        # Calculate boost/reduction multipliers
        # Objection Handling: Boost if objections exist
        if objections_count > 0:
            # High objections = important skill
            objection_multiplier = min(1 + (objections_count * 0.4), 2.5)
            weights.objection_handling = base * objection_multiplier
        else:
            # No objections = minimal weight
            weights.objection_handling = base * 0.3
            
        # Closing Skills: Boost if closing signals present
        if closing_signals > 0:
            closing_multiplier = min(1 + (closing_signals * 0.5), 2.8)
            weights.closing_skills = base * closing_multiplier
        else:
            # No signals = minimal weight
            weights.closing_skills = base * 0.3
            
        # Product Knowledge: Boost if many factual claims
        if factual_claims > 0:
            knowledge_multiplier = min(1 + (factual_claims * 0.15), 2.2)
            weights.product_knowledge = base * knowledge_multiplier
        else:
            weights.product_knowledge = base * 0.5
            
        # Needs Discovery: Boost if many questions asked
        if questions_asked > 0:
            discovery_multiplier = min(1 + (questions_asked * 0.2), 2.0)
            weights.needs_discovery = base * discovery_multiplier
        else:
            weights.needs_discovery = base * 0.5
            
        # Emotional Intelligence: Boost if emotional moments detected
        if emotional_moments > 0:
            emotion_multiplier = min(1 + (emotional_moments * 0.3), 2.0)
            weights.emotional_intelligence = base * emotion_multiplier
        else:
            weights.emotional_intelligence = base * 0.7
            
        # Rapport Building: Always important at the start
        # Give standard weight unless it was very short conversation
        total_turns = conversation_analysis.get('total_turns', 0)
        if total_turns < 4:
            weights.rapport_building = base * 0.6  # Quick convo = less rapport time
        else:
            weights.rapport_building = base * 1.2  # Normal importance
            
        # Active Listening & Communication Clarity: Standard weights
        # These are always evaluated
        weights.active_listening = base * 1.0
        weights.communication_clarity = base * 1.0
        
        # Normalize weights to sum to 1.0
        weights = self._normalize_weights(weights)
        
        return weights
    
    def _normalize_weights(self, weights: DynamicWeights) -> DynamicWeights:
        """Normalize weights so they sum to 1.0"""
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
        
        # Normalize
        weights.rapport_building /= total
        weights.active_listening /= total
        weights.needs_discovery /= total
        weights.product_knowledge /= total
        weights.objection_handling /= total
        weights.emotional_intelligence /= total
        weights.closing_skills /= total
        weights.communication_clarity /= total
        
        return weights


class ReportFormatter:
    """
    Formats the AI-generated report into the final structure
    This is part of Menna's responsibility
    """
    
    def __init__(self, mode: EvaluationMode, pass_threshold: float = 75.0):
        self.mode = mode
        self.pass_threshold = pass_threshold
        
    def format_report(
        self,
        conversation_data: ConversationData,
        ai_analysis: Dict[str, Any],
        ai_report: Dict[str, Any],
        weights: DynamicWeights
    ) -> EvaluationReport:
        """
        Format the AI output into the final report structure
        
        Args:
            conversation_data: Original conversation data
            ai_analysis: Analysis from first AI pass
            ai_report: Report from second AI pass
            weights: The dynamic weights used
            
        Returns:
            EvaluationReport ready for database storage and display
        """
        
        # Extract skill scores from AI report
        skill_scores = ai_report.get('skill_scores', {})
        
        # Calculate overall score using dynamic weights
        overall_score = self._calculate_overall_score(skill_scores, weights)
        
        # Determine pass/fail
        passed = overall_score >= self.pass_threshold if self.mode == EvaluationMode.TESTING else True
        
        # Extract checkpoints
        checkpoints = ai_report.get('checkpoints', {})
        
        # Format feedback based on mode
        if self.mode == EvaluationMode.TRAINING:
            top_strengths = ai_report.get('strengths', [])[:3]
            areas_to_improve = ai_report.get('areas_to_improve', [])[:3]
            turn_feedback = ai_report.get('turn_feedback', [])
            recommended_practice = ai_report.get('recommended_practice', [])
        else:  # TESTING mode
            top_strengths = ai_report.get('strengths', [])[:3]
            areas_to_improve = ai_report.get('areas_to_improve', [])[:3]
            turn_feedback = []  # No detailed feedback in testing mode
            recommended_practice = ai_report.get('recommended_practice', []) if not passed else []
        
        # Get final emotion
        final_emotion = "neutral"
        if conversation_data.emotion_log:
            final_emotion = conversation_data.emotion_log[-1].get('emotion', 'neutral')
        
        # Create report
        report = EvaluationReport(
            session_id=conversation_data.session_id,
            overall_score=round(overall_score, 2),
            passed=passed,
            mode=self.mode.value,
            skill_scores={
                'rapport_building': round(skill_scores.get('rapport_building', 0), 2),
                'active_listening': round(skill_scores.get('active_listening', 0), 2),
                'needs_discovery': round(skill_scores.get('needs_discovery', 0), 2),
                'product_knowledge': round(skill_scores.get('product_knowledge', 0), 2),
                'objection_handling': round(skill_scores.get('objection_handling', 0), 2),
                'emotional_intelligence': round(skill_scores.get('emotional_intelligence', 0), 2),
                'closing_skills': round(skill_scores.get('closing_skills', 0), 2),
                'communication_clarity': round(skill_scores.get('communication_clarity', 0), 2),
            },
            checkpoints={
                'rapport_established': checkpoints.get('rapport_established', False),
                'needs_identified': checkpoints.get('needs_identified', False),
                'value_demonstrated': checkpoints.get('value_demonstrated', False),
                'objection_handled': checkpoints.get('objection_handled', False),
                'closing_signal_recognized': checkpoints.get('closing_signal_recognized', False),
                'commitment_achieved': checkpoints.get('commitment_achieved', False),
            },
            top_strengths=top_strengths,
            areas_to_improve=areas_to_improve,
            turn_feedback=turn_feedback,
            recommended_practice=recommended_practice,
            timestamp=conversation_data.end_time,
            duration_seconds=conversation_data.duration_seconds,
            total_turns=len(conversation_data.transcript),
            final_emotion=final_emotion
        )
        
        return report
    
    def _calculate_overall_score(
        self,
        skill_scores: Dict[str, float],
        weights: DynamicWeights
    ) -> float:
        """Calculate weighted overall score"""
        
        overall = (
            skill_scores.get('rapport_building', 0) * weights.rapport_building +
            skill_scores.get('active_listening', 0) * weights.active_listening +
            skill_scores.get('needs_discovery', 0) * weights.needs_discovery +
            skill_scores.get('product_knowledge', 0) * weights.product_knowledge +
            skill_scores.get('objection_handling', 0) * weights.objection_handling +
            skill_scores.get('emotional_intelligence', 0) * weights.emotional_intelligence +
            skill_scores.get('closing_skills', 0) * weights.closing_skills +
            skill_scores.get('communication_clarity', 0) * weights.communication_clarity
        )
        
        return overall


class EvaluationManager:
    """
    Main Orchestrator - Menna's Core Implementation
    
    This class coordinates the entire evaluation process:
    1. Receives conversation data from Bakr
    2. Calculates dynamic weights
    3. Calls Ismail's AI pipeline
    4. Formats the final report
    5. Returns to Bakr for storage
    """
    
    def __init__(self, ai_pipeline, mode: EvaluationMode = EvaluationMode.TRAINING):
        """
        Initialize the evaluation manager
        
        Args:
            ai_pipeline: Ismail's AI pipeline instance
            mode: Training or Testing mode
        """
        self.ai_pipeline = ai_pipeline
        self.mode = mode
        self.weight_calculator = WeightCalculator()
        self.report_formatter = ReportFormatter(mode)
        
    def evaluate_conversation(self, conversation_data: ConversationData) -> EvaluationReport:
        """
        Main orchestration method - this is what Bakr calls
        
        Args:
            conversation_data: Complete conversation data from database
            
        Returns:
            EvaluationReport ready for storage and display
        """
        
        print(f"[MENNA] Starting evaluation for session {conversation_data.session_id}")
        print(f"[MENNA] Mode: {self.mode.value}")
        
        # Step 1: Run AI Analysis Pass (Ismail's code)
        print("[MENNA] Step 1: Running AI analysis...")
        ai_analysis = self.ai_pipeline.analyze_conversation(
            transcript=conversation_data.transcript,
            emotion_log=conversation_data.emotion_log,
            rag_context=conversation_data.rag_context
        )
        print(f"[MENNA] Analysis complete. Found {ai_analysis.get('objections_count', 0)} objections, "
              f"{ai_analysis.get('closing_signals_count', 0)} closing signals")
        
        # Step 2: Calculate Dynamic Weights
        print("[MENNA] Step 2: Calculating dynamic weights...")
        weights = self.weight_calculator.calculate_weights(ai_analysis)
        print(f"[MENNA] Weights calculated. Objection handling weight: {weights.objection_handling:.2%}, "
              f"Closing skills weight: {weights.closing_skills:.2%}")
        
        # Validate weights
        if not weights.validate():
            raise ValueError("Dynamic weights do not sum to 1.0!")
        
        # Step 3: Run AI Report Generation (Ismail's code)
        print("[MENNA] Step 3: Generating AI report...")
        ai_report = self.ai_pipeline.generate_report(
            analysis=ai_analysis,
            weights=weights,
            mode=self.mode.value
        )
        print(f"[MENNA] Report generated. Overall AI score: {ai_report.get('overall_score', 0)}")
        
        # Step 4: Format Final Report
        print("[MENNA] Step 4: Formatting final report...")
        final_report = self.report_formatter.format_report(
            conversation_data=conversation_data,
            ai_analysis=ai_analysis,
            ai_report=ai_report,
            weights=weights
        )
        print(f"[MENNA] Final score: {final_report.overall_score}/100, "
              f"Passed: {final_report.passed}")
        
        # Step 5: Return to Bakr for storage
        print("[MENNA] Evaluation complete! Returning to Bakr...")
        return final_report
    
    def get_quick_stats(self, conversation_data: ConversationData) -> Dict[str, Any]:
        """
        Generate quick stats immediately after call ends
        No AI needed - just basic calculations
        
        Args:
            conversation_data: Conversation data
            
        Returns:
            Dictionary with quick stats
        """
        
        # Count checkpoints that can be determined without AI
        checkpoints_count = 0
        
        # Simple heuristics for quick feedback
        if len(conversation_data.transcript) >= 2:
            checkpoints_count += 1  # Rapport likely established
        
        if len(conversation_data.transcript) >= 4:
            checkpoints_count += 1  # Needs likely discussed
        
        # Final emotion
        final_emotion = "neutral"
        if conversation_data.emotion_log:
            final_emotion = conversation_data.emotion_log[-1].get('emotion', 'neutral')
        
        return {
            'duration_seconds': conversation_data.duration_seconds,
            'total_turns': len(conversation_data.transcript),
            'final_emotion': final_emotion,
            'checkpoints_estimated': f"{checkpoints_count}/6",
            'status': 'ready_for_evaluation'
        }


# Example usage and testing
if __name__ == "__main__":
    # Mock AI Pipeline for testing
    class MockAIPipeline:
        def analyze_conversation(self, transcript, emotion_log, rag_context):
            return {
                'topics_discussed': ['price', 'location', 'features'],
                'objections_count': 2,
                'closing_signals_count': 1,
                'questions_asked': 5,
                'factual_claims_count': 8,
                'emotional_moments_count': 3,
                'total_turns': len(transcript)
            }
        
        def generate_report(self, analysis, weights, mode):
            return {
                'overall_score': 78.0,
                'skill_scores': {
                    'rapport_building': 85,
                    'active_listening': 70,
                    'needs_discovery': 75,
                    'product_knowledge': 90,
                    'objection_handling': 65,
                    'emotional_intelligence': 72,
                    'closing_skills': 60,
                    'communication_clarity': 88
                },
                'checkpoints': {
                    'rapport_established': True,
                    'needs_identified': True,
                    'value_demonstrated': True,
                    'objection_handled': True,
                    'closing_signal_recognized': False,
                    'commitment_achieved': False
                },
                'strengths': [
                    'Excellent product knowledge',
                    'Strong rapport building',
                    'Clear communication'
                ],
                'areas_to_improve': [
                    'Missed closing signal in turn 10',
                    'Objection handling timing',
                    'Closing skills'
                ],
                'turn_feedback': [],
                'recommended_practice': [
                    'Practice closing signal recognition',
                    'Practice objection handling with price concerns'
                ]
            }
    
    # Test the manager
    print("="*60)
    print("TESTING EVALUATION MANAGER")
    print("="*60)
    
    mock_pipeline = MockAIPipeline()
    manager = EvaluationManager(mock_pipeline, EvaluationMode.TRAINING)
    
    # Mock conversation data
    mock_conversation = ConversationData(
        session_id="test-123",
        transcript=[
            {'speaker': 'salesperson', 'message': 'السلام عليكم'},
            {'speaker': 'customer', 'message': 'أهلاً'},
            {'speaker': 'salesperson', 'message': 'ممكن أعرف ميزانيتك؟'},
            {'speaker': 'customer', 'message': '2 مليون'},
        ],
        emotion_log=[
            {'turn': 1, 'emotion': 'neutral'},
            {'turn': 2, 'emotion': 'interested'},
        ],
        rag_context={'price': 2500000, 'size': 150},
        customer_persona='medium',
        start_time='2026-02-06T10:00:00',
        end_time='2026-02-06T10:04:32',
        duration_seconds=272,
        mode=EvaluationMode.TRAINING
    )
    
    # Test quick stats
    print("\n1. Testing Quick Stats...")
    quick_stats = manager.get_quick_stats(mock_conversation)
    print(json.dumps(quick_stats, indent=2, ensure_ascii=False))
    
    # Test full evaluation
    print("\n2. Testing Full Evaluation...")
    report = manager.evaluate_conversation(mock_conversation)
    print(f"\nFinal Report:")
    print(f"  Overall Score: {report.overall_score}/100")
    print(f"  Passed: {report.passed}")
    print(f"  Strengths: {report.top_strengths}")
    print(f"  Improvements: {report.areas_to_improve}")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
