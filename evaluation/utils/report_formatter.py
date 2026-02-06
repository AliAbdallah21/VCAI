"""
Report Formatter for VCAI Evaluation System
Author: Mena Khaled

This module formats AI-generated reports into the final structure.
"""

from typing import Dict, Any, List
from .models import (
    EvaluationReport,
    ConversationData,
    DynamicWeights,
    EvaluationMode
)
from .config import EvaluationConfig


class ReportFormatter:
    """
    Formats the AI-generated report into the final structure
    This is part of Menna's responsibility
    """
    
    def __init__(
        self,
        mode: EvaluationMode,
        config: EvaluationConfig = None
    ):
        """
        Initialize report formatter
        
        Args:
            mode: Evaluation mode (TRAINING or TESTING)
            config: Configuration object (optional)
        """
        self.mode = mode
        self.config = config or EvaluationConfig()
        
        # Set pass threshold based on mode
        if mode == EvaluationMode.TRAINING:
            self.pass_threshold = self.config.TRAINING_MODE_PASS_THRESHOLD
        else:
            self.pass_threshold = self.config.TESTING_MODE_PASS_THRESHOLD
        
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
        passed = self._determine_pass_fail(overall_score)
        
        # Extract checkpoints
        checkpoints = self._extract_checkpoints(ai_report)
        
        # Format feedback based on mode
        top_strengths, areas_to_improve, turn_feedback, recommended_practice = (
            self._format_feedback(ai_report, passed)
        )
        
        # Get final emotion
        final_emotion = self._get_final_emotion(conversation_data)
        
        # Create and return report
        report = EvaluationReport(
            session_id=conversation_data.session_id,
            overall_score=round(overall_score, self.config.SCORE_DECIMAL_PLACES),
            passed=passed,
            mode=self.mode.value,
            skill_scores=self._round_skill_scores(skill_scores),
            checkpoints=checkpoints,
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
        """
        Calculate weighted overall score
        
        Args:
            skill_scores: Dictionary of skill scores (0-100)
            weights: DynamicWeights object
            
        Returns:
            Overall score (0-100)
        """
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
    
    def _determine_pass_fail(self, overall_score: float) -> bool:
        """
        Determine if the evaluation passed or failed
        
        Args:
            overall_score: Overall score (0-100)
            
        Returns:
            True if passed, False otherwise
        """
        if self.mode == EvaluationMode.TRAINING:
            # Always pass in training mode
            return True
        else:
            # Check against threshold in testing mode
            return overall_score >= self.pass_threshold
    
    def _extract_checkpoints(self, ai_report: Dict[str, Any]) -> Dict[str, bool]:
        """
        Extract and validate checkpoints from AI report
        
        Args:
            ai_report: AI-generated report
            
        Returns:
            Dictionary of checkpoint names to boolean values
        """
        checkpoints = ai_report.get('checkpoints', {})
        
        # Ensure all required checkpoints are present
        required_checkpoints = {
            'rapport_established': False,
            'needs_identified': False,
            'value_demonstrated': False,
            'objection_handled': False,
            'closing_signal_recognized': False,
            'commitment_achieved': False,
        }
        
        # Update with actual values from AI
        for checkpoint_name in required_checkpoints:
            if checkpoint_name in checkpoints:
                required_checkpoints[checkpoint_name] = checkpoints[checkpoint_name]
        
        return required_checkpoints
    
    def _format_feedback(
        self,
        ai_report: Dict[str, Any],
        passed: bool
    ) -> tuple:
        """
        Format feedback based on evaluation mode
        
        Args:
            ai_report: AI-generated report
            passed: Whether the evaluation passed
            
        Returns:
            Tuple of (top_strengths, areas_to_improve, turn_feedback, recommended_practice)
        """
        if self.mode == EvaluationMode.TRAINING:
            # Training mode: Full feedback
            top_strengths = ai_report.get('strengths', [])[:self.config.MAX_STRENGTHS_DISPLAY]
            areas_to_improve = ai_report.get('areas_to_improve', [])[:self.config.MAX_IMPROVEMENTS_DISPLAY]
            turn_feedback = ai_report.get('turn_feedback', [])
            recommended_practice = ai_report.get('recommended_practice', [])
            
        else:
            # Testing mode: Limited feedback
            top_strengths = ai_report.get('strengths', [])[:self.config.MAX_STRENGTHS_DISPLAY]
            areas_to_improve = ai_report.get('areas_to_improve', [])[:self.config.MAX_IMPROVEMENTS_DISPLAY]
            turn_feedback = []  # No detailed turn-by-turn feedback in testing
            
            # Only show practice recommendations if failed
            if not passed:
                recommended_practice = ai_report.get('recommended_practice', [])
            else:
                recommended_practice = []
        
        return top_strengths, areas_to_improve, turn_feedback, recommended_practice
    
    def _round_skill_scores(self, skill_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Round all skill scores to specified decimal places
        
        Args:
            skill_scores: Dictionary of skill scores
            
        Returns:
            Dictionary with rounded scores
        """
        rounded_scores = {}
        
        for skill_name in self.config.SKILL_NAMES:
            score = skill_scores.get(skill_name, 0)
            rounded_scores[skill_name] = round(score, self.config.SCORE_DECIMAL_PLACES)
        
        return rounded_scores
    
    def _get_final_emotion(self, conversation_data: ConversationData) -> str:
        """
        Extract the final emotion from conversation data
        
        Args:
            conversation_data: Conversation data
            
        Returns:
            Final emotion string
        """
        if conversation_data.emotion_log and len(conversation_data.emotion_log) > 0:
            return conversation_data.emotion_log[-1].get('emotion', 'neutral')
        return 'neutral'
    
    def generate_summary_text(self, report: EvaluationReport) -> str:
        """
        Generate a human-readable summary of the evaluation report
        
        Args:
            report: EvaluationReport object
            
        Returns:
            Formatted summary text
        """
        lines = []
        lines.append("=" * 60)
        lines.append(f"EVALUATION REPORT - Session {report.session_id}")
        lines.append("=" * 60)
        lines.append(f"Mode: {report.mode.upper()}")
        lines.append(f"Overall Score: {report.overall_score}/100")
        lines.append(f"Result: {'PASSED ✓' if report.passed else 'NEEDS IMPROVEMENT ✗'}")
        lines.append(f"Duration: {report.duration_seconds}s ({report.total_turns} turns)")
        lines.append(f"Final Emotion: {report.final_emotion}")
        lines.append("")
        
        # Checkpoints
        lines.append("CHECKPOINTS:")
        checkpoint_count = sum(1 for v in report.checkpoints.values() if v)
        lines.append(f"  Completed: {checkpoint_count}/{len(report.checkpoints)}")
        for checkpoint, achieved in report.checkpoints.items():
            status = "✓" if achieved else "✗"
            lines.append(f"  {status} {checkpoint.replace('_', ' ').title()}")
        lines.append("")
        
        # Skill Scores
        lines.append("SKILL SCORES:")
        for skill, score in report.skill_scores.items():
            bar = self._create_score_bar(score)
            lines.append(f"  {skill.replace('_', ' ').title():30s} {score:5.1f}/100 {bar}")
        lines.append("")
        
        # Strengths
        if report.top_strengths:
            lines.append("TOP STRENGTHS:")
            for i, strength in enumerate(report.top_strengths, 1):
                lines.append(f"  {i}. {strength}")
            lines.append("")
        
        # Areas to improve
        if report.areas_to_improve:
            lines.append("AREAS TO IMPROVE:")
            for i, area in enumerate(report.areas_to_improve, 1):
                lines.append(f"  {i}. {area}")
            lines.append("")
        
        # Recommendations
        if report.recommended_practice:
            lines.append("RECOMMENDED PRACTICE:")
            for i, practice in enumerate(report.recommended_practice, 1):
                lines.append(f"  {i}. {practice}")
            lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def _create_score_bar(self, score: float, width: int = 20) -> str:
        """
        Create a visual bar representation of a score
        
        Args:
            score: Score value (0-100)
            width: Width of the bar in characters
            
        Returns:
            String representation of the score bar
        """
        filled = int((score / 100) * width)
        empty = width - filled
        
        bar = "█" * filled + "░" * empty
        
        # Add color indicators (for terminals that support it)
        if score >= 80:
            return f"[{bar}]"  # Green zone
        elif score >= 60:
            return f"[{bar}]"  # Yellow zone
        else:
            return f"[{bar}]"  # Red zone
