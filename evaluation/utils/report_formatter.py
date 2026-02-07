# evaluation/utils/report_formatter.py
"""
Report Formatter for VCAI Evaluation System
Author: Menna Khaled

This module formats evaluation reports for different outputs.
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from evaluation.schemas.report_schema import FinalReport, QuickStats
from evaluation.schemas.analysis_schema import AnalysisReport
from evaluation.config import SKILL_CONFIGS, CHECKPOINT_CONFIGS


logger = logging.getLogger(__name__)


def format_for_database(report: FinalReport) -> Dict[str, Any]:
    """
    Convert FinalReport to JSON-serializable dict for database storage.
    
    Args:
        report: FinalReport from the evaluation pipeline
        
    Returns:
        Dictionary ready for JSON storage in database
        
    Example:
        >>> report = FinalReport(...)
        >>> db_data = format_for_database(report)
        >>> # Save to database: db.evaluations.insert(db_data)
    """
    logger.info("[FORMATTER] Converting report to database format")
    
    # Use Pydantic's model_dump to get JSON-serializable dict
    data = report.model_dump(mode='json')
    
    # Add metadata
    data['formatted_at'] = datetime.utcnow().isoformat()
    data['version'] = '1.0'
    
    return data


def format_executive_summary(report: FinalReport) -> str:
    """
    Generate a concise text summary of the evaluation report.
    
    Args:
        report: FinalReport from the evaluation pipeline
        
    Returns:
        Human-readable summary text
        
    Example:
        >>> summary = format_executive_summary(report)
        >>> print(summary)
        Overall Score: 78/100
        Status: PASSED
        Top Strengths: Excellent rapport building...
    """
    logger.info("[FORMATTER] Generating executive summary")
    
    lines = []
    
    # Header
    lines.append("EVALUATION SUMMARY")
    lines.append("=" * 60)
    
    # Overall results
    lines.append(f"Overall Score: {report.overall_score}/100")
    
    if report.mode == "testing":
        status = "PASSED ✓" if report.passed else "NEEDS IMPROVEMENT ✗"
        lines.append(f"Status: {status}")
        if report.pass_threshold:
            lines.append(f"Pass Threshold: {report.pass_threshold}")
    
    lines.append("")
    
    # Headline
    lines.append(f"Summary: {report.headline}")
    lines.append("")
    
    # Top strengths
    if report.strengths:
        lines.append("Top Strengths:")
        for i, strength in enumerate(report.strengths[:3], 1):
            lines.append(f"  {i}. {strength}")
        lines.append("")
    
    # Areas to improve
    if report.improvements:
        lines.append("Areas to Improve:")
        for i, improvement in enumerate(report.improvements[:3], 1):
            lines.append(f"  {i}. {improvement}")
        lines.append("")
    
    # Coaching plan (training mode)
    if report.coaching_plan:
        lines.append("Recommended Actions:")
        for i, action in enumerate(report.coaching_plan[:3], 1):
            lines.append(f"  {i}. {action}")
        lines.append("")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


def format_quick_stats_display(stats: QuickStats) -> Dict[str, Any]:
    """
    Format quick stats for immediate display after call ends.
    
    Args:
        stats: QuickStats object
        
    Returns:
        Dictionary formatted for UI display
        
    Example:
        >>> stats = QuickStats(...)
        >>> display_data = format_quick_stats_display(stats)
        >>> # Send to frontend for immediate feedback
    """
    logger.info("[FORMATTER] Formatting quick stats for display")
    
    return {
        'duration': stats.duration_formatted,
        'total_turns': stats.total_turns,
        'final_emotion': stats.final_customer_emotion,
        'status': 'ready_for_evaluation',
        'message': 'Call completed. Click "Get Evaluation" for detailed analysis.'
    }


def calculate_quick_stats(
    transcript: List[Dict[str, Any]],
    emotion_log: List[Dict[str, Any]],
    session_info: Dict[str, Any]
) -> QuickStats:
    """Calculate quick stats without LLM (for immediate feedback)."""
    logger.info("[FORMATTER] Calculating quick stats")
    
    # Calculate duration
    duration_seconds = session_info.get('duration_seconds', 0)
    duration_formatted = _format_duration(duration_seconds)
    
    # Count turns
    total_turns = len(transcript)
    salesperson_turns = len([t for t in transcript if t.get('speaker') == 'salesperson'])
    customer_turns = len([t for t in transcript if t.get('speaker') == 'customer'])
    
    # Get final emotion
    final_emotion = "neutral"
    if emotion_log and len(emotion_log) > 0:
        final_emotion = emotion_log[-1].get('emotion', 'neutral')
    
    return QuickStats(
        duration_seconds=duration_seconds,
        duration_formatted=duration_formatted,
        total_turns=total_turns,
        salesperson_turns=salesperson_turns,
        customer_turns=customer_turns,
        final_customer_emotion=final_emotion
    )


def format_skill_breakdown(
    report: FinalReport,
    include_descriptions: bool = True
) -> List[Dict[str, Any]]:
    """
    Format skill scores with metadata for display.
    
    Args:
        report: FinalReport from evaluation
        include_descriptions: Include skill descriptions
        
    Returns:
        List of skill data for UI rendering
        
    Example:
        >>> breakdown = format_skill_breakdown(report)
        >>> for skill in breakdown:
        ...     print(f"{skill['name_en']}: {skill['score']}/100")
    """
    logger.info("[FORMATTER] Formatting skill breakdown")
    
    breakdown = []
    
    # Get scores from report (this will be in the FinalReport.scores field)
    # For now, assume scores are in a ScoreBreakdown object
    if hasattr(report, 'scores') and hasattr(report.scores, 'skills'):
        skill_scores = {s.skill_key: s.score for s in report.scores.skills}
    else:
        # Fallback if structure is different
        skill_scores = {}
    
    for skill_key, config in SKILL_CONFIGS.items():
        skill_data = {
            'key': skill_key,
            'name_en': config.name_en,
            'name_ar': config.name_ar,
            'score': skill_scores.get(skill_key, 0),
            'weight': config.default_weight,
        }
        
        if include_descriptions:
            skill_data['description'] = config.description
        
        breakdown.append(skill_data)
    
    return breakdown


def format_checkpoint_summary(report: FinalReport) -> List[Dict[str, Any]]:
    """
    Format checkpoint achievements for display.
    
    Args:
        report: FinalReport from evaluation
        
    Returns:
        List of checkpoint data for UI rendering
        
    Example:
        >>> checkpoints = format_checkpoint_summary(report)
        >>> for cp in checkpoints:
        ...     icon = "✅" if cp['achieved'] else "❌"
        ...     print(f"{icon} {cp['name_en']}")
    """
    logger.info("[FORMATTER] Formatting checkpoint summary")
    
    summary = []
    
    # Get checkpoint achievements from report
    # Assume they're in report.checkpoints or similar
    checkpoint_achievements = {}
    if hasattr(report, 'checkpoints'):
        checkpoint_achievements = report.checkpoints
    
    for checkpoint_key, config in CHECKPOINT_CONFIGS.items():
        achieved = checkpoint_achievements.get(checkpoint_key, False)
        
        summary.append({
            'key': checkpoint_key,
            'name_en': config.name_en,
            'name_ar': config.name_ar,
            'description': config.description,
            'achieved': achieved,
            'icon': config.icon_achieved if achieved else config.icon_missed,
            'is_critical': config.is_critical,
            'order': config.order
        })
    
    # Sort by order
    summary.sort(key=lambda x: x['order'])
    
    return summary


def create_score_bar(score: float, width: int = 20) -> str:
    """
    Create a visual bar representation of a score.
    
    Args:
        score: Score value (0-100)
        width: Width of the bar in characters
        
    Returns:
        String representation of the score bar
        
    Example:
        >>> bar = create_score_bar(75)
        >>> print(f"Score: {bar}")
        Score: [███████████████░░░░░]
    """
    filled = int((score / 100) * width)
    empty = width - filled
    
    bar = "█" * filled + "░" * empty
    
    return f"[{bar}]"


def format_detailed_report(report: FinalReport) -> str:
    """
    Generate a comprehensive text report with all details.
    
    Args:
        report: FinalReport from evaluation
        
    Returns:
        Detailed formatted text report
    """
    logger.info("[FORMATTER] Generating detailed text report")
    
    lines = []
    
    # Header
    lines.append("=" * 80)
    lines.append("VCAI EVALUATION REPORT")
    lines.append("=" * 80)
    lines.append(f"Mode: {report.mode.upper()}")
    lines.append(f"Overall Score: {report.overall_score}/100")
    
    if report.mode == "testing":
        status = "PASSED ✓" if report.passed else "NEEDS IMPROVEMENT ✗"
        lines.append(f"Result: {status} (Threshold: {report.pass_threshold})")
    
    lines.append("")
    lines.append(f"Summary: {report.headline}")
    lines.append("")
    
    # Skill breakdown
    lines.append("SKILL SCORES:")
    lines.append("-" * 80)
    
    skill_breakdown = format_skill_breakdown(report, include_descriptions=False)
    for skill in skill_breakdown:
        bar = create_score_bar(skill['score'])
        lines.append(f"{skill['name_en']:30s} {skill['score']:5.1f}/100 {bar}")
    
    lines.append("")
    
    # Checkpoints
    lines.append("CHECKPOINTS:")
    lines.append("-" * 80)
    
    checkpoint_summary = format_checkpoint_summary(report)
    achieved_count = sum(1 for cp in checkpoint_summary if cp['achieved'])
    lines.append(f"Achieved: {achieved_count}/{len(checkpoint_summary)}")
    lines.append("")
    
    for cp in checkpoint_summary:
        lines.append(f"{cp['icon']} {cp['name_en']}")
        if cp['achieved']:
            lines.append(f"   {cp['description']}")
    
    lines.append("")
    
    # Strengths
    if report.strengths:
        lines.append("TOP STRENGTHS:")
        lines.append("-" * 80)
        for i, strength in enumerate(report.strengths, 1):
            lines.append(f"{i}. {strength}")
        lines.append("")
    
    # Improvements
    if report.improvements:
        lines.append("AREAS TO IMPROVE:")
        lines.append("-" * 80)
        for i, improvement in enumerate(report.improvements, 1):
            lines.append(f"{i}. {improvement}")
        lines.append("")
    
    # Coaching plan (training mode)
    if report.coaching_plan:
        lines.append("COACHING PLAN:")
        lines.append("-" * 80)
        for i, step in enumerate(report.coaching_plan, 1):
            lines.append(f"{i}. {step}")
        lines.append("")
    
    # Testing notes (testing mode)
    if report.testing_notes:
        lines.append("ASSESSMENT NOTES:")
        lines.append("-" * 80)
        for note in report.testing_notes:
            lines.append(f"• {note}")
        lines.append("")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)


def _format_duration(seconds: int) -> str:
    """
    Format duration in seconds to MM:SS format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "4:32")
    """
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes}:{secs:02d}"



# PATCH: Add this function to the END of evaluation/utils/report_formatter.py
# This is the LangGraph node that the graph imports

def compute_quick_stats_node(state: dict) -> dict:
    """
    LangGraph node that computes quick stats (no LLM needed).
    
    This is the first node in the evaluation pipeline.
    Reads: transcript, emotion_log, session_info
    Writes: quick_stats
    """
    from evaluation.state import update_state_status, record_node_timing, EvaluationProgress
    from evaluation.schemas import QuickStats
    import time
    
    t0 = time.perf_counter()
    
    logger.info("[QUICK_STATS_NODE] Computing quick stats...")
    
    try:
        state = update_state_status(state, "computing_quick_stats", EvaluationProgress.COMPUTING_QUICK_STATS)
        
        transcript = state.get("transcript", [])
        emotion_log = state.get("emotion_log", [])
        session_info = state.get("session_info") or {}
        
        # Calculate quick stats using existing function
        quick_stats = calculate_quick_stats(transcript, emotion_log, session_info)
        state["quick_stats"] = quick_stats
        
        state = update_state_status(state, "quick_stats_ready", EvaluationProgress.QUICK_STATS_READY)
        
        logger.info(f"[QUICK_STATS_NODE] Done: {quick_stats.total_turns} turns, {quick_stats.duration_formatted}")
        
    except Exception as e:
        logger.error(f"[QUICK_STATS_NODE] Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Create minimal stats on error so pipeline can continue
        state["quick_stats"] = QuickStats(
            duration_formatted="0:00",
            total_turns=len(state.get("transcript", [])),
            final_customer_emotion="neutral"
        )
    
    state = record_node_timing(state, "compute_quick_stats", time.perf_counter() - t0)
    return state