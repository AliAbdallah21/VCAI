# evaluation/schemas/report_schema.py
"""
Report Schemas - Pydantic models for Pass 2 (Synthesizer) output.

These schemas define:
- TurnFeedback: Feedback for a specific turn
- ScoreBreakdown: Detailed score breakdown by category
- FinalReport: Complete evaluation report for frontend display
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════════

class EvaluationMode(str, Enum):
    """Evaluation mode affects report tone and format."""
    TRAINING = "training"  # Encouraging, growth-focused
    TESTING = "testing"    # Objective, pass/fail


class PassFailStatus(str, Enum):
    """Pass/fail status for testing mode."""
    PASSED = "passed"
    FAILED = "failed"
    BORDERLINE = "borderline"  # Close to threshold


class FeedbackTone(str, Enum):
    """Tone of feedback for a specific item."""
    POSITIVE = "positive"
    CONSTRUCTIVE = "constructive"
    CRITICAL = "critical"
    NEUTRAL = "neutral"


# ═══════════════════════════════════════════════════════════════════════════════
# Turn-Level Feedback
# ═══════════════════════════════════════════════════════════════════════════════

class TurnFeedback(BaseModel):
    """Feedback for a specific conversation turn."""
    
    turn_number: int = Field(..., description="Which turn (1-indexed)")
    speaker: Literal["salesperson", "customer"] = Field(..., description="Who spoke")
    
    # The actual content
    text: str = Field(..., description="What was said")
    
    # Assessment
    assessment: str = Field(
        default="neutral",
        description="Assessment: 'excellent', 'good', 'adequate', 'needs_improvement', 'poor'"
    )
    tone: FeedbackTone = Field(
        default=FeedbackTone.NEUTRAL,
        description="Tone of feedback"
    )
    
    # Feedback content
    what_was_good: Optional[str] = Field(
        None,
        description="What was done well (if anything)"
    )
    what_to_improve: Optional[str] = Field(
        None,
        description="What could be better (if anything)"
    )
    suggested_alternative: Optional[str] = Field(
        None,
        description="Example of better response (training mode only)"
    )
    
    # Skills demonstrated
    skills_shown: list[str] = Field(
        default_factory=list,
        description="Skills demonstrated in this turn"
    )
    
    # Flags
    is_highlight: bool = Field(
        default=False,
        description="Should this turn be highlighted as notable"
    )
    is_critical_moment: bool = Field(
        default=False,
        description="Was this a critical moment (closing signal, objection, etc.)"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Score Breakdown
# ═══════════════════════════════════════════════════════════════════════════════

class SkillScoreDetail(BaseModel):
    """Detailed score for one skill category."""
    
    skill_name: str = Field(..., description="Skill name (display-friendly)")
    skill_key: str = Field(..., description="Skill key (for programmatic use)")
    
    score: int = Field(..., ge=0, le=100, description="Score 0-100")
    weight: float = Field(..., ge=0.0, le=1.0, description="Weight in final score")
    weighted_contribution: float = Field(
        ...,
        description="score * weight (contribution to overall)"
    )
    
    was_tested: bool = Field(
        default=True,
        description="Whether this skill was actually tested"
    )
    
    # Feedback
    summary: str = Field(
        default="",
        description="One-line summary of performance in this skill"
    )
    strengths: list[str] = Field(
        default_factory=list,
        description="Specific strengths"
    )
    areas_to_improve: list[str] = Field(
        default_factory=list,
        description="Specific areas to work on"
    )
    
    # Evidence
    evidence_turns: list[int] = Field(
        default_factory=list,
        description="Turn numbers where this skill was demonstrated"
    )


class ScoreBreakdown(BaseModel):
    """Complete score breakdown for the evaluation."""
    
    # Overall
    overall_score: int = Field(..., ge=0, le=100, description="Final weighted score")
    pass_threshold: int = Field(default=75, description="Score needed to pass")
    
    # Pass/fail (testing mode)
    status: PassFailStatus = Field(
        default=PassFailStatus.PASSED,
        description="Pass/fail status"
    )
    points_from_passing: int = Field(
        default=0,
        description="How many points above/below threshold"
    )
    
    # Skill breakdown
    skills: list[SkillScoreDetail] = Field(
        default_factory=list,
        description="Detailed breakdown by skill"
    )
    
    # Summary stats
    skills_tested: int = Field(default=0, description="Number of skills tested")
    skills_passed: int = Field(default=0, description="Skills scoring above threshold")
    strongest_skill: Optional[str] = Field(None, description="Highest scoring skill")
    weakest_skill: Optional[str] = Field(None, description="Lowest scoring skill")


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint Summary
# ═══════════════════════════════════════════════════════════════════════════════

class CheckpointSummary(BaseModel):
    """Summary of checkpoints achieved."""
    
    name: str = Field(..., description="Checkpoint name")
    achieved: bool = Field(..., description="Whether achieved")
    
    # Display
    icon: str = Field(default="⬜", description="Emoji icon: ✅ or ❌")
    description: str = Field(default="", description="What this checkpoint means")
    
    # Details (if achieved)
    achieved_at_turn: Optional[int] = Field(None, description="Turn where achieved")
    how_achieved: Optional[str] = Field(None, description="Brief explanation")
    
    # Details (if not achieved)
    why_missed: Optional[str] = Field(
        None,
        description="Why it wasn't achieved (training mode only)"
    )
    how_to_achieve: Optional[str] = Field(
        None,
        description="Tips for next time (training mode only)"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Quick Stats (Immediate display after call)
# ═══════════════════════════════════════════════════════════════════════════════

class QuickStats(BaseModel):
    """
    Quick statistics shown immediately after call ends.
    These don't require LLM evaluation - just data aggregation.
    """
    
    # Time
    duration_seconds: int = Field(..., description="Call duration")
    duration_formatted: str = Field(..., description="Formatted: '4:32'")
    
    # Turns
    total_turns: int = Field(..., description="Total conversation turns")
    salesperson_turns: int = Field(..., description="Salesperson message count")
    customer_turns: int = Field(..., description="Customer message count")
    
    # Emotions
    final_customer_emotion: str = Field(..., description="Last detected emotion")
    emotion_improved: bool = Field(
        default=False,
        description="Did emotion improve from start to end"
    )
    
    # Checkpoints (quick version)
    checkpoints_achieved: int = Field(default=0, description="Number achieved")
    checkpoints_total: int = Field(default=6, description="Total possible")
    checkpoint_list: list[str] = Field(
        default_factory=list,
        description="List of checkpoint names with ✓/✗"
    )
    
    # Outcome
    call_outcome: str = Field(
        default="completed",
        description="Outcome: 'closed', 'follow_up', 'interested', 'disengaged', 'completed'"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Final Report
# ═══════════════════════════════════════════════════════════════════════════════

class FinalReport(BaseModel):
    """
    Complete evaluation report - output of the full evaluation pipeline.
    This is what gets saved to database and displayed to user.
    """
    
    # ─── Metadata ─────────────────────────────────────────────────────────────
    
    report_id: str = Field(..., description="Unique report ID")
    session_id: str = Field(..., description="Session that was evaluated")
    user_id: str = Field(..., description="User who did the training")
    persona_id: str = Field(..., description="Customer persona used")
    persona_name: str = Field(default="", description="Persona display name")
    
    mode: EvaluationMode = Field(..., description="Training or testing mode")
    
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When report was generated"
    )
    
    # ─── Quick Stats (always available immediately) ───────────────────────────
    
    quick_stats: QuickStats = Field(..., description="Immediate stats after call")
    
    # ─── Scores ───────────────────────────────────────────────────────────────
    
    scores: ScoreBreakdown = Field(..., description="Full score breakdown")
    
    # ─── Checkpoints ──────────────────────────────────────────────────────────
    
    checkpoints: list[CheckpointSummary] = Field(
        default_factory=list,
        description="Checkpoint achievements"
    )
    
    # ─── Executive Summary ────────────────────────────────────────────────────
    
    executive_summary: str = Field(
        default="",
        description="2-3 sentence overall summary"
    )
    
    # ─── Key Takeaways ────────────────────────────────────────────────────────
    
    top_strengths: list[str] = Field(
        default_factory=list,
        description="Top 3 things done well"
    )
    top_improvements: list[str] = Field(
        default_factory=list,
        description="Top 3 things to work on"
    )
    
    # ─── Detailed Feedback ────────────────────────────────────────────────────
    
    # Turn-by-turn (optional - can be large)
    turn_feedback: list[TurnFeedback] = Field(
        default_factory=list,
        description="Feedback for each turn (training mode)"
    )
    include_turn_feedback: bool = Field(
        default=True,
        description="Whether turn_feedback is populated"
    )
    
    # Highlighted moments
    highlight_turns: list[int] = Field(
        default_factory=list,
        description="Turn numbers worth highlighting"
    )
    
    # ─── Actionable Next Steps ────────────────────────────────────────────────
    
    recommended_practice: list[str] = Field(
        default_factory=list,
        description="Specific scenarios to practice (training mode)"
    )
    
    # ─── Testing Mode Specific ────────────────────────────────────────────────
    
    # Only populated if mode == TESTING
    passed: Optional[bool] = Field(
        None,
        description="Whether passed the assessment (testing mode only)"
    )
    certification_eligible: Optional[bool] = Field(
        None,
        description="Whether eligible for certification (testing mode only)"
    )
    retake_recommended: Optional[bool] = Field(
        None,
        description="Whether retake is recommended (testing mode only)"
    )
    
    # ─── Raw Data (for debugging/auditing) ────────────────────────────────────
    
    # Store raw analysis for debugging (optional)
    include_raw_analysis: bool = Field(
        default=False,
        description="Whether raw_analysis is populated"
    )
    raw_analysis: Optional[dict] = Field(
        None,
        description="Raw AnalysisReport as dict (for debugging)"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# API Response Models
# ═══════════════════════════════════════════════════════════════════════════════

class EvaluationStatusResponse(BaseModel):
    """Response for evaluation status check."""
    
    session_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    progress: Optional[int] = Field(None, description="Progress 0-100 if processing")
    report_id: Optional[str] = Field(None, description="Report ID if completed")
    error: Optional[str] = Field(None, description="Error message if failed")


class EvaluationTriggerResponse(BaseModel):
    """Response when evaluation is triggered."""
    
    session_id: str
    status: Literal["started", "already_exists", "error"]
    message: str
    report_id: Optional[str] = Field(None, description="Existing report ID if already evaluated")


class QuickStatsResponse(BaseModel):
    """Response for immediate stats after call ends."""
    
    session_id: str
    stats: QuickStats
    evaluation_available: bool = Field(
        default=True,
        description="Whether full evaluation can be requested"
    )