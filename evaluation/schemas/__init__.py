# evaluation/schemas/__init__.py
"""
Evaluation Schemas - Pydantic models for the evaluation system.

Usage:
    from evaluation.schemas import (
        # Analysis (Pass 1)
        ConversationProfile,
        SkillAssessment,
        AnalysisReport,
        
        # Report (Pass 2)
        FinalReport,
        ScoreBreakdown,
        QuickStats,
        
        # API
        EvaluationStatusResponse,
        EvaluationTriggerResponse,
    )
"""

# Analysis schemas (Pass 1 output)
from .analysis_schema import (
    # Enums
    ConversationStage,
    ObjectionType,
    ClosingSignalType,
    SkillCategory,
    RiskLevel,
    
    # Sub-models
    ObjectionInstance,
    ClosingSignal,
    EmotionTransition,
    FactClaim,
    CheckpointAchievement,
    
    # Main models
    ConversationProfile,
    SkillAssessment,
    FactVerification,
    AnalysisReport,
)

# Report schemas (Pass 2 output)
from .report_schema import (
    # Enums
    EvaluationMode,
    PassFailStatus,
    FeedbackTone,
    
    # Sub-models
    TurnFeedback,
    SkillScoreDetail,
    ScoreBreakdown,
    CheckpointSummary,
    QuickStats,
    
    # Main models
    FinalReport,
    
    # API responses
    EvaluationStatusResponse,
    EvaluationTriggerResponse,
    QuickStatsResponse,
)

__all__ = [
    # Analysis enums
    "ConversationStage",
    "ObjectionType", 
    "ClosingSignalType",
    "SkillCategory",
    "RiskLevel",
    
    # Analysis sub-models
    "ObjectionInstance",
    "ClosingSignal",
    "EmotionTransition",
    "FactClaim",
    "CheckpointAchievement",
    
    # Analysis main
    "ConversationProfile",
    "SkillAssessment",
    "FactVerification",
    "AnalysisReport",
    
    # Report enums
    "EvaluationMode",
    "PassFailStatus",
    "FeedbackTone",
    
    # Report sub-models
    "TurnFeedback",
    "SkillScoreDetail",
    "ScoreBreakdown",
    "CheckpointSummary",
    "QuickStats",
    
    # Report main
    "FinalReport",
    
    # API
    "EvaluationStatusResponse",
    "EvaluationTriggerResponse",
    "QuickStatsResponse",
]