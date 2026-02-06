# evaluation/schemas/analysis_schema.py
"""
Analysis Schemas - Pydantic models for Pass 1 (Analyzer) output.

These schemas define:
- ConversationProfile: What happened in the conversation
- SkillAssessment: How well each skill was demonstrated
- AnalysisReport: Complete analysis output from analyzer
"""

from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════════

class ConversationStage(str, Enum):
    """Stages a sales conversation can go through."""
    OPENING = "opening"
    RAPPORT_BUILDING = "rapport_building"
    NEEDS_DISCOVERY = "needs_discovery"
    PRESENTATION = "presentation"
    OBJECTION_HANDLING = "objection_handling"
    NEGOTIATION = "negotiation"
    CLOSING = "closing"
    FOLLOW_UP = "follow_up"


class ObjectionType(str, Enum):
    """Types of customer objections."""
    PRICE = "price"
    LOCATION = "location"
    TIMING = "timing"
    TRUST = "trust"
    COMPETITION = "competition"
    FEATURES = "features"
    PAYMENT = "payment"
    OTHER = "other"


class ClosingSignalType(str, Enum):
    """Types of closing signals customers give."""
    ASKING_NEXT_STEPS = "asking_next_steps"
    ASKING_AVAILABILITY = "asking_availability"
    ASKING_PAYMENT_DETAILS = "asking_payment_details"
    POSITIVE_CONFIRMATION = "positive_confirmation"
    SCHEDULING_REQUEST = "scheduling_request"
    PAPERWORK_QUESTION = "paperwork_question"


class SkillCategory(str, Enum):
    """Sales skills that can be evaluated."""
    RAPPORT_BUILDING = "rapport_building"
    ACTIVE_LISTENING = "active_listening"
    NEEDS_DISCOVERY = "needs_discovery"
    PRODUCT_KNOWLEDGE = "product_knowledge"
    OBJECTION_HANDLING = "objection_handling"
    EMOTIONAL_INTELLIGENCE = "emotional_intelligence"
    CLOSING_SKILLS = "closing_skills"
    COMMUNICATION_CLARITY = "communication_clarity"


class RiskLevel(str, Enum):
    """Risk level of losing the customer."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ═══════════════════════════════════════════════════════════════════════════════
# Sub-Models
# ═══════════════════════════════════════════════════════════════════════════════

class ObjectionInstance(BaseModel):
    """A single objection raised during the conversation."""
    
    type: ObjectionType = Field(..., description="Type of objection")
    turn_number: int = Field(..., description="Turn where objection was raised")
    customer_text: str = Field(..., description="What the customer said")
    was_handled: bool = Field(..., description="Whether salesperson addressed it")
    handling_quality: Optional[str] = Field(
        None, 
        description="Quality of handling: 'excellent', 'good', 'partial', 'poor', 'not_handled'"
    )
    salesperson_response: Optional[str] = Field(
        None, 
        description="How the salesperson responded"
    )


class ClosingSignal(BaseModel):
    """A closing signal given by the customer."""
    
    type: ClosingSignalType = Field(..., description="Type of closing signal")
    turn_number: int = Field(..., description="Turn where signal was given")
    customer_text: str = Field(..., description="What the customer said")
    was_recognized: bool = Field(..., description="Did salesperson recognize it")
    was_acted_upon: bool = Field(..., description="Did salesperson act on it")
    salesperson_response: Optional[str] = Field(
        None,
        description="How salesperson responded to the signal"
    )


class EmotionTransition(BaseModel):
    """A significant emotion change during the conversation."""
    
    from_emotion: str = Field(..., description="Previous emotion state")
    to_emotion: str = Field(..., description="New emotion state")
    turn_number: int = Field(..., description="Turn where transition happened")
    trigger: Optional[str] = Field(
        None,
        description="What likely caused the transition"
    )


class FactClaim(BaseModel):
    """A factual claim made by the salesperson."""
    
    turn_number: int = Field(..., description="Turn where claim was made")
    claim_text: str = Field(..., description="The claim that was made")
    topic: str = Field(..., description="Topic: price, location, features, etc.")
    is_accurate: Optional[bool] = Field(
        None,
        description="Whether claim matches RAG context (None if not verifiable)"
    )
    rag_reference: Optional[str] = Field(
        None,
        description="Relevant RAG content for verification"
    )


class CheckpointAchievement(BaseModel):
    """A milestone achieved during the conversation."""
    
    checkpoint: str = Field(..., description="Name of checkpoint")
    achieved: bool = Field(..., description="Whether it was achieved")
    turn_number: Optional[int] = Field(
        None,
        description="Turn where achieved (None if not achieved)"
    )
    evidence: Optional[str] = Field(
        None,
        description="Evidence of achievement"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Conversation Profile
# ═══════════════════════════════════════════════════════════════════════════════

class ConversationProfile(BaseModel):
    """
    Complete profile of what happened in the conversation.
    This determines WHAT to evaluate (dynamic weights).
    """
    
    # Basic info
    total_turns: int = Field(..., description="Total conversation turns")
    duration_seconds: int = Field(..., description="Call duration in seconds")
    
    # Stages
    stages_present: list[ConversationStage] = Field(
        default_factory=list,
        description="Which conversation stages occurred"
    )
    stages_completed: list[ConversationStage] = Field(
        default_factory=list,
        description="Which stages were successfully completed"
    )
    
    # Topics discussed
    topics_discussed: list[str] = Field(
        default_factory=list,
        description="Topics covered: price, location, features, payment, etc."
    )
    
    # Objections
    objections: list[ObjectionInstance] = Field(
        default_factory=list,
        description="All objections raised and how they were handled"
    )
    
    # Closing signals
    closing_signals: list[ClosingSignal] = Field(
        default_factory=list,
        description="Closing signals given by customer"
    )
    
    # Emotions
    emotion_journey: list[str] = Field(
        default_factory=list,
        description="Sequence of customer emotions throughout conversation"
    )
    emotion_transitions: list[EmotionTransition] = Field(
        default_factory=list,
        description="Significant emotion changes"
    )
    dominant_emotion: str = Field(
        default="neutral",
        description="Most frequent/impactful emotion"
    )
    final_emotion: str = Field(
        default="neutral",
        description="Customer's emotion at end of call"
    )
    
    # Risk tracking
    peak_risk_level: RiskLevel = Field(
        default=RiskLevel.LOW,
        description="Highest risk level reached during conversation"
    )
    risk_moments: list[int] = Field(
        default_factory=list,
        description="Turn numbers where risk was elevated"
    )
    
    # Checkpoints
    checkpoints: list[CheckpointAchievement] = Field(
        default_factory=list,
        description="Milestone achievements"
    )
    
    # RAG relevance
    rag_was_needed: bool = Field(
        default=False,
        description="Whether factual knowledge was needed"
    )
    rag_topics_relevant: list[str] = Field(
        default_factory=list,
        description="Which RAG topics were relevant to this conversation"
    )
    
    # Outcome
    call_outcome: str = Field(
        default="unknown",
        description="Outcome: 'closed', 'follow_up_scheduled', 'customer_interested', 'customer_disengaged', 'unknown'"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Skill Assessment
# ═══════════════════════════════════════════════════════════════════════════════

class SkillAssessment(BaseModel):
    """Assessment of a single skill category."""
    
    skill: SkillCategory = Field(..., description="Which skill")
    was_tested: bool = Field(..., description="Was this skill actually tested in conversation")
    
    # Only filled if was_tested=True
    score: Optional[int] = Field(
        None,
        ge=0, le=100,
        description="Score 0-100 (None if not tested)"
    )
    weight: float = Field(
        default=0.0,
        ge=0.0, le=1.0,
        description="Dynamic weight based on relevance (0 if not tested)"
    )
    
    instances: list[str] = Field(
        default_factory=list,
        description="Specific examples/evidence from transcript"
    )
    
    strengths: list[str] = Field(
        default_factory=list,
        description="What was done well"
    )
    improvements: list[str] = Field(
        default_factory=list,
        description="What could be improved"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Fact Verification
# ═══════════════════════════════════════════════════════════════════════════════

class FactVerification(BaseModel):
    """Verification of factual claims against RAG context."""
    
    total_claims: int = Field(default=0, description="Total factual claims made")
    verified_accurate: int = Field(default=0, description="Claims verified as accurate")
    verified_inaccurate: int = Field(default=0, description="Claims verified as wrong")
    unverifiable: int = Field(default=0, description="Claims that couldn't be verified")
    
    accuracy_rate: Optional[float] = Field(
        None,
        ge=0.0, le=1.0,
        description="Accuracy rate (verified_accurate / (accurate + inaccurate))"
    )
    
    claims: list[FactClaim] = Field(
        default_factory=list,
        description="All claims with verification status"
    )
    
    critical_errors: list[str] = Field(
        default_factory=list,
        description="Serious factual mistakes that could lose the sale"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis Report (Pass 1 Output)
# ═══════════════════════════════════════════════════════════════════════════════

class AnalysisReport(BaseModel):
    """
    Complete output from the Analyzer (Pass 1).
    This is input to the Synthesizer (Pass 2).
    """
    
    # Session info
    session_id: str = Field(..., description="Session being evaluated")
    
    # Conversation understanding
    profile: ConversationProfile = Field(
        ...,
        description="What happened in the conversation"
    )
    
    # Skill assessments
    skills: list[SkillAssessment] = Field(
        default_factory=list,
        description="Assessment of each relevant skill"
    )
    
    # Fact checking
    fact_verification: FactVerification = Field(
        default_factory=FactVerification,
        description="Verification of factual claims"
    )
    
    # Dynamic weights (sum to 1.0)
    skill_weights: dict[str, float] = Field(
        default_factory=dict,
        description="Dynamic weights for each skill based on conversation profile"
    )
    
    # Raw scores (before weighting)
    raw_scores: dict[str, int] = Field(
        default_factory=dict,
        description="Raw score for each tested skill"
    )
    
    # Calculated overall (weighted average)
    weighted_overall_score: int = Field(
        default=0,
        ge=0, le=100,
        description="Weighted average of all tested skills"
    )
    
    # Key moments
    best_moments: list[str] = Field(
        default_factory=list,
        description="Turn references for best moments (e.g., 'Turn 5: Excellent objection handling')"
    )
    worst_moments: list[str] = Field(
        default_factory=list,
        description="Turn references for moments needing improvement"
    )
    missed_opportunities: list[str] = Field(
        default_factory=list,
        description="Opportunities that were missed"
    )
    
    # Metadata
    analysis_timestamp: str = Field(
        default="",
        description="When analysis was performed (ISO format)"
    )