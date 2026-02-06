# backend/models/evaluation.py
"""
Evaluation Report SQLAlchemy model.
"""

from sqlalchemy import Column, String, Text, Integer, Boolean, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
import uuid

from backend.database import Base


class EvaluationReport(Base):
    """Evaluation report model - stores final evaluation results."""
    
    __tablename__ = "evaluation_reports"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, unique=True)
    
    # Report identification
    report_id = Column(String(100), unique=True, nullable=False)
    
    # Status tracking
    status = Column(String(50), default="pending")  # pending, processing, completed, failed
    progress = Column(Integer, default=0)  # 0-100
    mode = Column(String(50), default="training")  # training, testing
    
    # Scores (denormalized for query performance)
    overall_score = Column(Integer)
    pass_threshold = Column(Integer, default=75)
    passed = Column(Boolean)
    
    # Full report data
    report_json = Column(JSONB)  # Complete FinalReport as JSON
    
    # Quick stats (computed immediately after session ends)
    quick_stats_json = Column(JSONB)  # QuickStats as JSON
    
    # Error handling
    error_message = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    
    # Relationships
    session = relationship("Session", backref="evaluation_report")
    
    def __repr__(self):
        return f"<EvaluationReport {self.report_id} ({self.status})>"
