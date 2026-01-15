# backend/models/persona.py
"""
Persona SQLAlchemy model.
"""

from sqlalchemy import Column, String, Text, Integer, Boolean, DateTime, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from backend.database import Base


class Persona(Base):
    """Persona model - represents a virtual customer personality."""
    
    __tablename__ = "personas"
    
    id = Column(String(100), primary_key=True)
    name_ar = Column(String(255), nullable=False)
    name_en = Column(String(255), nullable=False)
    description_ar = Column(Text)
    description_en = Column(Text)
    personality_prompt = Column(Text, nullable=False)
    difficulty = Column(String(50), nullable=False)  # easy, medium, hard
    patience_level = Column(Integer, default=50)  # 0-100
    emotion_sensitivity = Column(Integer, default=50)  # 0-100
    traits = Column(JSONB, default=[])
    voice_id = Column(String(100))
    avatar_url = Column(String(500))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    sessions = relationship("Session", back_populates="persona")
    
    def __repr__(self):
        return f"<Persona {self.name_en} ({self.difficulty})>"
    
    def to_dict(self):
        """Convert to dictionary for API response."""
        return {
            "id": self.id,
            "name_ar": self.name_ar,
            "name_en": self.name_en,
            "description_ar": self.description_ar,
            "description_en": self.description_en,
            "difficulty": self.difficulty,
            "patience_level": self.patience_level,
            "emotion_sensitivity": self.emotion_sensitivity,
            "traits": self.traits,
            "voice_id": self.voice_id,
            "avatar_url": self.avatar_url
        }