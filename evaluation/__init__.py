# evaluation/__init__.py
"""
VCAI Evaluation System
Main Module

This module provides comprehensive evaluation of sales conversations
using AI-powered analysis with dynamic weighting.

Main Entry Point:
- EvaluationManager: Orchestrates the evaluation process

Architecture:
- manager.py: Main orchestrator (Menna)
- graphs/: LangGraph pipeline (Ismail)
- schemas/: Data models (Ali)
- utils/: Helper functions (Menna)
- config.py: Configuration (Ali)
"""

# Main exports for external use
from .manager import EvaluationManager, create_manager, quick_evaluate

# Schema exports (from Ali's work)
from .schemas import (
    FinalReport,
    QuickStats,
    AnalysisReport,
    ConversationProfile,
    ScoreBreakdown
)

# State exports (from Ali's work)
from .state import EvaluationState, create_initial_state

# Config exports (from Ali's work)
from .config import (
    SKILL_CONFIGS,
    CHECKPOINT_CONFIGS,
    settings
)

# Utility exports (from Menna's work)
from .utils import (
    calculate_dynamic_weights,
    format_for_database,
    format_executive_summary,
    calculate_quick_stats
)


__version__ = "1.0.0"
__author__ = "VCAI Team: Ali, Ismail, Menna, Bakr"


__all__ = [
    # Main class
    'EvaluationManager',
    'create_manager',
    'quick_evaluate',
    
    # Schemas
    'FinalReport',
    'QuickStats',
    'AnalysisReport',
    'ConversationProfile',
    'ScoreBreakdown',
    
    # State
    'EvaluationState',
    'create_initial_state',
    
    # Config
    'SKILL_CONFIGS',
    'CHECKPOINT_CONFIGS',
    'settings',
    
    # Utils
    'calculate_dynamic_weights',
    'format_for_database',
    'format_executive_summary',
    'calculate_quick_stats',
]


# Convenience function for quick info
def info():
    """Print information about the evaluation module"""
    print("=" * 60)
    print("VCAI Evaluation System")
    print("=" * 60)
    print(f"Version: {__version__}")
    print(f"Team: {__author__}")
    print()
    print("Components:")
    print("  - EvaluationManager (Menna): Main orchestrator")
    print("  - LangGraph Pipeline (Ismail): AI evaluation nodes")
    print("  - Schemas (Ali): Data models")
    print("  - Utils (Menna): Weight calc & formatting")
    print("  - Config (Ali): Settings & skill definitions")
    print()
    print("Skills Evaluated:")
    for i, (key, config) in enumerate(SKILL_CONFIGS.items(), 1):
        print(f"  {i}. {config.name_en} ({config.name_ar})")
    print()
    print("Checkpoints:")
    for i, (key, config) in enumerate(CHECKPOINT_CONFIGS.items(), 1):
        print(f"  {i}. {config.name_en} ({config.name_ar})")
    print("=" * 60)


def get_version() -> str:
    """Return the version of the evaluation module"""
    return __version__