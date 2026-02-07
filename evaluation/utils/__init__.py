# evaluation/utils/__init__.py
"""
VCAI Evaluation Utilities
Author: Menna Khaled

Utility functions for the evaluation system:
- Dynamic weight calculation
- Report formatting
- Quick stats computation
"""

from .weight_calculator import (
    calculate_dynamic_weights,
    get_weight_explanation,
    validate_weights
)

from .report_formatter import (
    format_for_database,
    format_executive_summary,
    format_quick_stats_display,
    calculate_quick_stats,
    format_skill_breakdown,
    format_checkpoint_summary,
    format_detailed_report,
    create_score_bar
)


__all__ = [
    # Weight calculation
    'calculate_dynamic_weights',
    'get_weight_explanation',
    'validate_weights',
    
    # Report formatting
    'format_for_database',
    'format_executive_summary',
    'format_quick_stats_display',
    'calculate_quick_stats',
    'format_skill_breakdown',
    'format_checkpoint_summary',
    'format_detailed_report',
    'create_score_bar',
]