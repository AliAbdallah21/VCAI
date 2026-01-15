# orchestration/config.py
"""
Configuration settings for orchestration agent.
"""

from dataclasses import dataclass
from shared.constants import (
    CHECKPOINT_INTERVAL,
    RECENT_MESSAGES_COUNT,
    SESSION_MAX_DURATION,
    SESSION_MAX_TURNS,
    RAG_TOP_K
)


@dataclass
class OrchestrationConfig:
    """Configuration for the orchestration agent."""
    
    # Memory settings
    checkpoint_interval: int = CHECKPOINT_INTERVAL      # Create checkpoint every N turns
    recent_messages_count: int = RECENT_MESSAGES_COUNT  # Recent messages to keep
    
    # Session limits
    max_duration_seconds: int = SESSION_MAX_DURATION    # Max session duration
    max_turns: int = SESSION_MAX_TURNS                  # Max turns per session
    
    # RAG settings
    rag_top_k: int = RAG_TOP_K                          # Documents to retrieve
    
    # Processing settings
    enable_emotion: bool = True                         # Enable emotion detection
    enable_rag: bool = True                             # Enable RAG retrieval
    enable_checkpoints: bool = True                     # Enable memory checkpoints
    
    # Logging
    verbose: bool = True                                # Print debug logs
    log_timings: bool = True                            # Log node timings
    
    # Mock vs Real
    use_mocks: bool = True                              # Use mock functions (for development)


# Default configuration
DEFAULT_CONFIG = OrchestrationConfig()


def get_config(
    use_mocks: bool = True,
    verbose: bool = True,
    **kwargs
) -> OrchestrationConfig:
    """
    Get orchestration configuration with optional overrides.
    
    Args:
        use_mocks: Whether to use mock functions
        verbose: Whether to print debug logs
        **kwargs: Additional config overrides
    
    Returns:
        OrchestrationConfig: Configuration object
    """
    return OrchestrationConfig(
        use_mocks=use_mocks,
        verbose=verbose,
        **kwargs
    )