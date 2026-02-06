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
    checkpoint_interval: int = CHECKPOINT_INTERVAL      # Create checkpoint every N turns (default: 5)
    recent_messages_count: int = RECENT_MESSAGES_COUNT  # Recent messages to keep (default: 10)
    
    # Session limits
    max_duration_seconds: int = SESSION_MAX_DURATION    # Max session duration
    max_turns: int = SESSION_MAX_TURNS                  # Max turns per session
    
    # RAG settings
    rag_top_k: int = RAG_TOP_K                          # Documents to retrieve
    
    # Processing settings
    enable_emotion: bool = True                         # Enable emotion detection
    enable_rag: bool = True                             # Enable RAG retrieval
    enable_checkpoints: bool = True                     # Enable memory checkpoints
    enable_streaming: bool = True                       # Enable streaming LLM+TTS (sentence-by-sentence)
    
    # Logging
    verbose: bool = True                                # Print debug logs
    log_timings: bool = True                            # Log node timings
    
    # ══════════════════════════════════════════════════════════════════════════
    # IMPORTANT: Set to False to use real agents
    # ══════════════════════════════════════════════════════════════════════════
    use_mocks: bool = False                             # Use real functions (NOT mocks)


# Default configuration - uses REAL agents
DEFAULT_CONFIG = OrchestrationConfig()


def get_config(
    use_mocks: bool = False,    # Default: use real agents
    verbose: bool = True,
    enable_streaming: bool = True,  # Default: streaming enabled
    **kwargs
) -> OrchestrationConfig:
    """
    Get orchestration configuration with optional overrides.
    
    Args:
        use_mocks: Whether to use mock functions (default: False = real agents)
        verbose: Whether to print debug logs
        enable_streaming: Whether to use streaming LLM+TTS (default: True)
        **kwargs: Additional config overrides
    
    Returns:
        OrchestrationConfig: Configuration object
    """
    return OrchestrationConfig(
        use_mocks=use_mocks,
        verbose=verbose,
        enable_streaming=enable_streaming,
        **kwargs
    )


# Convenience configs for different scenarios
def get_development_config() -> OrchestrationConfig:
    """Config for development with mocks."""
    return OrchestrationConfig(
        use_mocks=True,
        verbose=True,
        enable_checkpoints=False,  # Faster without checkpoints
        enable_streaming=True       # Can test streaming with mocks
    )


def get_production_config() -> OrchestrationConfig:
    """Config for production with real agents."""
    return OrchestrationConfig(
        use_mocks=False,
        verbose=False,
        enable_checkpoints=True,
        enable_streaming=True,      # Streaming enabled in production
        log_timings=True
    )


def get_testing_config() -> OrchestrationConfig:
    """Config for testing."""
    return OrchestrationConfig(
        use_mocks=True,
        verbose=True,
        enable_checkpoints=True,
        enable_streaming=False,     # Disable streaming for faster tests
        checkpoint_interval=2       # More frequent checkpoints for testing
    )