# orchestration/agent.py
"""
Main Orchestration Agent for VCAI.
Entry point for conversation processing.
"""

import time
from datetime import datetime
from typing import Optional
import numpy as np

from orchestration.state import (
    ConversationState,
    create_initial_state,
    reset_turn_state,
    get_state_summary
)
from orchestration.config import OrchestrationConfig, get_config
from orchestration.graphs import create_conversation_graph, create_simple_graph
from shared.types import Persona, Scenario


class OrchestrationAgent:
    """
    Main orchestration agent that manages conversation flow.
    
    Usage:
        # Initialize
        agent = OrchestrationAgent(use_mocks=True)
        
        # Start session
        agent.start_session(
            session_id="session_001",
            user_id="user_001",
            persona_id="difficult_customer"
        )
        
        # Process turn
        result = agent.process_turn(audio_input)
        
        # Get response audio
        audio_output = result["audio_output"]
        
        # End session
        agent.end_session()
    """
    
    def __init__(
        self,
        use_mocks: bool = True,
        verbose: bool = True,
        simple_mode: bool = False,
        **config_kwargs
    ):
        """
        Initialize the orchestration agent.
        
        Args:
            use_mocks: Use mock functions (True for development)
            verbose: Print debug logs
            simple_mode: Use simplified graph (no memory/RAG)
            **config_kwargs: Additional config options
        """
        self.config = get_config(
            use_mocks=use_mocks,
            verbose=verbose,
            **config_kwargs
        )
        
        self.simple_mode = simple_mode
        self.graph = None
        self.state: Optional[ConversationState] = None
        self.session_active = False
        
        if verbose:
            print(f"[AGENT] Initialized (mocks={use_mocks}, simple={simple_mode})")
    
    def start_session(
        self,
        session_id: str,
        user_id: str,
        persona_id: str = None,
        scenario_id: str = None
    ) -> ConversationState:
        """
        Start a new conversation session.
        
        Args:
            session_id: Unique session identifier
            user_id: User/trainee identifier
            persona_id: Optional persona to load
            scenario_id: Optional scenario to load
        
        Returns:
            ConversationState: Initial state
        """
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"[AGENT] Starting session: {session_id}")
            print(f"{'='*60}")
        
        # Load persona if specified
        persona = None
        if persona_id:
            persona = self._load_persona(persona_id)
        
        # Load scenario if specified
        scenario = None
        if scenario_id:
            scenario = self._load_scenario(scenario_id)
        
        # Create initial state
        self.state = create_initial_state(
            session_id=session_id,
            user_id=user_id,
            persona=persona,
            scenario=scenario
        )
        
        # Create graph
        if self.simple_mode:
            self.graph = create_simple_graph(self.config)
        else:
            self.graph = create_conversation_graph(self.config)
        
        self.session_active = True
        
        if self.config.verbose:
            if persona:
                print(f"[AGENT] Persona: {persona['name']}")
            print(f"[AGENT] Session started at {datetime.now()}")
        
        return self.state
    
    def process_turn(self, audio_input: np.ndarray) -> ConversationState:
        """
        Process a single conversation turn.
        
        Args:
            audio_input: Audio from salesperson (16kHz, float32)
        
        Returns:
            ConversationState: Updated state with response
        """
        if not self.session_active:
            raise RuntimeError("No active session. Call start_session() first.")
        
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"[AGENT] Processing turn {self.state['turn_count'] + 1}")
            print(f"{'='*60}")
        
        turn_start = time.time()
        
        # Reset turn-specific state
        self.state = reset_turn_state(self.state)
        
        # Set audio input
        self.state["audio_input"] = audio_input
        
        # Run through graph
        self.state = self.graph.invoke(self.state)
        
        # Calculate total time
        total_time = time.time() - turn_start
        self.state["node_timings"]["total"] = total_time
        
        if self.config.verbose:
            self._print_turn_summary()
        
        return self.state
    
    def end_session(self) -> dict:
        """
        End the current session.
        
        Returns:
            dict: Session summary
        """
        if not self.session_active:
            return {"error": "No active session"}
        
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"[AGENT] Ending session: {self.state['session_id']}")
            print(f"{'='*60}")
        
        summary = {
            "session_id": self.state["session_id"],
            "user_id": self.state["user_id"],
            "total_turns": self.state["turn_count"],
            "history_length": len(self.state["history"]),
            "ended_at": datetime.now().isoformat()
        }
        
        self.session_active = False
        self.state["is_active"] = False
        
        if self.config.verbose:
            print(f"[AGENT] Total turns: {summary['total_turns']}")
            print(f"[AGENT] Session ended")
        
        return summary
    
    def get_state(self) -> Optional[ConversationState]:
        """Get current state."""
        return self.state
    
    def get_response_audio(self) -> Optional[np.ndarray]:
        """Get the audio output from last turn."""
        if self.state:
            return self.state.get("audio_output")
        return None
    
    def get_response_text(self) -> Optional[str]:
        """Get the text response from last turn."""
        if self.state:
            return self.state.get("llm_response")
        return None
    
    def _load_persona(self, persona_id: str) -> Persona:
        """Load persona by ID."""
        if self.config.use_mocks:
            from orchestration.mocks import get_persona
        else:
            from persona.agent import get_persona
        
        return get_persona(persona_id)
    
    def _load_scenario(self, scenario_id: str) -> Scenario:
        """Load scenario by ID."""
        # TODO: Implement scenario loading
        # For now, return None
        return None
    
    def _print_turn_summary(self) -> None:
        """Print summary of the processed turn."""
        state = self.state
        
        print(f"\n[TURN SUMMARY]")
        print(f"  Input: '{state.get('transcription', 'N/A')[:50]}...'")
        print(f"  Emotion: {state.get('emotion', {}).get('primary_emotion', 'N/A')}")
        print(f"  Output: '{state.get('llm_response', 'N/A')[:50]}...'")
        
        if state.get("node_timings"):
            print(f"\n[TIMINGS]")
            for node, time_taken in state["node_timings"].items():
                print(f"  {node}: {time_taken:.3f}s")


# Convenience functions
def create_agent(
    use_mocks: bool = True,
    verbose: bool = True,
    simple_mode: bool = False
) -> OrchestrationAgent:
    """
    Create an orchestration agent.
    
    Args:
        use_mocks: Use mock functions
        verbose: Print debug logs
        simple_mode: Use simplified graph
    
    Returns:
        OrchestrationAgent: Ready-to-use agent
    """
    return OrchestrationAgent(
        use_mocks=use_mocks,
        verbose=verbose,
        simple_mode=simple_mode
    )


def quick_test():
    """
    Quick test of the orchestration agent.
    Run: python -c "from orchestration.agent import quick_test; quick_test()"
    """
    print("="*60)
    print("VCAI Orchestration Agent - Quick Test")
    print("="*60)
    
    # Create agent
    agent = create_agent(use_mocks=True, verbose=True)
    
    # Start session
    agent.start_session(
        session_id="test_session",
        user_id="test_user",
        persona_id="difficult_customer"
    )
    
    # Simulate audio input (3 seconds of random noise)
    audio_input = np.random.randn(16000 * 3).astype(np.float32) * 0.1
    
    # Process turn
    result = agent.process_turn(audio_input)
    
    # Get outputs
    response_text = agent.get_response_text()
    response_audio = agent.get_response_audio()
    
    print(f"\n{'='*60}")
    print(f"RESULT")
    print(f"{'='*60}")
    print(f"Response text: {response_text}")
    print(f"Response audio: {response_audio.shape if response_audio is not None else 'None'}")
    
    # End session
    summary = agent.end_session()
    print(f"\nSession summary: {summary}")
    
    print("\n✅ Quick test completed!")


if __name__ == "__main__":
    quick_test()