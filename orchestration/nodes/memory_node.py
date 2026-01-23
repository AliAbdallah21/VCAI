# orchestration/nodes/memory_node.py
"""
Memory node for orchestration.
Handles message storage, retrieval, and checkpoint creation.
"""

import time
from datetime import datetime
import uuid

from orchestration.state import ConversationState
from orchestration.config import OrchestrationConfig
from shared.types import Message, MemoryCheckpoint


def memory_load_node(
    state: ConversationState,
    config: OrchestrationConfig = None
) -> ConversationState:
    """
    Load memory at the start of turn processing.
    
    This node:
    1. Retrieves session memory (checkpoints + recent messages)
    2. Updates state with memory context
    
    Args:
        state: Current conversation state
        config: Orchestration configuration
    
    Returns:
        ConversationState: Updated state with memory
    """
    start_time = time.time()
    
    if config and config.verbose:
        print("\n[MEMORY LOAD] Loading session memory...")
    
    try:
        session_id = state.get("session_id")
        
        # Get memory functions based on config
        if config and config.use_mocks:
            from orchestration.mocks import get_session_memory
        else:
            from memory.agent import get_session_memory
        
        # Load memory
        memory = get_session_memory(session_id)
        
        # Update state
        state["memory"] = memory
        
        # Log timing
        elapsed = time.time() - start_time
        state["node_timings"]["memory_load"] = elapsed
        
        if config and config.verbose:
            print(f"[MEMORY LOAD] Loaded {len(memory['checkpoints'])} checkpoints, {len(memory['recent_messages'])} recent messages")
            print(f"[MEMORY LOAD] Time: {elapsed:.3f}s")
        
    except Exception as e:
        # Use empty memory on error
        state["memory"] = {
            "session_id": state.get("session_id", ""),
            "checkpoints": [],
            "recent_messages": [],
            "total_turns": 0
        }
        if config and config.verbose:
            print(f"[MEMORY LOAD] Error (using empty memory): {str(e)}")
    
    return state


def memory_save_node(
    state: ConversationState,
    config: OrchestrationConfig = None
) -> ConversationState:
    """
    Save messages after turn processing.
    
    This node:
    1. Stores salesperson message
    2. Stores VC response message
    3. Creates checkpoint if needed (every N turns)
    4. Updates history
    
    Args:
        state: Current conversation state
        config: Orchestration configuration
    
    Returns:
        ConversationState: Updated state
    """
    start_time = time.time()
    
    if config and config.verbose:
        print("\n[MEMORY SAVE] Saving turn to memory...")
    
    try:
        session_id = state.get("session_id")
        turn_count = state.get("turn_count", 0)
        
        # Get memory functions based on config
        if config and config.use_mocks:
            from orchestration.mocks import store_message, store_checkpoint
        else:
            from memory.agent import store_message, store_checkpoint
        
        # ══════════════════════════════════════════════════════════════════
        # 1. Store salesperson message
        # ══════════════════════════════════════════════════════════════════
        if state.get("transcription"):
            salesperson_msg: Message = {
                "id": str(uuid.uuid4()),
                "turn": turn_count,
                "speaker": "salesperson",
                "text": state["transcription"],
                "emotion": state.get("emotion"),
                "audio_path": None,
                "timestamp": datetime.now()
            }
            store_message(session_id, salesperson_msg)
            
            # Add to history
            state["history"].append(salesperson_msg)
            
            if config and config.verbose:
                print(f"[MEMORY SAVE] Stored salesperson message: {state['transcription'][:50]}...")
        
        # ══════════════════════════════════════════════════════════════════
        # 2. Store VC message
        # ══════════════════════════════════════════════════════════════════
        if state.get("llm_response"):
            vc_msg: Message = {
                "id": str(uuid.uuid4()),
                "turn": turn_count,
                "speaker": "vc",
                "text": state["llm_response"],
                "emotion": None,
                "audio_path": None,
                "timestamp": datetime.now()
            }
            store_message(session_id, vc_msg)
            
            # Add to history
            state["history"].append(vc_msg)
            
            if config and config.verbose:
                print(f"[MEMORY SAVE] Stored VC message: {state['llm_response'][:50]}...")
        
        # ══════════════════════════════════════════════════════════════════
        # 3. Check if checkpoint needed
        # ══════════════════════════════════════════════════════════════════
        if config and config.enable_checkpoints:
            if _should_create_checkpoint(turn_count, config):
                if config.verbose:
                    print(f"[MEMORY SAVE] Creating checkpoint at turn {turn_count}...")
                _create_checkpoint(state, config)
        
        # ══════════════════════════════════════════════════════════════════
        # 4. Increment turn count
        # ══════════════════════════════════════════════════════════════════
        state["turn_count"] = turn_count + 1
        
        # Log timing
        elapsed = time.time() - start_time
        state["node_timings"]["memory_save"] = elapsed
        
        if config and config.verbose:
            print(f"[MEMORY SAVE] Saved turn {turn_count}")
            print(f"[MEMORY SAVE] History length: {len(state['history'])}")
            print(f"[MEMORY SAVE] Time: {elapsed:.3f}s")
        
    except Exception as e:
        if config and config.verbose:
            print(f"[MEMORY SAVE] Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    return state


def _should_create_checkpoint(turn_count: int, config: OrchestrationConfig) -> bool:
    """
    Determine if a checkpoint should be created.
    
    Creates checkpoint when:
    - turn_count > 0 (not first turn)
    - turn_count is divisible by checkpoint_interval
    
    Args:
        turn_count: Current turn number
        config: Configuration
    
    Returns:
        bool: True if checkpoint should be created
    """
    interval = config.checkpoint_interval if config else 5
    return turn_count > 0 and turn_count % interval == 0


def _create_checkpoint(state: ConversationState, config: OrchestrationConfig) -> None:
    """
    Create a memory checkpoint from recent messages.
    
    Uses LLM to summarize conversation and extract key points.
    
    Args:
        state: Current state
        config: Configuration
    """
    try:
        # Get functions based on config
        if config and config.use_mocks:
            from orchestration.mocks import store_checkpoint
            # Mock summarization
            summary = "ملخص المحادثة"
            key_points = ["نقطة 1", "نقطة 2"]
        else:
            from memory.agent import store_checkpoint
            from llm.agent import summarize_conversation, extract_key_points
            
            # Get messages to summarize
            session_id = state.get("session_id")
            turn_count = state.get("turn_count", 0)
            history = state.get("history", [])
            
            # Get last N messages (based on checkpoint interval)
            interval = config.checkpoint_interval if config else 5
            messages_to_summarize = history[-(interval * 2):]  # *2 because each turn has 2 messages
            
            if not messages_to_summarize:
                if config.verbose:
                    print("[MEMORY SAVE] No messages to summarize")
                return
            
            # Use LLM to summarize
            if config.verbose:
                print(f"[MEMORY SAVE] Summarizing {len(messages_to_summarize)} messages...")
            
            summary = summarize_conversation(messages_to_summarize)
            key_points = extract_key_points(messages_to_summarize)
            
            if config.verbose:
                print(f"[MEMORY SAVE] Summary: {summary[:100]}...")
                print(f"[MEMORY SAVE] Key points: {key_points}")
        
        session_id = state.get("session_id")
        turn_count = state.get("turn_count", 0)
        interval = config.checkpoint_interval if config else 5
        
        # Create checkpoint
        checkpoint: MemoryCheckpoint = {
            "id": str(uuid.uuid4()),
            "session_id": session_id,
            "turn_range": (max(0, turn_count - interval + 1), turn_count),
            "summary": summary,
            "key_points": key_points,
            "customer_preferences": {},  # TODO: Extract from conversation
            "objections_raised": [],      # TODO: Extract from conversation
            "created_at": datetime.now()
        }
        
        # Store checkpoint
        success = store_checkpoint(session_id, checkpoint)
        
        if config and config.verbose:
            if success:
                print(f"[MEMORY SAVE] ✅ Checkpoint created for turns {checkpoint['turn_range']}")
            else:
                print(f"[MEMORY SAVE] ❌ Failed to store checkpoint")
        
    except Exception as e:
        if config and config.verbose:
            print(f"[MEMORY SAVE] Checkpoint error: {str(e)}")
            import traceback
            traceback.print_exc()