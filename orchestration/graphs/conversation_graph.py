# orchestration/graphs/conversation_graph.py
"""
Main LangGraph conversation graph.
Defines the flow of conversation processing.
"""

from typing import Callable
from langgraph.graph import StateGraph, END

from orchestration.state import ConversationState
from orchestration.config import OrchestrationConfig, DEFAULT_CONFIG
from orchestration.nodes import (
    stt_node,
    emotion_node,
    memory_load_node,
    memory_save_node,
    llm_node,
    tts_node
)


def create_conversation_graph(config: OrchestrationConfig = None) -> StateGraph:
    """
    Create the LangGraph conversation workflow.

    RAG is NOT part of this pipeline. The virtual customer responds solely based
    on its persona configuration. RAG is used only in the post-session evaluation
    pipeline to fact-check what the salesperson said against the knowledge base.

    Flow:
        START
          ↓
        memory_load (load session context)
          ↓
        stt (audio → text)
          ↓
        emotion (detect emotion)
          ↓
        llm (generate response using persona + memory + emotion)
          ↓
        tts (text → audio)
          ↓
        memory_save (store turn)
          ↓
        END

    Args:
        config: Orchestration configuration

    Returns:
        StateGraph: Compiled LangGraph workflow
    """
    if config is None:
        config = DEFAULT_CONFIG

    workflow = StateGraph(ConversationState)

    # Create node wrappers that inject config
    def _memory_load(state):
        return memory_load_node(state, config)

    def _stt(state):
        return stt_node(state, config)

    def _emotion(state):
        return emotion_node(state, config)

    def _llm(state):
        return llm_node(state, config)

    def _tts(state):
        return tts_node(state, config)

    def _memory_save(state):
        return memory_save_node(state, config)

    # Add nodes
    workflow.add_node("memory_load", _memory_load)
    workflow.add_node("stt", _stt)
    workflow.add_node("emotion", _emotion)
    workflow.add_node("llm", _llm)
    workflow.add_node("tts", _tts)
    workflow.add_node("memory_save", _memory_save)

    # Define edges (linear flow — no RAG node)
    workflow.set_entry_point("memory_load")
    workflow.add_edge("memory_load", "stt")
    workflow.add_edge("stt", "emotion")
    workflow.add_edge("emotion", "llm")
    workflow.add_edge("llm", "tts")
    workflow.add_edge("tts", "memory_save")
    workflow.add_edge("memory_save", END)

    return workflow.compile()


def create_simple_graph(config: OrchestrationConfig = None) -> StateGraph:
    """
    Create a simplified graph without memory nodes.
    Useful for testing or minimal setup.

    Flow:
        START → stt → emotion → llm → tts → END
    
    Args:
        config: Orchestration configuration
    
    Returns:
        StateGraph: Compiled LangGraph workflow
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    workflow = StateGraph(ConversationState)
    
    def _stt(state):
        return stt_node(state, config)
    
    def _emotion(state):
        return emotion_node(state, config)
    
    def _llm(state):
        return llm_node(state, config)
    
    def _tts(state):
        return tts_node(state, config)
    
    workflow.add_node("stt", _stt)
    workflow.add_node("emotion", _emotion)
    workflow.add_node("llm", _llm)
    workflow.add_node("tts", _tts)
    
    workflow.set_entry_point("stt")
    workflow.add_edge("stt", "emotion")
    workflow.add_edge("emotion", "llm")
    workflow.add_edge("llm", "tts")
    workflow.add_edge("tts", END)
    
    return workflow.compile()


# Conditional routing (for future use)
def should_continue(state: ConversationState) -> str:
    """
    Determine if conversation should continue or end.

    Args:
        state: Current state

    Returns:
        str: "continue" or "end"
    """
    if state.get("should_end"):
        return "end"
    if state.get("error"):
        return "end"
    return "continue"