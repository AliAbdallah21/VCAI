# evaluation/graphs/evaluation_graph.py
"""
LangGraph Evaluation Pipeline
Author: Ismail

This creates and runs the evaluation graph:
    compute_quick_stats → analyzer → synthesizer → END
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from evaluation.state import EvaluationState
from evaluation.pipeline.analyzer import analyzer_node
from evaluation.pipeline.synthesizer import synthesizer_node
from evaluation.utils.report_formatter import compute_quick_stats_node


# Compiled graph (singleton)
_compiled_graph = None


def create_evaluation_graph():
    """
    Create the LangGraph Pipeline:

        compute_quick_stats (no LLM)
            ↓
        analyzer_node (LLM)
            ↓
        synthesizer_node (LLM)
            ↓
           END
    """

    graph = StateGraph(EvaluationState)

    graph.add_node("compute_quick_stats", compute_quick_stats_node)
    graph.add_node("analyzer", analyzer_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.set_entry_point("compute_quick_stats")
    graph.add_edge("compute_quick_stats", "analyzer")
    graph.add_edge("analyzer", "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()


def get_graph():
    """Get or create the compiled graph (singleton pattern)."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = create_evaluation_graph()
    return _compiled_graph


def run_evaluation(state: EvaluationState) -> EvaluationState:
    """
    Run the evaluation pipeline synchronously.
    
    This is what Menna's EvaluationManager calls.
    
    Args:
        state: Initial EvaluationState with session_id and mode
        
    Returns:
        Final EvaluationState with analysis_report and final_report
    """
    graph = get_graph()
    
    # Run the graph
    final_state = graph.invoke(state)
    
    return final_state


async def run_evaluation_async(state: EvaluationState) -> EvaluationState:
    """
    Run the evaluation pipeline asynchronously.
    
    Args:
        state: Initial EvaluationState with session_id and mode
        
    Returns:
        Final EvaluationState with analysis_report and final_report
    """
    graph = get_graph()
    
    # Run the graph asynchronously
    final_state = await graph.ainvoke(state)
    
    return final_state