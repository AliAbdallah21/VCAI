from __future__ import annotations

from langgraph.graph import END, StateGraph

from evaluation.state import EvaluationState
from evaluation.pipeline.analyzer import analyzer_node
from evaluation.pipeline.synthesizer import synthesizer_node

# ✅ Quick stats node comes from utils 
from evaluation.utils.report_formatter import compute_quick_stats_node


def create_evaluation_graph():
    """
    LangGraph Pipeline :

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
