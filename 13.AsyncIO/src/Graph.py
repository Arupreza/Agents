from typing import Any
from langgraph.graph import StateGraph, START, END
from src.State import AgentState
from src.nodes import PlanningNode, VectorSearchNode, WebSearchNode, MergeNode, AnalysisNode, ReportNode, need_web_fallback
# =============================================================================
# BUILD GRAPH
# =============================================================================

def BuildGraph() -> Any:
    builder = StateGraph(AgentState)

    builder.add_node("planning", PlanningNode)
    builder.add_node("vector_search", VectorSearchNode)
    builder.add_node("web_search", WebSearchNode)
    builder.add_node("merge", MergeNode)
    builder.add_node("analysis", AnalysisNode)
    builder.add_node("report", ReportNode)

    builder.add_edge(START, "planning")
    builder.add_edge("planning", "vector_search")

    builder.add_conditional_edges(
        "vector_search", need_web_fallback, 
        {
        "web_search": "web_search",
        "merge": "merge",
    })

    builder.add_edge("web_search", "merge")
    builder.add_edge("merge", "analysis")
    builder.add_edge("analysis", "report")
    builder.add_edge("report", END)

    return builder.compile()