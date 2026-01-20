from src.artifact.Settings import GetVectorStore
from typing import Any, Dict
from langchain_core.messages import AIMessage
from src.State import AgentState
# =============================================================================
# LANGGRAPH NODES
# =============================================================================

async def PlanningNode(state: AgentState) -> Dict[str, Any]:
    settings = state["settings"]
    topics = state["topics"]

    # Warm-up vectorstore once (deterministic behavior)
    _ = GetVectorStore(settings)

    msg = (
        "Planning complete.\n"
        f"- Topics: {len(topics)}\n"
        f"- Vector HIT threshold (max distance): {settings.vector_hit_max_distance}\n"
        f"- Vector concurrency: {settings.max_concurrent_vector}\n"
        f"- Web concurrency: {settings.max_concurrent_web}\n"
        f"- LLM concurrency: {settings.max_concurrent_llm}\n"
    )
    return {"messages": [AIMessage(content=msg)]}