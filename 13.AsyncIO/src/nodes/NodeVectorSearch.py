from typing import Any, Dict
from langchain_core.messages import AIMessage
from src.State import AgentState
from src.artifact.VectorSearch import BatchVectorSearch

async def VectorSearchNode(state: AgentState) -> Dict[str, Any]:
    settings = state["settings"]
    topics = state["topics"]

    results = await BatchVectorSearch(topics, settings)
    missing = [r["topic"] for r in results if not r.get("hit")]

    return {
        "vector_results": results,
        "missing_topics": missing,
        "messages": [AIMessage(content=f"Vector search done. Hits: {len(topics)-len(missing)}, Missing: {len(missing)}")]
    }