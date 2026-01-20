from typing import Any, Dict
from langchain_core.messages import AIMessage
from src.State import AgentState
from src.artifact.WebSearch import BatchWebSearch

def need_web_fallback(state: AgentState) -> str:
    return "web_search" if state.get("missing_topics") else "merge"


async def WebSearchNode(state: AgentState) -> Dict[str, Any]:
    settings = state["settings"]
    missing = state["missing_topics"]

    results = await BatchWebSearch(missing, settings)
    return {
        "web_results": results,
        "messages": [AIMessage(content=f"Web fallback done for {len(missing)} missing topics.")]
    }