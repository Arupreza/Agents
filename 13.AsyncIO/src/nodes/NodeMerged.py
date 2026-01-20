from typing import Any, Dict, List
from langchain_core.messages import AIMessage
from src.State import AgentState

async def MergeNode(state: AgentState) -> Dict[str, Any]:
    """
    Merge results per topic:
    - vector context if HIT
    - web context only if MISS (because we only searched missing topics anyway)
    """
    vector_map: Dict[str, Dict[str, Any]] = {r["topic"]: r for r in state.get("vector_results", [])}
    web_map: Dict[str, Dict[str, Any]] = {r["topic"]: r for r in state.get("web_results", [])}

    """vector_map = {}

    # 2. Get the list of results from the state (default to empty list if missing)
    vector_results = state.get("vector_results", [])

    # 3. Iterate through each result dictionary in the list
    for r in vector_results:
        # 4. Use the value of the "topic" key as the new Dictionary Key
        key = r["topic"]
        
        # 5. Store the entire result dictionary as the value for that key
        vector_map[key] = r"""

    merged: List[Dict[str, Any]] = []
    for t in state["topics"]:
        v = vector_map.get(t, {})
        w = web_map.get(t, {})
        merged.append({
            "topic": t,
            "hit": v.get("hit", False),
            "best_distance": v.get("best_distance"),
            "vector": v.get("vector"),
            "web": w.get("web"),
            "vector_error": v.get("error"),
            "web_error": w.get("error"),
        })

    return {
        "merged_items": merged,
        "messages": [AIMessage(content="Merge complete. Proceeding to analysis.")]
    }