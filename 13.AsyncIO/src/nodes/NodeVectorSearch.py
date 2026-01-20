from typing import Any, Dict
from langchain_core.messages import AIMessage
from src.State import AgentState
from src.artifact.VectorSearch import BatchVectorSearch

async def VectorSearchNode(state: AgentState) -> Dict[str, Any]:
    settings = state["settings"]
    topics = state["topics"]

    results = await BatchVectorSearch(topics, settings)
    missing = [r["topic"] for r in results if not r.get("hit")]
    
    # 1. Initialize an empty list to store the missing topics
    """missing = []

    # 2. Start iterating through the list of results
    for r in results:
        # 3. Check if the 'hit' key is False or missing
        # r.get("hit") returns the value of "hit". 'not' flips it to True if there was no hit.
        if not r.get("hit"):
            # 4. If it was a miss, add the name of the topic to our list
            missing.append(r["topic"])"""

    return {
        "vector_results": results,
        "missing_topics": missing,
        "messages": [AIMessage(content=f"Vector search done. Hits: {len(topics)-len(missing)}, Missing: {len(missing)}")]
    }