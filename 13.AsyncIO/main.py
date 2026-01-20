"""
SearchAgent_LangGraph_Fallback.py (FULL CODE)

Requirement implemented
-----------------------
- PGVector-first retrieval:
    * For each topic, search PGVector
    * Decide HIT vs MISS using a similarity threshold (distance)
- Tavily fallback:
    * Only topics classified as MISS will be searched on Tavily
- LLM synthesis uses:
    * PGVector context if HIT
    * Tavily context if MISS (or if vector failed/timeout)

Why this fixes your issue
-------------------------
`similarity_search(k=3)` almost always returns some docs, even if irrelevant.
So "bool(docs)" is NOT a good "availability" check.
We instead use `similarity_search_with_score()` and a distance threshold.

Asyncio primitives preserved
----------------------------
- asyncio.to_thread()
- asyncio.gather()
- asyncio.Semaphore
- asyncio.TaskGroup
- asyncio.wait_for()

Dependencies (recommended)
-------------------------
uv add langgraph langchain-openai langchain-postgres langchain-tavily python-dotenv psycopg[binary] sqlalchemy

Env vars required
-----------------
OPENAI_API_KEY
TAVILY_API_KEY
CTX_USER CTX_PASS CTX_HOST CTX_PORT CTX_DB

Run
---
python SearchAgent_LangGraph_Fallback.py
"""

# =============================================================================
# MAIN
# =============================================================================

import asyncio
import os
from dotenv import load_dotenv

# Internal Imports (From your packages)
from src.State import AgentState
from src.artifact.Settings import Parameter # Assuming you have a Settings class here
from src.Graph import BuildGraph  # This is the entry point for your LangGraph
from langchain_core.messages import HumanMessage

def require_env(keys: list[str]):
    """Validate environment variables."""
    for key in keys:
        if not os.getenv(key):
            raise ValueError(f"MISSING CRITICAL ENV VAR: {key}")

async def main() -> None:
    # 1. Load and Validate
    load_dotenv()
    require_env(["OPENAI_API_KEY", "TAVILY_API_KEY"])
    
    # 2. Configuration
    topics = [
        "CAN bus security vulnerabilities",
        "Automotive intrusion detection systems",
        "Give me mango tree plantation plan",
    ]

    settings = Parameter(
        vector_hit_max_distance=0.35,
        max_concurrent_vector=2,
        max_concurrent_web=2,
        max_concurrent_llm=2,
        vector_timeout_s=8.0,
        web_timeout_s=10.0,
    )

    # 3. Build the Graph
    # Ensure src/nodes/Graph.py has a function build_graph()
    app = BuildGraph()

    # 4. Initialize State
    initial_state: AgentState = {
        "messages": [HumanMessage(content=f"Processing topics: {topics}")],
        "topics": topics,
        "settings": settings,
        "vector_results": [],
        "missing_topics": [],
        "web_results": [],
        "merged_items": [],
        "analyses": [],
        "report": "",
    }

    # 5. Execute
    print("--- Starting Agentic RAG Pipeline ---")
    final_state = await app.ainvoke(initial_state)

    print("\n" + "=" * 80)
    print("FINAL REPORT")
    print("=" * 80)
    print(final_state.get("report", "No report generated."))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\nCRITICAL FAILURE: {e}")