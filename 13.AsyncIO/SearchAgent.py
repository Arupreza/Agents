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

import asyncio
import os
import sys
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict, Annotated, Tuple

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_tavily import TavilySearch
from langchain_postgres.vectorstores import PGVector

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()


# =============================================================================
# SETTINGS
# =============================================================================

@dataclass(frozen=True)
class Settings:
    # Vector DB (must match your ingestion)
    embed_model: str = "text-embedding-3-small"
    collection_name: str = "research_papers"
    vector_top_k: int = 3
    use_jsonb: bool = True

    # HIT/MISS rule (cosine distance: smaller is better)
    # Start with 0.35; tune by printing best_distance for a few topics.
    vector_hit_max_distance: float = 0.35
    
    """
    That implies the similarity metric is cosine distance (not cosine similarity).
    Cosine similarity: higher is better (1.0 = identical direction)
    Cosine distance: lower is better (0.0 = identical direction)
    Many implementations define cosine distance as:

    cosine_distance = 1 − cosine_similarity

    So:
    If similarity = 0.90 → distance = 0.10 (very close)
    If similarity = 0.50 → distance = 0.50 (weak)
    If similarity = 0.10 → distance = 0.90 (basically unrelated)
    Your code uses similarity_search_with_score() which (in PGVector setups) 
    typically returns a distance-like score for vector_cosine_ops, meaning 
    smaller is better.
    """
    # Tavily
    tavily_max_results: int = 2

    # Timeouts in Seconds
    vector_timeout_s: float = 8.0
    web_timeout_s: float = 10.0

    # Concurrency
    max_concurrent_vector: int = 3
    max_concurrent_web: int = 3
    max_concurrent_llm: int = 3

    # LLM
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0


def require_env(keys: List[str]) -> None:
    missing = []
    for k in keys:
        value = os.getenv(k)
        if value is None or value.strip() == "":
            missing.append(k)

    if missing:
        print(f"CRITICAL ERROR: Missing environment variables: {', '.join(missing)}")
        sys.exit(1)


def get_connection_string(prefix: str = "CTX") -> str:
    keys = [f"{prefix}_USER", f"{prefix}_PASS", f"{prefix}_HOST", f"{prefix}_PORT", f"{prefix}_DB"]
    require_env(keys)
    user = os.getenv(f"{prefix}_USER")
    password = os.getenv(f"{prefix}_PASS")
    host = os.getenv(f"{prefix}_HOST")
    port = os.getenv(f"{prefix}_PORT")
    db = os.getenv(f"{prefix}_DB")
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{db}"


# =============================================================================
# THREAD-SAFE SINGLETON: langchain_postgres PGVector
# =============================================================================

_VECTORSTORE: Optional[PGVector] = None
_VECTORSTORE_LOCK = threading.Lock()

"""
Why we use _VECTORSTORE_LOCK (thread lock)

In this project, vector search is called inside asyncio.to_thread(...), which executes
synchronous code in a threadpool. That means multiple topics can trigger vector search
at the same time in different threads.

If two threads try to initialize PGVector simultaneously, SQLAlchemy may attempt to
register the same tables/metadata more than once, which can raise errors such as:
"Table 'langchain_pg_collection' is already defined ...".

The lock guarantees that only one thread performs the one-time PGVector initialization,
and all other threads reuse the same singleton instance.
"""

def GetVectorstore(settings: Settings) -> PGVector:
    """
    Thread-safe singleton creation for vectorstore.
    Prevents any possible concurrent initialization hazards.
    """
    global _VECTORSTORE
    if _VECTORSTORE is not None:
        return _VECTORSTORE

    with _VECTORSTORE_LOCK:
        if _VECTORSTORE is None:
            vector_connection = get_connection_string("CTX")
            embeddings = OpenAIEmbeddings(model=settings.embed_model)
            _VECTORSTORE = PGVector(
                embeddings=embeddings,
                connection=vector_connection,
                collection_name=settings.collection_name,
                use_jsonb=settings.use_jsonb,
            )
    return _VECTORSTORE


# =============================================================================
# HELPERS
# =============================================================================

def _format_docs(docs: List[Any]) -> str:
    parts: List[str] = []
    for d in docs:
        meta = d.metadata or {}
        src = meta.get("source", "unknown")
        page = meta.get("page", "N/A")
        parts.append(f"[Source] {src} (Page {page})\n{d.page_content}")
    return "\n\n".join(parts)


# =============================================================================
# VECTOR SEARCH (PGVector-first) WITH THRESHOLD
# =============================================================================

async def VectorSearch(topic: str, settings: Settings) -> Dict[str, Any]:
    """
    Per-topic vector search with:
    - asyncio.to_thread() wrapping sync DB call
    - asyncio.wait_for() timeout
    - HIT defined by best_distance <= vector_hit_max_distance

    IMPORTANT:
    - Uses similarity_search_with_score() so we can decide relevance.
    - Without a threshold, everything looks like a "hit" because k docs always return.
    """
    print(f"🔎 vector_search: {topic}")

    def _sync() -> Dict[str, Any]:
        store = GetVectorstore(settings)

        # returns List[Tuple[Document, float]] where float is distance/score
        pairs: List[Tuple[Any, float]] = store.similarity_search_with_score(
            topic, k=settings.vector_top_k
        )

        if not pairs:
            return {"topic": topic, "hit": False, "vector": None, "best_distance": None}

        # best (smallest distance)
        best_document, best_distance_from_vector = min(pairs, key=lambda x: x[1])

        """ pairs = [
            (docA, 0.42),
            (docB, 0.18),
            (docC, 0.31),
        ]
        
        Each element is a tuple:
            index 0 = document
            index 1 = distance (cosine distance in your setup)

        ### From tuple by lambda inside function extract the value 
        lambda x: x[1]
        
        def get_second_item(x):
            return x[1]
        """
        
        hit = float(best_distance_from_vector) <= settings.vector_hit_max_distance

        # For debug/tuning:
        # Print distances so you can tune threshold.
        print(f"   [best_distance_from_vector] {topic} -> {float(best_distance_from_vector):.4f} | hit={hit}")

        if hit:
            docs = [doc for (doc, _dist) in pairs]
            vec_text = _format_docs(docs)
        else:
            vec_text = None

        return {
            "topic": topic,
            "hit": hit,
            "vector": vec_text,
            "best_distance": float(best_distance_from_vector),
        }

    try:
        return await asyncio.wait_for(asyncio.to_thread(_sync), timeout=settings.vector_timeout_s)
    except asyncio.TimeoutError:
        print(f"⏱️ vector timeout: {topic}")
        return {
            "topic": topic,
            "hit": False,
            "vector": None,
            "best_distance": None,
            "error": "vector_timeout",
        }
    except Exception as e:
        # Treat errors as MISS so web fallback can proceed
        print(f"⚠️ vector error ({topic}): {repr(e)}")
        return {
            "topic": topic,
            "hit": False,
            "vector": None,
            "best_distance": None,
            "error": f"vector_error:{type(e).__name__}",
        }


async def BatchVectorSearch(topics: List[str], settings: Settings) -> List[Dict[str, Any]]:
    """
    Batch vector search:
    - asyncio.Semaphore limits concurrency
    - asyncio.gather runs tasks concurrently
    """
    sem = asyncio.Semaphore(settings.max_concurrent_vector)

    async def _one(t: str) -> Dict[str, Any]:
        async with sem:
            return await VectorSearch(t, settings)

    return await asyncio.gather(*[_one(t) for t in topics])


# =============================================================================
# WEB SEARCH (TAVILY FALLBACK ONLY FOR MISS TOPICS)
# =============================================================================

async def WebSearch(topic: str, settings: Settings) -> Dict[str, Any]:
    """
    Per-topic Tavily web search with:
    - asyncio.to_thread wrapping sync tool call
    - asyncio.wait_for timeout
    """
    print(f"🌐 web_search (fallback): {topic}")
    tool = TavilySearch(max_results=settings.tavily_max_results)

    def _sync() -> Any:
        return tool.invoke({"query": topic})

    try:
        response = await asyncio.wait_for(asyncio.to_thread(_sync), timeout=settings.web_timeout_s)
        return {"topic": topic, "web": response}
    except asyncio.TimeoutError:
        print(f"⏱️ web timeout: {topic}")
        return {"topic": topic, "web": None, "error": "web_timeout"}
    except Exception as e:
        print(f"⚠️ web error ({topic}): {repr(e)}")
        return {"topic": topic, "web": None, "error": f"web_error:{type(e).__name__}"}


async def BatchWebSearch(topics: List[str], settings: Settings) -> List[Dict[str, Any]]:
    """
    Batch web search:
    - asyncio.Semaphore limits concurrency
    - asyncio.gather runs tasks concurrently
    """
    sem = asyncio.Semaphore(settings.max_concurrent_web)

    async def _one(t: str) -> Dict[str, Any]:
        async with sem:
            return await WebSearch(t, settings)

    return await asyncio.gather(*[_one(t) for t in topics])


# =============================================================================
# ANALYSIS (LLM) - TASKGROUP + SEMAPHORE + TO_THREAD
# =============================================================================

async def Analyze(
    topic: str,
    vector_ctx: Optional[str],
    web_ctx: Optional[Any],
    settings: Settings,
    llm: ChatOpenAI,
) -> Dict[str, str]:
    print(f"🤖 analyze: {topic}")

    prompt_parts: List[str] = [f"Topic: {topic}", ""]

    if vector_ctx:
        prompt_parts += [
            "[Internal Knowledge Base (PGVector)]",
            str(vector_ctx)[:9000],
            "",
        ]
    if web_ctx:
        prompt_parts += [
            "[Web Search (Tavily Fallback)]",
            str(web_ctx)[:9000],
            "",
        ]

    prompt_parts += [
        "Task: Provide a 2-sentence synthesis.",
        "- Prefer internal knowledge if present.",
        "- If internal is missing, rely on web results.",
        "- If uncertain, state uncertainty explicitly.",
    ]

    prompt = "\n".join(prompt_parts)
    resp = await asyncio.to_thread(llm.invoke, prompt)
    return {"topic": topic, "summary": resp.content}


async def BatchAnalyze(items: List[Dict[str, Any]], settings: Settings) -> List[Dict[str, str]]:
    """
    - asyncio.TaskGroup structured concurrency
    - asyncio.Semaphore to limit concurrent LLM calls
    """
    llm = ChatOpenAI(model=settings.llm_model, temperature=settings.llm_temperature)
    sem = asyncio.Semaphore(settings.max_concurrent_llm)

    async def _one(item: Dict[str, Any]) -> Dict[str, str]:
        async with sem:
            return await Analyze(
                topic=item["topic"],
                vector_ctx=item.get("vector"),
                web_ctx=item.get("web"),
                settings=settings,
                llm=llm,
            )

    results: List[Dict[str, str]] = []
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(_one(i)) for i in items]

    for t in tasks:
        results.append(t.result())
    return results


def FormatReport(analyses: List[Dict[str, str]]) -> str:
    lines = ["# Hybrid Research Report (PGVector-first, Tavily fallback)", ""]
    for a in analyses:
        lines += [f"## {a['topic']}", a["summary"], ""]
    lines.append("*Policy: PGVector first; Tavily only when PGVector is below relevance threshold.*")
    return "\n".join(lines)


# =============================================================================
# LANGGRAPH STATE
# =============================================================================

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    topics: List[str]
    settings: Settings

    vector_results: List[Dict[str, Any]]
    missing_topics: List[str]
    web_results: List[Dict[str, Any]]
    merged_items: List[Dict[str, Any]]
    analyses: List[Dict[str, str]]
    report: str


# =============================================================================
# LANGGRAPH NODES
# =============================================================================

async def PlanningNode(state: AgentState) -> Dict[str, Any]:
    settings = state["settings"]
    topics = state["topics"]

    # Warm-up vectorstore once (deterministic behavior)
    _ = GetVectorstore(settings)

    msg = (
        "Planning complete.\n"
        f"- Topics: {len(topics)}\n"
        f"- Vector HIT threshold (max distance): {settings.vector_hit_max_distance}\n"
        f"- Vector concurrency: {settings.max_concurrent_vector}\n"
        f"- Web concurrency: {settings.max_concurrent_web}\n"
        f"- LLM concurrency: {settings.max_concurrent_llm}\n"
    )
    return {"messages": [AIMessage(content=msg)]}


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


async def MergeNode(state: AgentState) -> Dict[str, Any]:
    """
    Merge results per topic:
    - vector context if HIT
    - web context only if MISS (because we only searched missing topics anyway)
    """
    vector_map: Dict[str, Dict[str, Any]] = {r["topic"]: r for r in state.get("vector_results", [])}
    web_map: Dict[str, Dict[str, Any]] = {r["topic"]: r for r in state.get("web_results", [])}

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


async def AnalysisNode(state: AgentState) -> Dict[str, Any]:
    analyses = await BatchAnalyze(state["merged_items"], state["settings"])
    return {
        "analyses": analyses,
        "messages": [AIMessage(content=f"Analysis complete: {len(analyses)} summaries generated.")]
    }


async def ReportNode(state: AgentState) -> Dict[str, Any]:
    report = FormatReport(state["analyses"])
    return {"report": report, "messages": [AIMessage(content=report)]}


# =============================================================================
# BUILD GRAPH
# =============================================================================

def build_graph() -> Any:
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


# =============================================================================
# MAIN
# =============================================================================

async def main() -> None:
    require_env(["OPENAI_API_KEY", "TAVILY_API_KEY"])
    require_env(["CTX_USER", "CTX_PASS", "CTX_HOST", "CTX_PORT", "CTX_DB"])

    # Example: Your exact topics (3 should hit; mango should miss and fallback)
    topics = [
        "CAN bus security vulnerabilities",
        "Automotive intrusion detection systems",
        "Give me mango tree plantation plan",
    ]

    settings = Settings(
        # Tune threshold if needed after observing printed best_distance
        vector_hit_max_distance=0.35,

        max_concurrent_vector=2,
        max_concurrent_web=2,
        max_concurrent_llm=2,

        vector_timeout_s=8.0,
        web_timeout_s=10.0,
    )

    app = build_graph()

    initial_state: AgentState = {
        "messages": [HumanMessage(content=f"Run PGVector-first with Tavily fallback on: {topics}")],
        "topics": topics,
        "settings": settings,
        "vector_results": [],
        "missing_topics": [],
        "web_results": [],
        "merged_items": [],
        "analyses": [],
        "report": "",
    }

    final_state = await app.ainvoke(initial_state)

    print("\n" + "=" * 80)
    print("FINAL REPORT")
    print("=" * 80)
    print(final_state["report"])


if __name__ == "__main__":
    asyncio.run(main())