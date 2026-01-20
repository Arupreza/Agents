import os
import sys
from dataclasses import dataclass
import threading
from typing import Any, List, Optional
from langchain_postgres.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings

# =============================================================================
# SETTINGS
# =============================================================================

@dataclass(frozen=True)
class Parameter:
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

def GetVectorStore(settings: Parameter) -> PGVector:
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