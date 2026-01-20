from typing import Any, Dict, List, Tuple
from src.artifact.Settings import Parameter, GetVectorStore, _format_docs
import asyncio

# =============================================================================
# VECTOR SEARCH (PGVector-first) WITH THRESHOLD
# =============================================================================

async def Vector(topic: str, settings: Parameter) -> Dict[str, Any]:
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
        store = GetVectorStore(settings)

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

        ### From tuple extract the value
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


async def BatchVectorSearch(topics: List[str], settings: Parameter) -> List[Dict[str, Any]]:
    """
    Batch vector search:
    - asyncio.Semaphore limits concurrency
    - asyncio.gather runs tasks concurrently
    """
    sem = asyncio.Semaphore(settings.max_concurrent_vector)

    async def _one(t: str) -> Dict[str, Any]:
        async with sem:
            return await Vector(t, settings)

    return await asyncio.gather(*[_one(t) for t in topics])