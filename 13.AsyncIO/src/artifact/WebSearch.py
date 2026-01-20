from typing import Any, Dict, List
from src.artifact.Settings import Parameter
import asyncio
from langchain_tavily import TavilySearch

# =============================================================================
# WEB SEARCH (TAVILY FALLBACK ONLY FOR MISS TOPICS)
# =============================================================================

async def Web(topic: str, settings: Parameter) -> Dict[str, Any]:
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


async def BatchWebSearch(topics: List[str], settings: Parameter) -> List[Dict[str, Any]]:
    """
    Batch web search:
    - asyncio.Semaphore limits concurrency
    - asyncio.gather runs tasks concurrently
    """
    sem = asyncio.Semaphore(settings.max_concurrent_web)

    async def _one(t: str) -> Dict[str, Any]:
        async with sem:
            return await Web(t, settings)

    return await asyncio.gather(*[_one(t) for t in topics])