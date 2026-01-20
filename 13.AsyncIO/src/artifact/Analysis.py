from typing import Any, Dict, List, Optional
from src.artifact.Settings import Parameter
from langchain_openai import ChatOpenAI
import asyncio

# =============================================================================
# ANALYSIS (LLM) - TASKGROUP + SEMAPHORE + TO_THREAD
# =============================================================================

async def Analyze(
    topic: str,
    vector_ctx: Optional[str],
    web_ctx: Optional[Any],
    settings: Parameter,
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


async def BatchAnalyze(items: List[Dict[str, Any]], settings: Parameter) -> List[Dict[str, str]]:
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