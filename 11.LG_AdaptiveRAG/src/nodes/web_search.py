from typing import Any, Dict
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from src.state import GraphState

# Initialize the tool outside the function for better performance
web_search_tool = TavilySearchResults(k=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    """
    Web search based on the question using Tavily.
    """
    print("---NODE: WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])

    # 1. Execute search - ensure it returns a list of dicts
    # TavilySearchResults.invoke returns: [{"url": "...", "content": "..."}, ...]
    tavily_results = web_search_tool.invoke({"query": question})
    
    # 2. Safely join results into a single string
    # We use .get("content") to avoid KeyErrors
    joined_tavily_result = "\n".join(
        [res.get("content", "") for res in tavily_results]
    )
    
    # 3. Convert to LangChain Document format to maintain state consistency
    web_results = Document(page_content=joined_tavily_result)
    
    # 4. Append to existing documents
    documents.append(web_results)
    
    return {"documents": documents, "question": question}