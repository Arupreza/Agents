from typing import Any, Dict, List, TypedDict, Annotated
from src.artifact.Settings import Parameter
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    topics: List[str]
    settings: Parameter
    vector_results: List[Dict[str, Any]]
    missing_topics: List[str]
    web_results: List[Dict[str, Any]]
    merged_items: List[Dict[str, Any]]
    analyses: List[Dict[str, str]]
    report: str