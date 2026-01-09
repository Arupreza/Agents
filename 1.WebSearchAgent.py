from dotenv import load_dotenv
load_dotenv()

from typing import List, TypedDict, Annotated
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# -----------------------------
# Output schema
# -----------------------------
class JobLinks(BaseModel):
    links: List[str] = Field(description="List of job posting URLs")

# -----------------------------
# State definition
# -----------------------------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# -----------------------------
# Tools
# -----------------------------
search_tool = TavilySearch(max_results=5)
tools = [search_tool]

# -----------------------------
# LLM with tools
# -----------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# -----------------------------
# Agent node
# -----------------------------
def agent(state: AgentState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# -----------------------------
# Conditional edge
# -----------------------------
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return "end"
    return "continue"

# -----------------------------
# Build graph
# -----------------------------
workflow = StateGraph(AgentState)

# Nodes
workflow.add_node("agent", agent)
workflow.add_node("tools", ToolNode(tools))

# Edges
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)
workflow.add_edge("tools", "agent")

# Compile
app = workflow.compile()

# -----------------------------
# Run
# -----------------------------
def main():
    system_msg = SystemMessage(content="""
You are a job search assistant.
Task: Find AI engineer jobs using LangChain in the Bay Area.
Steps:
1. Search for "AI engineer LangChain Bay Area jobs"
2. Extract ONLY the job posting URLs from search results
3. Return URLs in a clean list format

Be thorough - search multiple times if needed to find relevant postings.
""")
    
    user_msg = HumanMessage(content="Find AI engineer LangChain jobs in Bay Area")
    
    result = app.invoke(
        {"messages": [system_msg, user_msg]},
        {"recursion_limit": 10}
    )
    
    # Extract final response
    final_message = result["messages"][-1]
    print("\n=== Job Links ===")
    print(final_message.content)

if __name__ == "__main__":
    main()