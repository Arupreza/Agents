from dotenv import load_dotenv
load_dotenv()

import os

from typing import TypedDict, Annotated, Sequence
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# -----------------------------
# State definition
# -----------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# -----------------------------
# Tools
# -----------------------------
search_tool = TavilySearch(
    max_results=5,
    description="Search the web for job postings and career information"
)
tools = [search_tool]

# -----------------------------
# LLM with ReAct prompting
# -----------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# -----------------------------
# ReAct System Prompt
# -----------------------------
REACT_SYSTEM_PROMPT = """You are a job search assistant.

Task: Find AI engineer jobs using LangChain in the Bay Area.

Instructions:
1. Search for "AI engineer LangChain Bay Area jobs" using the tavily_search tool
2. Extract ONLY the job posting URLs from search results
3. Search multiple times with different queries if needed to find more postings
4. When you have collected enough URLs, provide them in a clean bulleted list

Output Format (STRICT):
- https://url1.com
- https://url2.com
- https://url3.com

Requirements:
- Each URL must be a direct job posting link (contains /jobs/, /careers/, /job/, /positions/)
- Exclude company homepages, blog posts, or articles
- Minimum 3 job posting URLs
- Use bullet points with "-" prefix
- No extra text, explanations, or commentary in final output

Begin searching!
"""

# -----------------------------
# Agent node with ReAct reasoning
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
    # Check if agent wants to use tools
    if not last_message.tool_calls:
        return "end"
    return "continue"

# -----------------------------
# Build ReAct graph
# -----------------------------
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", agent)
workflow.add_node("tools", ToolNode(tools))

# Add edges
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
    print("Starting ReAct Agent...\n")
    print("=" * 60)
    
    result = app.invoke(
        {
            "messages": [
                SystemMessage(content=REACT_SYSTEM_PROMPT),
                HumanMessage(content="Find Agentic AI engineer jobs position available only on linkedin")
            ]
        },
        {"recursion_limit": 15}
    )
    
    print("\n" + "=" * 60)
    print("\n=== JOB LINKS ===\n")
    
    # Extract only final answer (last AI message)
    final_msg = result["messages"][-1]
    print(final_msg.content)

if __name__ == "__main__":
    main()