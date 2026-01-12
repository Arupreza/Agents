import os
from typing import Literal
from dotenv import load_dotenv

# LangChain / OpenAI Imports
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
# LangGraph Imports
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()

# --- 1. TOOLS DEFINITION ---

@tool
def fahrenheit_converter(celsius: float) -> float:
    """
    Converts a temperature from Celsius to Fahrenheit. 
    Use this when you have a temperature in Celsius and need Fahrenheit.
    """
    return (celsius * 9/5) + 32

# Initialize search and combine tools
search_tool = TavilySearch(max_results=1)
tools = [search_tool, fahrenheit_converter]

# --- 2. LLM SETUP & BINDING ---

# We use gpt-4o-mini for fast, accurate tool calling
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
#llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", google_api_key=os.getenv("GEMINI_API_KEY"))
# CRITICAL: This allows the LLM to "see" and use your tools
llm_with_tools = llm.bind_tools(tools)

# --- 3. GRAPH NODES ---

def run_agent_with_tools(state: MessagesState):
    """Reasoning node: Enforces specific tool usage for skill development."""
    
    # Updated System Message
    sys_msg = SystemMessage(content=(
        "You are a helpful assistant. "
        "1. Use search for real-time weather data. "
        "2. ALWAYS convert any Celsius temperature found to Fahrenheit "
        "using the fahrenheit_converter tool before giving the final answer."
        "In final answer the temperature will be is both in Celsius and Fahrenheit."
    ))
    
    messages = [sys_msg] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Prebuilt node that automatically handles tool execution
tool_node = ToolNode(tools)

# --- 4. CONDITIONAL LOGIC ---
# --- CONDITIONAL ROUTER ---
# This function acts as the "Decision Gate" for the graph.
# 1. It inspects the last message in the conversation state.
# 2. 'isinstance(last_message, AIMessage)' ensures the message came from the LLM.
# 3. 'last_message.tool_calls' checks if the LLM generated a request to use a tool.
# 4. Returns "ACT" to trigger the 'act' node (ToolNode) if tools are needed.
# 5. Returns "END" if the LLM has provided a final text answer without tool requests.

def should_continue(state: MessagesState) -> Literal["ACT", "END"]:
    """Routes the flow based on whether the LLM wants to use a tool."""
    last_message = state["messages"][-1]
    
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "ACT"
    return "END"

# --- 5. GRAPH CONSTRUCTION ---

# [ START: app.invoke() ]
#                   |
#                   v
#        +----------------------+
#        |  agent_reason (Node) | <--- LLM: "Should I use a tool or answer?"
#        +----------------------+
#                   |
#                   v
#        /----------------------\
#       |    should_continue?    | <--- Router (Conditional Edge)
#        \----------------------/
#           |                |
#           | "ACT"          | "END"
#           v                v
#   +----------------+    +------------+
#   |   act (Node)   |    |    END     | <--- Final result returned to user
#   | (ToolNode)     |    +------------+
#   +----------------+
#           |
#           +---- (Loop Back) ----> [ agent_reason ]

workflow = StateGraph(MessagesState)

# Define the two main nodes
workflow.add_node("agent_reason", run_agent_with_tools)
workflow.add_node("act", tool_node)

# Entry point
workflow.set_entry_point("agent_reason")

# Conditional path: Reason -> (Act OR End)
workflow.add_conditional_edges(
    "agent_reason",
    should_continue,
    {
        "ACT": "act",
        "END": END,
    }
)

# Loop: After acting, always return to agent to interpret results
workflow.add_edge("act", "agent_reason")

app = workflow.compile()

# --- 6. EXECUTION ---

if __name__ == "__main__":
    # Optional: Save visualization
    try:
        with open("graph.png", "wb") as f:
            f.write(app.get_graph().draw_mermaid_png())
        print("✅ Graph saved as graph.png")
    except:
        pass

    print("🤖 GPT ReAct Agent Ready (Type 'exit' to quit)")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break
        if not user_input:
            continue

        # Invoke the graph with the user's message
        final_state = app.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config={"recursion_limit": 10}
        )

        # The final answer is the last message in the list
        print(f"\nAssistant: {final_state['messages'][-1].content}\n")