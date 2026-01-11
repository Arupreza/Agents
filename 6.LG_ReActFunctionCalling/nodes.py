from langgraph.prebuilt import ToolNode
from react import llm_with_tools, tools

def run_agent_with_tools(state):
    # GPT models perform better when the full history is provided
    # The return appends the AIMessage to the state
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools)