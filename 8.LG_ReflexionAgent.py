"""
END-TO-END LangGraph Agent (Draft -> Search -> Revise -> (loop search/revise) -> End)

This version INCLUDES:
- JsonOutputToolsParser (for JSON tool-call extraction)
- PydanticToolsParser (to parse tool-call args into Pydantic objects)

Main flow is NOT changed:
- draft node returns the AIMessage from the LLM
- execute_tools ToolNode runs run_queries based on the tool call name
- revise node returns the AIMessage from the LLM
- event_loop still stops after 2 ToolMessages
"""

import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List

load_dotenv()

from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.messages import ToolMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, START, StateGraph, MessagesState


# -----------------------------
# 1) Schema Definitions
# -----------------------------
class Reflection(BaseModel):
    missing: str = Field(..., description="What important information is missing.")
    superfluous: str = Field(..., description="What information is unnecessary.")


class AnswerQuestion(BaseModel):
    """Answer The Question"""
    answer: str = Field(..., description="Detailed answer to the question.")
    reflection: Reflection = Field(..., description="Reflection on the answer.")
    search_queries: List[str] = Field(..., description="1-3 search queries for research.")


class ReviseAnswer(AnswerQuestion):
    """Revise your original answer."""
    references: List[str] = Field(..., description="Citations motivating your updated answer.")


# -----------------------------
# 2) Prompts
# -----------------------------
actor_prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are expert researcher. Current time: {time}
1. {first_instruction}
2. Reflect and critique your answer.
3. Recommend search queries to research information."""),
    MessagesPlaceholder(variable_name="messages"),
]).partial(time=lambda: datetime.datetime.now().isoformat())

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed answer."
)

revise_instruction = """Revise your previous answer. Use numerical citations [1], [2].
Keep the answer under 100 words. Add a References section at the bottom."""

revisor_prompt_template = actor_prompt_template.partial(
    first_instruction=revise_instruction
)


# -----------------------------
# 3) LLM & Parsers
# -----------------------------
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Parsers (NOW USED)
parser_json = JsonOutputToolsParser(return_id=True)
parser_answer = PydanticToolsParser(tools=[AnswerQuestion])
parser_revise = PydanticToolsParser(tools=[ReviseAnswer])

# LLM chains
first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion],
    tool_choice="AnswerQuestion",
)

revisor = revisor_prompt_template | llm.bind_tools(
    tools=[ReviseAnswer],
    tool_choice="ReviseAnswer",
)


# -----------------------------
# 4) Search Tool
# -----------------------------
tavily_tool = TavilySearchResults(max_results=5)

def run_queries(search_queries: list[str], **kwargs):
    """Executes Tavily searches for each query. **kwargs absorbs extra tool-call args."""
    return tavily_tool.batch([{"query": q} for q in search_queries])


# ToolNode: executes tools by matching tool-call name
execute_tools = ToolNode([
    StructuredTool.from_function(
        func=run_queries,
        name="AnswerQuestion",
        description="Search for missing information",
    ),
    StructuredTool.from_function(
        func=run_queries,
        name="ReviseAnswer",
        description="Search for citations",
    )
])


# -----------------------------
# 5) Nodes & Logic
# -----------------------------
def draft_node(state: MessagesState):
    response = first_responder.invoke({"messages": state["messages"]})

    # Use parsers (for learning/debug) without changing main flow
    try:
        tool_calls_json = parser_json.invoke(response)
        tool_calls_obj = parser_answer.invoke(response)  # list of AnswerQuestion objects
        print("\n[DRAFT] Parsed tool calls (JSON):", tool_calls_json)
        if tool_calls_obj:
            print("[DRAFT] Parsed Pydantic object:", tool_calls_obj[0])
    except Exception as e:
        print("\n[DRAFT] Parser error:", repr(e))

    # Main flow unchanged: return the AIMessage
    return {"messages": [response]}


def revise_node(state: MessagesState):
    response = revisor.invoke({"messages": state["messages"]})

    # Use parsers (for learning/debug) without changing main flow
    try:
        tool_calls_json = parser_json.invoke(response)
        tool_calls_obj = parser_revise.invoke(response)  # list of ReviseAnswer objects
        print("\n[REVISE] Parsed tool calls (JSON):", tool_calls_json)
        if tool_calls_obj:
            print("[REVISE] Parsed Pydantic object:", tool_calls_obj[0])
    except Exception as e:
        print("\n[REVISE] Parser error:", repr(e))

    return {"messages": [response]}


def event_loop(state: MessagesState):
    """Stop after two ToolMessages have been produced."""
    count_tool_visit = sum(isinstance(m, ToolMessage) for m in state["messages"])
    if count_tool_visit >= 2:
        return END
    return "execute_tools"

# +---------------------------+
# |           START           |
# +---------------------------+
#               |
#               v
# +---------------------------+
# |           draft           |
# |  (draft_node)             |
# |  - first_responder.invoke |
# |  - (parse tool call JSON) |
# |  - (parse to AnswerQuestion)
# +---------------------------+
#               |
#               v
# +---------------------------+
# |        execute_tools       |
# |   (ToolNode)               |
# |   - reads last AI tool call |
# |   - calls run_queries(...)  |
# |   - returns ToolMessage     |
# +---------------------------+
#               |
#               v
# +---------------------------+
# |           revise          |
# |  (revise_node)            |
# |  - revisor.invoke         |
# |  - (parse tool call JSON) |
# |  - (parse to ReviseAnswer)|
# +---------------------------+
#               |
#               v
# +-------------------------------+
# |         event_loop            |
# | count ToolMessages in state   |
# |  - if >= 2  -> END            |
# |  - else    -> execute_tools   |
# +-------------------------------+
#         |                  |
#         | (>=2)            | (<2)
#         v                  |
# +------------------+       |
# |       END        |<------+
# +------------------+


# -----------------------------
# 6) Graph Construction
# -----------------------------
builder = StateGraph(MessagesState)
builder.add_node("draft", draft_node)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revise_node)

builder.add_edge(START, "draft")
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")
builder.add_conditional_edges("revise", event_loop)

graph = builder.compile()


# -----------------------------
# 7) Execution
# -----------------------------
if __name__ == "__main__":
    inputs = {
        "messages": [
            HumanMessage(content="Write about AI-Powered SOC startups that raised capital.")
        ]
    }
    response = graph.invoke(inputs)

    print("\n================ FINAL MESSAGES ================\n")
    for m in response["messages"]:
        m.pretty_print()