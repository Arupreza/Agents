import datetime
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Annotated, TypedDict
load_dotenv()

from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, START, StateGraph, MessagesState

# 1. Schema Definitions
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

# 2. Prompts
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

revisor_prompt_template = actor_prompt_template.partial(first_instruction=revise_instruction)

# 3. LLM & Tools
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

first_responder = first_responder_prompt_template | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
revisor = revisor_prompt_template | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")

tavily_tool = TavilySearchResults(max_results=5)

def run_queries(search_queries: list[str], **kwargs): # Added **kwargs to handle extra fields
    return tavily_tool.batch([{"query": query} for query in search_queries])

execute_tools = ToolNode([
    StructuredTool.from_function(
        func=run_queries, 
        name="AnswerQuestion",
        description="Search for missing information"
    ),
    StructuredTool.from_function(
        func=run_queries, 
        name="ReviseAnswer",
        description="Search for citations"
    )
])

# 4. Nodes & Logic
def draft_node(state: MessagesState):
    response = first_responder.invoke({"messages": state["messages"]})
    return {"messages": [response]}

def revise_node(state: MessagesState):
    response = revisor.invoke({"messages": state["messages"]})
    return {"messages": [response]}

def event_loop(state: MessagesState):
    # Count how many ToolMessages exist to limit iterations
    count_tool_visit = sum(isinstance(item, ToolMessage) for item in state["messages"])
    if count_tool_visit >= 2:
        return END
    return "execute_tools"

# +-----------+
#        |   START   |
#        +-----------+
#              |
#              v
#        +-----------+
#        |   draft   | (first_responder)
#        +-----------+
#              |
#              v
#        +---------------+
#        | execute_tools | (run_queries)
#        +---------------+
#              |
#              v
#        +-----------+
#        |   revise  | (revisor)
#        +-----------+
#              |
#              +-----------------------+
#              |                       |
#       [ event_loop ]                 |
#       (Check Count)                  |
#              |                       |
#       +------+-------+               |
#       |              |               |
#       v              v               |
#    [ END ]    [ execute_tools ] <----+
# (If count >= 2) (If count < 2)

# 5. Graph Construction
builder = StateGraph(MessagesState)
builder.add_node("draft", draft_node)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revise_node)

builder.add_edge(START, "draft")
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")
builder.add_conditional_edges("revise", event_loop)

graph = builder.compile()

# 6. Execution
if __name__ == "__main__":
    inputs = {"messages": [HumanMessage(content="Write about AI-Powered SOC startups that raised capital.")]}
    response = graph.invoke(inputs)
    
    for m in response["messages"]:
        m.pretty_print()