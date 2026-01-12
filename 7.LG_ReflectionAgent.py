from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os

load_dotenv()

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet. "
            "Always provide detailed recommendations, including requests for length, virality, style, etc."
        ), 
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts. "
            "Generate the best twitter post possible for the user's request. "
            "If the user provides critique, respond with a revised version of your previous attempts." 
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm


class MessgeGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


REFLECT = 'reflect'
GENERATE = 'generate'

# 1. Define Nodes First
class MessgeGraph(TypedDict):
    # Annotated tells LangGraph to use add_messages to APPEND rather than OVERWRITE
    messages: Annotated[list[BaseMessage], add_messages]

def generation_node(state: MessgeGraph):
    response = generation_chain.invoke({"messages": state["messages"]})
    # You MUST return a list here because 'messages' is defined as a list
    return {"messages": [response]}

def reflection_node(state: MessgeGraph):
    response = reflection_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=response.content)]}


builder = StateGraph(state_schema=MessgeGraph)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)  
builder.set_entry_point(GENERATE)


def should_continue(state: MessgeGraph):
    if len(state["messages"]) >= 6:
        return END
    return REFLECT


builder.add_conditional_edges(GENERATE, should_continue)
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()




if __name__ == "__main__":
    print("Graph compiled successfully.")
    
    inputs = {"messages": [HumanMessage(content="Write a tweet about LangGraph.")]}

    for event in graph.stream(inputs):
        for node_name, value in event.items():
            print(f"--- Node: {node_name} ---")
            # Now value["messages"] is ALWAYS a list
            print(value["messages"][-1].content)
            print("-" * 30)