import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI # Changed from Google
from langchain_tavily import TavilySearch
from langchain_core.tools import tool

load_dotenv()

@tool
def fahrenheit_converter(celsius: float) -> float:
    """Converts Celsius to Fahrenheit."""
    return (celsius * 9/5) + 32

# 1. Initialize Tools
tools = [TavilySearch(max_results=1), fahrenheit_converter]

# 2. Initialize GPT Model
# gpt-4o-mini is cost-effective and excellent at tool calling
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 3. Bind Tools
llm_with_tools = llm.bind_tools(tools)