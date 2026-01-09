# LangGraph Job Search Agent

AI-powered job search agent using LangGraph, OpenAI, and Tavily. Finds AI engineer positions with LangChain experience in the Bay Area.

## Architecture

**State Machine Pattern:**
- `agent` node: LLM decides to search or finish
- `tools` node: Executes Tavily web search
- Conditional routing: Loops until job URLs extracted

**vs Legacy LangChain:**
- Replaces deprecated `AgentExecutor`
- Explicit state graph vs implicit chains
- Full control over agent reasoning loop

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- API keys:
  - [OpenAI](https://platform.openai.com/api-keys)
  - [Tavily](https://app.tavily.com/) (1000 free searches/month)
  - [LangSmith](https://smith.langchain.com/) (optional, 5000 free traces/month)

## Setup

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone/create project
mkdir langchain-job-search && cd langchain-job-search

# Initialize uv project
uv init

# Install dependencies
uv add langchain-openai langchain-tavily langgraph python-dotenv pydantic

# Create .env file
cat > .env << 'EOF'
OPENAI_API_KEY=sk-your-key-here
TAVILY_API_KEY=tvly-your-key-here
LANGCHAIN_API_KEY=lsv2_pt_your-key-here  # Optional: for tracing
EOF

# Add your code to main.py
```

## Usage

```bash
# Run agent
uv run python main.py

# View traces (if LangSmith enabled)
# https://smith.langchain.com/o/YOUR_ORG/projects/p/job-search-agent
```

## Configuration

**Modify search query:**
```python
user_msg = HumanMessage(content="Find [your query] jobs in [location]")
```

**Change model:**
```python
llm = ChatOpenAI(model="gpt-4o", temperature=0)  # More capable, higher cost
```

**Adjust search depth:**
```python
search_tool = TavilySearch(max_results=10)  # More results per search
{"recursion_limit": 20}  # Allow more agent loops
```

## Project Structure

```
langchain-job-search/
├── .venv/              # Auto-managed by uv
├── pyproject.toml      # Dependencies
├── uv.lock            # Lock file
├── .env               # API keys (gitignore this)
├── main.py            # Agent code
└── README.md          # This file
```

## How It Works

1. **Initialization:** Agent receives system prompt + user query
2. **Decision:** LLM analyzes task, decides to call `tavily_search` tool
3. **Search:** Tavily executes web search, returns results
4. **Loop:** Agent processes results, may search again for better coverage
5. **Completion:** Agent extracts URLs, returns final list

**State flow:**
```
agent → [has tool_calls?] → tools → agent → [has tool_calls?] → END
```

## LangSmith Tracing

Traces show:
- Agent reasoning per iteration
- Tool inputs/outputs
- Token usage and cost
- Execution latency
- Error diagnostics

**Enable:**
```python
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "job-search-agent"
```

## Costs (Estimates)

- **OpenAI gpt-4o-mini:** ~$0.0001 per run (5-10 tool calls)
- **Tavily:** Free tier sufficient for development
- **LangSmith:** Free tier sufficient for debugging

## Troubleshooting

**`TAVILY_API_KEY` not found:**
```bash
# Verify .env exists and is formatted correctly
cat .env
# Should show: TAVILY_API_KEY=tvly-...
```

**Agent loops infinitely:**
- Check `recursion_limit` setting
- Review LangSmith traces for reasoning errors
- Simplify system prompt

**No results found:**
- Job market may lack LangChain-specific roles
- Broaden search: "AI engineer Python jobs Bay Area"
- Increase `max_results` in TavilySearch

## Migration from LangChain Chains

**Old pattern (deprecated):**
```python
chain = prompt | llm | parser
result = chain.invoke(...)
```

**New pattern (LangGraph):**
```python
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_fn)
app = workflow.compile()
result = app.invoke(...)
```

**Key difference:** Explicit state management and tool execution control.

## References

- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [LangSmith](https://docs.smith.langchain.com/)
- [Tavily API](https://docs.tavily.com/)
- [uv Docs](https://docs.astral.sh/uv/)