import os
import sys
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings # Corrected Import
from langchain_postgres.vectorstores import PGVector
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# 1. SETUP & CONFIG
load_dotenv()

def get_connection_string():
    """Constructs the SQLAlchemy connection string for pgvector."""
    user = os.getenv("CTX_USER")
    password = os.getenv("CTX_PASS")
    host = os.getenv("CTX_HOST")
    port = os.getenv("CTX_PORT")
    db = os.getenv("CTX_DB")
    
    if not all([user, password, host, port, db]):
        print("CRITICAL ERROR: Missing CTX environment variables in .env")
        sys.exit(1)
        
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{db}"

# 2. INITIALIZE PGVECTOR STORE
# Using text-embedding-3-small (1536 dims) matches your existing HNSW index
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = PGVector(
    embeddings=embeddings,
    collection_name="research_papers",
    connection=get_connection_string(),
    use_jsonb=True,
)

# 3. DEFINE THE RAG TOOL
@tool
def search_papers(query: str) -> str:
    """Search Reza's research papers for technical info on CAN bus or security."""
    # O(log N) search speed powered by your HNSW index
    results = vectorstore.similarity_search(query, k=3)
    if not results:
        return "No relevant information found in the context database."
    
    return "\n\n".join([
        f"[Source] {doc.metadata.get('source')} (Page {doc.metadata.get('page', 'N/A')})\n{doc.page_content}"
        for doc in results
    ])

# 4. CREATE THE AGENT (Updated to OpenAI)
# temperature=0 ensures precise, non-hallucinated academic answers
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
tools = [search_papers]

instructions = (
    "You are a specialized academic research assistant. Use the 'search_papers' tool "
    "to provide accurate information based on the provided papers. Always include "
    "specific citations (filename and page number)."
)

# Using the modern 2026 'prompt' parameter
app = create_react_agent(llm, tools=tools, prompt=instructions)

# 5. EXECUTION LOOP
def main():
    print("🤖 Research Agent (GPT-3.5 + pgvector + HNSW) Ready")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['exit', 'quit']: break
        if not user_input: continue

        try:
            result = app.invoke({"messages": [("user", user_input)]})
            print(f"\nAssistant: {result['messages'][-1].content}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")

if __name__ == "__main__":
    main()