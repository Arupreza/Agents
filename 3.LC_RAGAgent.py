import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import create_agent 
from langchain_core.tools import tool

# 1. SETUP & CONFIG
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FAISS_PATH = "Data/faiss_index"

# 2. LOAD VECTOR STORE
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.load_local(
    FAISS_PATH, 
    embeddings, 
    allow_dangerous_deserialization=True
)

# 3. DEFINE RAG TOOL
@tool
def search_papers(query: str) -> str:
    """Search Reza's research papers for information on CAN bus or security."""
    results = vectorstore.similarity_search(query, k=3)
    if not results:
        return "No relevant information found."
    
    return "\n\n".join([
        f"[Source] {doc.metadata.get('source')} (Page {doc.metadata.get('page')})\n{doc.page_content}"
        for doc in results
    ])

# 4. CREATE MODERN AGENT
# In 2026, we pass the system instructions via the 'system_prompt' keyword.
llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", google_api_key=GEMINI_API_KEY)
tools = [search_papers]

instructions = (
    "You are a research assistant. Use the 'search_papers' tool to find info. "
    "Always cite your sources. If you don't know, say you don't know."
)

# FIX: Use 'system_prompt' instead of 'prompt'
app = create_agent(llm, tools=tools, system_prompt=instructions)

# 5. INTERACTIVE LOOP
def main():
    print("🤖 RAG Agent Ready (2026 Standard)")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['exit', 'quit']: break
        if not user_input: continue

        try:
            # Modern agents expect input in a "messages" list
            result = app.invoke({"messages": [("user", user_input)]})
            # The agent returns a list of messages; the last one is the response
            print(f"\nAssistant: {result['messages'][-1].content}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")

if __name__ == "__main__":
    main()