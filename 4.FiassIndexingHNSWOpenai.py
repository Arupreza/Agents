"""
TUTORIAL: Batch Indexing with FAISS HNSW
Goal: Efficiently index large PDF datasets by processing chunks in manageable batches.
This approach prevents API timeouts and stays within OpenAI's Rate Limits.
"""

import os
import faiss
import time
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.docstore.in_memory import InMemoryDocstore

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found.")

DATA_DIR = "Data/RezaPapers"
SAVE_PATH = "Data/faiss_index"
BATCH_SIZE = 100  # Number of chunks processed per API call
EMBED_MODEL = "text-embedding-3-small"

# ==========================================
# 2. DOCUMENT PROCESSING
# ==========================================
print("🔎 Step 1: Loading and Splitting Documents...")

loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
chunks = splitter.split_documents(documents)
print(f"✅ Total chunks created: {len(chunks)}")

# ==========================================
# 3. BATCHED HNSW CONSTRUCTION
# ==========================================
print(f"🔎 Step 2: Building Index in Batches of {BATCH_SIZE}...")

# 1. Initialize Embeddings and empty FAISS Index
embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
dim = 1536 
index = faiss.IndexHNSWFlat(dim, 64)
index.hnsw.efConstruction = 200

# 2. Create the VectorStore container (initially empty)
vectorstore = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

# 

# 3. The Batch Loop
# We iterate through the chunks list using the BATCH_SIZE as our step
for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i : i + BATCH_SIZE]
    print(f"🚀 Indexing batch {i//BATCH_SIZE + 1}/{(len(chunks)-1)//BATCH_SIZE + 1}...")
    
    # Each call here generates embeddings for only the current batch
    vectorstore.add_documents(batch)
    
    # Small pause to avoid hitting Rate Limits (Tokens Per Minute)
    time.sleep(0.5) 

print("✅ All batches indexed successfully.")

# ==========================================
# 4. PERSISTENCE & TESTING
# ==========================================
print("🔎 Step 3: Saving and Testing...")

Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
vectorstore.save_local(SAVE_PATH)

# Reload to verify integrity
new_db = FAISS.load_local(SAVE_PATH, embeddings, allow_dangerous_deserialization=True)

query = "intrusion detection in CAN bus"
results = new_db.similarity_search(query, k=3)

for i, res in enumerate(results):
    print(f"\n[Result {i+1}] Source: {res.metadata.get('source')}")
    print(f"Snippet: {res.page_content[:150]}...")

print("\n🚀 Batch Indexing Tutorial Complete.")