"""
TUTORIAL: Building an Optimized HNSW Vector Store with FAISS and LangChain
Goal: Index PDFs with high-performance search using Hierarchical Navigable Small World (HNSW).
"""

import os
import faiss
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
# We load API keys and define hyperparameters for chunking and indexing.
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found. Check your .env file.")

DATA_DIR = "Data/RezaPapers"  # Directory containing PDFs
SAVE_PATH = "Data/faiss_index" # Folder where the index will be saved
CHUNK_SIZE = 800               # Max characters per chunk
CHUNK_OVERLAP = 120            # Context overlap
EMBED_MODEL = "text-embedding-3-small" # Dimension: 1536

# ==========================================
# 2. DOCUMENT PROCESSING
# ==========================================
print("🔎 Step 1: Loading and Splitting Documents...")

# DirectoryLoader crawls the folder for all PDFs
loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

if not documents:
    raise ValueError(f"❌ No PDFs found in {DATA_DIR}")

# Recursive splitting maintains paragraph/sentence integrity better than character splitting
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)
chunks = splitter.split_documents(documents)
print(f"✅ Created {len(chunks)} text chunks.")

# ==========================================
# 3. HNSW INDEX CONSTRUCTION
# ==========================================
print("🔎 Step 2: Configuring FAISS HNSW Index...")

# Initialize the embedding model
embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

# Define HNSW parameters
# M: Number of bi-directional links per node. Higher = more accurate/more memory.
# efConstruction: Accuracy vs Speed tradeoff during index building.
dim = 1536  # Dimension for text-embedding-3-small
M = 64
ef_construction = 200

# Create the low-level FAISS index
index = faiss.IndexHNSWFlat(dim, M)
index.hnsw.efConstruction = ef_construction

# Wrap the FAISS index into LangChain's VectorStore
vectorstore = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

# Add documents (This step triggers the embedding generation via OpenAI)
vectorstore.add_documents(chunks)
print("✅ HNSW Index successfully built.")

# ==========================================
# 4. PERSISTENCE (SAVE/LOAD)
# ==========================================
print("🔎 Step 3: Saving Index to Disk...")
Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
vectorstore.save_local(SAVE_PATH)

# Reloading ensures the index is persistent and valid
new_db = FAISS.load_local(
    SAVE_PATH, 
    embeddings, 
    allow_dangerous_deserialization=True # Required for loading pickle files
)
print("✅ Index saved and reloaded successfully.")

# ==========================================
# 5. SIMILARITY SEARCH (TESTING)
# ==========================================
print("🔎 Step 4: Performing Query...")

query = "intrusion detection in CAN bus"
# Search for top 3 matches
results = new_db.similarity_search(query, k=3)

for i, res in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(f"Source: {res.metadata.get('source')}")
    print(f"Content: {res.page_content[:200]}...")

print("\n🚀 Tutorial Complete.")