import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

# --- FIXED PATH CONFIGURATION ---
# 1. This is: ~/Arupreza/Agents/9.LG_CorrectiveRAG/src/
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Step out of 'src' to: ~/Arupreza/Agents/9.LG_CorrectiveRAG/
PACKAGE_ROOT = os.path.dirname(current_dir)

# 3. Step out of the package to the root: ~/Arupreza/Agents/
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)

# 4. Final Path to your actual data folder
INDEX_PATH = os.path.join(PROJECT_ROOT, "Data", "faiss_index")

# Initialize Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def ingest_and_merge_pdf(file_path: str):
    """Processes a PDF and joins it with the existing FAISS index."""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    # Load and Split
    loader = PyPDFLoader(file_path)
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=120
    ).split_documents(loader.load())
    
    # Load or Create Index
    if os.path.exists(os.path.join(INDEX_PATH, "index.faiss")):
        vectorstore = FAISS.load_local(
            INDEX_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        vectorstore.add_documents(chunks)
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)

    # Ensure the Data directory exists
    os.makedirs(INDEX_PATH, exist_ok=True)
    
    vectorstore.save_local(INDEX_PATH)
    print(f"✅ Successfully added {file_path} to index at {INDEX_PATH}")

def get_retriever(search_type="similarity", k=5):
    """
    Loads the FAISS index and returns a retriever object.
    """
    faiss_file = os.path.join(INDEX_PATH, "index.faiss")
    if not os.path.exists(faiss_file):
        raise FileNotFoundError(
            f"❌ FAISS index not found at {faiss_file}. "
            "You must run ingestion first."
        )

    vectorstore = FAISS.load_local(
        INDEX_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )

    # Returns the LangChain Retriever Object
    return vectorstore.as_retriever(search_kwargs={"k": k}, search_type=search_type)