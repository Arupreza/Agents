import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
load_dotenv()
# Use your actual absolute path
INDEX_PATH = "/home/lisa/Arupreza/Agents/Data/faiss_index"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def ingest_and_merge_pdf(file_path: str):
    """Processes a PDF and joins it with the existing FAISS index."""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    # Load and Split
    loader = PyPDFLoader(file_path)
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120).split_documents(loader.load())
    
    # Load or Create Index
    if os.path.exists(os.path.join(INDEX_PATH, "index.faiss")):
        vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        vectorstore.add_documents(chunks)
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(INDEX_PATH)
    print(f"✅ Successfully added {file_path} to index.")

def retriever(search_type="similarity", k=5):
    """
    Loads the FAISS index and returns a retriever object.
    """
    # 1. Strict Path Check
    faiss_file = os.path.join(INDEX_PATH, "index.faiss")
    if not os.path.exists(faiss_file):
        raise FileNotFoundError(
            f"❌ FAISS index not found at {faiss_file}. "
            "You must run ingestion first."
        )

    # 2. Load the local index
    vectorstore = FAISS.load_local(
        INDEX_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )

    # 3. Return the instantiated object
    return vectorstore.as_retriever(
        search_kwargs={"k": k}, 
        search_type=search_type
    )