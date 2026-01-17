import os
import sys
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_postgres.vectorstores import PGVector

# 1. Environment & Setup
load_dotenv()

EMBED_MODEL = "text-embedding-3-small"
PAPERS_PATH = "/home/lisa/Arupreza/Agentic-RAG-Tutorial/Data/RezaPapers"

def get_connection_string(prefix):
    """Constructs and validates the SQLAlchemy connection string."""
    keys = [f"{prefix}_USER", f"{prefix}_PASS", f"{prefix}_HOST", f"{prefix}_PORT", f"{prefix}_DB"]
    env_vars = {k: os.getenv(k) for k in keys}
    
    missing = [k for k, v in env_vars.items() if v is None]
    if missing:
        print(f"CRITICAL ERROR: Missing environment variables: {missing}")
        sys.exit(1)

    return f"postgresql+psycopg://{env_vars[f'{prefix}_USER']}:{env_vars[f'{prefix}_PASS']}@{env_vars[f'{prefix}_HOST']}:{env_vars[f'{prefix}_PORT']}/{env_vars[f'{prefix}_DB']}"

def initialize_context_db():
    # 1. Load Documents
    print(f"--- Step 1: Loading papers from {PAPERS_PATH} ---")
    if not os.path.exists(PAPERS_PATH):
        print(f"Error: Path {PAPERS_PATH} does not exist.")
        return None

    loader = PyPDFDirectoryLoader(PAPERS_PATH)
    docs = loader.load()
    print(f"Success: Loaded {len(docs)} document pages.")
    
    # 2. Split and Sanitize
    print("--- Step 2: Splitting and Sanitizing Text ---")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # CRITICAL: Remove NULL (0x00) bytes that crash PostgreSQL
    for doc in splits:
        doc.page_content = doc.page_content.replace("\x00", "")
    
    print(f"Prepared {len(splits)} sanitized text chunks.")
    
    # 3. Initialize Vector Store
    print("--- Step 3: Generating Embeddings and Storing in PGVector ---")
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    connection = get_connection_string("CTX")
    
    vectorstore = PGVector.from_documents(
        embedding=embeddings,
        documents=splits,
        collection_name="research_papers",
        connection=connection,
        use_jsonb=True,
    )
    
# 4. Apply HNSW Optimization
    print("--- Step 4: Applying HNSW Index for High-Speed Retrieval ---")
    # In langchain_postgres, we access the engine through ._engine
    try:
        from sqlalchemy import text
        
        with vectorstore._engine.begin() as conn:
            # Drop old index if exists to avoid conflicts
            conn.execute(text("DROP INDEX IF EXISTS research_hnsw_idx;"))
            
            # Create HNSW Index
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS research_hnsw_idx 
                ON langchain_pg_embedding 
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
            """))
            print("HNSW Index successfully applied to ContextDB.")
    except Exception as index_error:
        print(f"Warning: Could not create HNSW index automatically: {index_error}")
        print("The data is stored, but search may be slower.")

if __name__ == "__main__":
    try:
        context_vdb = initialize_context_db()
        print("\n[COMPLETE] End-to-end processing finished successfully.")
    except Exception as e:
        print(f"\n[FAILED] An error occurred: {e}")