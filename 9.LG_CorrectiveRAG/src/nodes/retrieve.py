from typing import Any, Dict
from src.state import GraphState
from src.ingestion import get_retriever

def retrieve(state: GraphState) -> Dict[str, Any]:
    """
    Retrieves documents from the FAISS index based on the user question.
    """
    print("---NODE: RETRIEVING RELEVANT DOCUMENTS---")
    
    # 1. Extract the question from the current graph state
    question = state["question"]
    
    # 2. Initialize the Retriever object
    # This calls the factory function in src/ingestion.py
    retriever_obj = get_retriever()
    
    # 3. Execute the retrieval
    # retriever_obj is a LangChain Runnable, so .invoke() is the correct method
    documents = retriever_obj.invoke(question)
    
    # 4. Return the new data to update the State
    # We do not need to return "question" again unless we modified it
    return {"documents": documents}