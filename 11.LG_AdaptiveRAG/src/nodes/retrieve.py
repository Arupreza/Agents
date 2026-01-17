# 9.LG_CorrectiveRAG/src/nodes/retrieve.py

from typing import Any, Dict
from src.state import GraphState
# This MUST match the 'def' name in ingestion.py
from src.ingestion import get_retriever 

def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---NODE: RETRIEVING---")
    question = state["question"]
    
    # Initialize the object
    retriever_obj = get_retriever()
    
    # Call invoke
    documents = retriever_obj.invoke(question)
    
    return {"documents": documents}