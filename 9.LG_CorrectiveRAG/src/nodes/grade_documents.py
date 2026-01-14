from typing import Any, Dict
# FIX: Use the 'src.' prefix for internal imports
from src.chains.retrieval_grader import retrieval_grader
from src.state import GraphState

def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question.
    """

    print("---NODE: CHECK DOCUMENT RELEVANCE---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False
    
    for d in documents:
        # Invoking the grader chain
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # If even one document is irrelevant, we flag for web search
            web_search = True
    
    # Return only the keys that need updating
    return {"documents": filtered_docs, "web_search": web_search}