from typing import Any, Dict
from src.chains.retrieval_grader import retrieval_grader
from src.state import GraphState

def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Filter retrieved documents by relevance.
    Web-search should be triggered ONLY if no relevant documents remain.
    """
    print("---NODE: CHECK DOCUMENT RELEVANCE---")

    question = state["question"]
    documents = state.get("documents") or []

    if not documents:
        print("---GRADE: NO DOCUMENTS RETRIEVED -> WEB SEARCH---")
        return {"documents": [], "web_search": True}

    filtered_docs = []

    for i, d in enumerate(documents, 1):
        # Defensive: some loaders may produce None/empty page_content
        content = (getattr(d, "page_content", "") or "").strip()
        if not content:
            print(f"---GRADE: DOC {i} EMPTY -> SKIP---")
            continue

        score = retrieval_grader.invoke({"question": question, "document": content})
        grade = (getattr(score, "binary_score", "") or "").strip().lower()

        if grade == "yes":
            print(f"---GRADE: DOC {i} RELEVANT---")
            filtered_docs.append(d)
        else:
            print(f"---GRADE: DOC {i} NOT RELEVANT---")

    # Web-search ONLY if none are relevant
    web_search = (len(filtered_docs) == 0)

    if web_search:
        print("---DECISION: 0 RELEVANT DOCS -> WEB SEARCH---")
    else:
        print(f"---DECISION: {len(filtered_docs)} RELEVANT DOCS -> GENERATE---")

    return {"documents": filtered_docs, "web_search": web_search}