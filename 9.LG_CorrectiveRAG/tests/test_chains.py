from dotenv import load_dotenv
load_dotenv()

from chains.retrieval_grader import retrieval_grader, GradingDocuments
from ingestion import retriever  # This is the function
from chains.generation import generation_chain
from pprint import pprint

# def test_retrival_grader_answer_yes() -> None:
#     question = "IDS"
    
#     # STEP 1: Call the function to get the actual Retriever Object
#     retriever_obj = retriever() 
    
#     # STEP 2: Now call .invoke() on the object
#     docs = retriever_obj.invoke(question)
#     doc_txt = docs[0].page_content # Use index 0 to be safe

#     res: GradingDocuments = retrieval_grader.invoke(
#         {"question": question, "document": doc_txt}
#     )

#     assert res.binary_score == "yes"

# def test_retrival_grader_answer_no() -> None:
#     question = "IDS"
    
#     # STEP 1: Call the function
#     retriever_obj = retriever() 
    
#     # STEP 2: Use the object
#     docs = retriever_obj.invoke(question)
#     doc_txt = docs[0].page_content

#     res: GradingDocuments = retrieval_grader.invoke(
#         {"question": "how to make IDS", "document": doc_txt}
#     )

#     assert res.binary_score == "no"

def test_generation_chain() -> None:
    question = "What is IVN"
    retriever_obj = retriever()
    docs = retriever_obj.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    pprint(generation)