from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# 1. Initialize the LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# 2. Define the Schema
class GradeRetrievedDocuments(BaseModel): # Renamed to match the variable below
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(..., description="Documents are relevant to the question, 'yes' or 'no'")

# 3. Create Structured LLM
# We use method="function_calling" as gpt-3.5-turbo does not support json_schema
structured_llm_grader = llm.with_structured_output(GradeRetrievedDocuments, method="function_calling")

# 4. Define Prompts
system_prompt = """You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

# 5. Construct the Chain
retrieval_grader = grade_prompt | structured_llm_grader