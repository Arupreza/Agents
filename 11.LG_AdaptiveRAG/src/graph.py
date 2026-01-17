from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from src.chains.answer_grader import answer_grader
from src.chains.hallucination_grader import hallucination_grader
from src.chains.router import question_router, RouteQuery
# Correct paths to include the 'src' directory
from src.consts import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH
from src.nodes import retrieve, grade_documents, generate, web_search
from src.state import GraphState

load_dotenv()

def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")

    docs = state.get("documents", [])  # after grading, these should be "relevant only"
    if not docs or len(docs) == 0:
        print("---DECISION: NO RELEVANT DOCS, DO WEB SEARCH---")
        return WEBSEARCH

    print("---DECISION: GENERATE FROM VECTOR DB---")
    return GENERATE

def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("---CHECK HALLUCINATIONS---")

    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    h_score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    grounded = h_score.binary_score
    if grounded:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")

        a_score = answer_grader.invoke(
            {"question": question, "generation": generation}
        )

        answers_question = a_score.binary_score
        if answers_question:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

def route_question(state: GraphState) -> str:
    print("---ROUTE QUESTION---")
    question = state["question"]

    source: RouteQuery = question_router.invoke({"question": question})
    datasource = (source.datasource or "").strip().lower()

    # IMPORTANT: RouteQuery.datasource is Literal["vectorstore", "websearch"]
    if datasource == "websearch":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return WEBSEARCH  # your node name constant
    if datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return RETRIEVE   # your node name constant

    # Default fallback (never return None)
    print(f"---ROUTER WARNING: unexpected datasource={datasource!r} -> default WEBSEARCH---")
    return WEBSEARCH 
    
workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

# Only keep the conditional entry point
workflow.set_conditional_entry_point(
    route_question,
    {
        WEBSEARCH: WEBSEARCH,
        RETRIEVE: RETRIEVE,
    },
)
# Remove: workflow.set_entry_point(RETRIEVE)

workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    },
)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": WEBSEARCH,  # Get new documents instead of looping
        "useful": END,
        "not useful": WEBSEARCH,
    },
)
workflow.add_edge(WEBSEARCH, GENERATE)

app = workflow.compile()