from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

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

workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    },
)
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()