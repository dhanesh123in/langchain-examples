import operator
from typing_extensions import TypedDict
from typing import List, Annotated, Callable
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.schema import Document
from langgraph.graph import END, StateGraph
import json

class GraphState(TypedDict):
    """Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node."""
    question: str  # User question
    generation: str  # LLM generation
    web_search: str  # Binary decision to run web search
    max_retries: int  # Max number of retries for answer generation
    answers: int  # Number of answers generated
    loop_step: Annotated[int, operator.add]
    documents: List[Document]  # List of retrieved documents

def create_workflow_graph(
    llm: Callable,
    llm_json_mode: Callable,
    retriever: Callable,
    web_search_tool: Callable,
    prompts: dict
) -> StateGraph:
    """Create the workflow graph with the provided components."""
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def route_question(state: GraphState) -> str:
        """Route question to web search or RAG."""
        print("---ROUTE QUESTION---")
        route_question = llm_json_mode.invoke(
            [SystemMessage(content=prompts["router_instructions"])]
            + [HumanMessage(content=state["question"])]
        )
        source = json.loads(route_question.content)["datasource"]
        if source == "websearch":
            print("---ROUTE QUESTION TO WEB SEARCH---")
            return "websearch"
        elif source == "vectorstore":
            print("---ROUTE QUESTION TO RAG---")
            return "vectorstore"


    def retrieve(state: GraphState) -> dict:
        """Retrieve documents from vectorstore."""
        print("---RETRIEVE---")
        question = state["question"]
        documents = retriever.invoke(question)
        return {"documents": documents}


    def grade_documents(state: GraphState) -> dict:
        """Grade documents for relevance."""
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        filtered_docs = []
        web_search = "No"
        for d in documents:
            doc_grader_prompt_formatted = prompts["doc_grader_prompt"].format(
                document=d.page_content, question=question
            )
            result = llm_json_mode.invoke(
                [SystemMessage(content=prompts["doc_grader_instructions"])]
                + [HumanMessage(content=doc_grader_prompt_formatted)]
            )
            grade = json.loads(result.content)["binary_score"]
            if grade.lower() == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                web_search = "Yes"
                continue
        return {"documents": filtered_docs, "web_search": web_search}


    def web_search(state: GraphState) -> dict:
        """Perform web search."""
        print("---WEB SEARCH---")
        question = state["question"]
        documents = state.get("documents", [])

        docs = web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        documents.append(web_results)
        return {"documents": documents}


    def generate(state: GraphState) -> dict:
        """Generate answer using RAG on retrieved documents."""
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        loop_step = state.get("loop_step", 0)

        docs_txt = format_docs(documents)
        rag_prompt_formatted = prompts["rag_prompt"].format(context=docs_txt, question=question)
        generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
        return {"generation": generation, "loop_step": loop_step + 1}


    def decide_to_generate(state: GraphState) -> str:
        """Decide whether to generate or add web search."""
        print("---ASSESS GRADED DOCUMENTS---")
        web_search = state["web_search"]
        if web_search == "Yes":
            print("---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
            return "websearch"
        else:
            print("---DECISION: GENERATE---")
            return "generate"

    def grade_generation(state: GraphState) -> str:
        """Grade generation for hallucinations and question answering."""
        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        max_retries = state.get("max_retries", 3)

        hallucination_grader_prompt_formatted = prompts["hallucination_grader_prompt"].format(
            documents=format_docs(documents), generation=generation.content
        )
        result = llm_json_mode.invoke(
            [SystemMessage(content=prompts["hallucination_grader_instructions"])]
            + [HumanMessage(content=hallucination_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)["binary_score"]

        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            print("---GRADE GENERATION vs QUESTION---")
            answer_grader_prompt_formatted = prompts["answer_grader_prompt"].format(
                question=question, generation=generation.content
            )
            result = llm_json_mode.invoke(
                [SystemMessage(content=prompts["answer_grader_instructions"])]
                + [HumanMessage(content=answer_grader_prompt_formatted)]
            )
            grade = json.loads(result.content)["binary_score"]
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            elif state["loop_step"] <= max_retries:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
            else:
                print("---DECISION: MAX RETRIES REACHED---")
                return "max retries"
        elif state["loop_step"] <= max_retries:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"
        else:
            print("---DECISION: MAX RETRIES REACHED---")
            return "max retries"

    # Create workflow graph
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("websearch", web_search)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)

    # Build graph
    workflow.set_conditional_entry_point(
        route_question,
        {
            "websearch": "websearch",
            "vectorstore": "retrieve",
        },
    )
    workflow.add_edge("websearch", "generate")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "websearch": "websearch",
            "generate": "generate",
        },
    )
    workflow.add_conditional_edges(
        "generate",
        grade_generation,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "websearch",
            "max retries": END,
        },
    )

    return workflow.compile()

