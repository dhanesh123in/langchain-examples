from langchain_ollama import ChatOllama
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from workflow import create_workflow_graph
from prompts import *
from retriever import Retriever

class AdaptiveRAG:
    def __init__(self):
        # Initialize language models
        self.llm = ChatOllama(model="llama3.2:3b-instruct-fp16", temperature=0)
        self.llm_json_mode = ChatOllama(model="llama3.2:3b-instruct-fp16", temperature=0, format="json")
        
        # Set environment variables
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "local-llama32-rag"
        
        self.retriever = Retriever().retriever
        # Initialize web search
        self.web_search_tool = TavilySearchResults(k=3)
        
        # Initialize workflow graph
        self.graph = create_workflow_graph(
            llm=self.llm,
            llm_json_mode=self.llm_json_mode,
            retriever=self.retriever,
            web_search_tool=self.web_search_tool,
            prompts={
                "rag_prompt": rag_prompt,
                "doc_grader_prompt": doc_grader_prompt,
                "doc_grader_instructions": doc_grader_instructions,
                "router_instructions": router_instructions,
                "hallucination_grader_prompt": hallucination_grader_prompt,
                "hallucination_grader_instructions": hallucination_grader_instructions,
                "answer_grader_prompt": answer_grader_prompt,
                "answer_grader_instructions": answer_grader_instructions
            }
        )
        
    def query(self, question: str) -> dict:
        """Query the RAG system with a question using the workflow graph."""
        # Initialize state
        state = {
            "question": question,
            "generation": "",
            "web_search": "No",
            "max_retries": 3,
            "answers": 0,
            "loop_step": 0,
            "documents": []
        }
        
        # Run the workflow graph
        final_state = self.graph.invoke(state)
        
        # Format the response
        return {
            "route": "workflow",
            "reason": "Used workflow graph for processing",
            "answer": final_state["generation"].content if final_state["generation"] else "No answer generated",
            "documents_used": len(final_state["documents"]),
            "web_search_used": final_state["web_search"] == "Yes"
        }



