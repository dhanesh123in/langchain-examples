# Langchain Examples

## Adaptive RAG Q&A with LangGraph, Ollama (LLama3) and Tavily Search

We implement the following Adaptive RAG that combines LLM-based routing, Vectorstore, LLM-based response generation, LLM-based hallucination grader and LLM-based evaluation of the generation
![sparkle](/assets/adaptive_rag_flow.png)

This is based on the example [here](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag_local/#llm) but a PGVector vectorstore is used here. Also `lxml` parser is used in beautiful soup, which extracts more contents than the default html parser for the population of the vectorstore.

### Instructions

1. Create a virtual environment with requirements.txt
2. Create the following environment variables
   ```
    % export TAVILY_API_KEY=<add key here>
    % export LANGSMITH_API_KEY=<add key here>
   ```
   You can get this keys after you create accounts in [Tavily](https://www.tavily.com) and [LangSmith](https://www.langchain.com/langsmith)
3. Install Ollama and pull the required LLM model "llama3.2:3b-instruct-fp16"
4. Run postgres in a docker container with following command
    ```
    % docker run --name pgvector-container -e POSTGRES_USER=langchain -e POSTGRES_PASSWORD=langchain -e POSTGRES_DB=langchain -p 6024:5432 -d pgvector/pgvector:pg16
    ```
5. Populate vector store in postgres by running the python code  
   ```
   % python adaptive_rag/populate_vectorstore.py
   ```
6. Run the streamlit app using the following command
    ```
    % cd adaptive_rag
    % streamlit run app.py
    ```
![sparkle](/assets/Screenshot%20from%202025-04-13%2016-04-42.png)

7. Check LangSmith for traces of the run. An example can be found [here](https://smith.langchain.com/public/34e3de35-eb70-42c8-a227-adfe57dbad69/r) 

![sparkle](/assets/Screenshot%20from%202025-04-13%2016-09-08.png)

