
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_postgres import PGVector

class Retriever():

    def __init__(self, k=3):

        connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"  # Uses psycopg3!
        collection_name = "my_docs"

        vector_store = PGVector(
            embeddings=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local"),
            collection_name=collection_name,
            connection=connection,
            use_jsonb=True,
        )

        # Create retriever
        self.retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": k})

        