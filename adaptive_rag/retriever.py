from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings


class Retriever():

    def __init__(self, k=3):
        urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
        ]

        # Load documents
        docs = [WebBaseLoader(url, default_parser='lxml').load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=200
        )
        doc_splits = text_splitter.split_documents(docs_list)

        # Add to vectorDB
        vectorstore = SKLearnVectorStore.from_documents(
            documents=doc_splits,
            embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local"),
        )

        # Create retriever
        self.retriever = vectorstore.as_retriever(k=k)

        