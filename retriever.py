from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma


class Retriever:
    def __init__(self):
        self.bm25_retriever = None
        self.semantic_retriever = None
        self.ensemble_retriever = None

    def initialize_retrievers(self, documents):
        # Initialize BM25 retriever
        self.bm25_retriever = BM25Retriever.from_documents(documents)

        # Initialize semantic retriever
        embeddings = HuggingFaceEmbeddings()
        vectorstore = Chroma.from_documents(documents, embeddings)
        self.semantic_retriever = vectorstore.as_retriever()

        # Create ensemble retriever
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.semantic_retriever],
            weights=[0.5, 0.5],
        )

    def retrieve(self, query, hierarchical_indexer, k=5):
        relevant_chunks = hierarchical_indexer.get_relevant_chunks(query, k)

        if self.ensemble_retriever is None:
            self.initialize_retrievers(relevant_chunks)

        return self.ensemble_retriever.get_relevant_documents(query)
