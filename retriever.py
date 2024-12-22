from langchain.retrievers import TFIDFRetriever
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma


class Retriever:
    def __init__(self):
        self.tfidf_retriever = None
        self.semantic_retriever = None

    def initialize_retrievers(self, documents):
        # Initialize TF-IDF retriever
        self.tfidf_retriever = TFIDFRetriever.from_documents(documents)

        # Initialize semantic retriever
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(documents, embeddings)
        self.semantic_retriever = vectorstore.as_retriever()

    def retrieve(self, query, hierarchical_indexer, k=5):
        relevant_chunks = hierarchical_indexer.get_relevant_chunks(query, k)

        if self.tfidf_retriever is None or self.semantic_retriever is None:
            self.initialize_retrievers(relevant_chunks)

        tfidf_results = self.tfidf_retriever.get_relevant_documents(query)
        semantic_results = self.semantic_retriever.get_relevant_documents(query)

        # Combine and deduplicate results
        combined_results = list(set(tfidf_results + semantic_results))
        return combined_results[:k]
