from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings


class HierarchicalIndexer:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.embeddings = HuggingFaceEmbeddings()
        self.vector_store = None

    def build_index(self, textbook_name, content):
        documents = [Document(page_content=content, metadata={"source": textbook_name})]
        split_docs = self.text_splitter.split_documents(documents)

        self.vector_store = Chroma.from_documents(split_docs, self.embeddings)

    def get_relevant_chunks(self, query, k=5):
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Build the index first.")

        return self.vector_store.similarity_search(query, k=k)
