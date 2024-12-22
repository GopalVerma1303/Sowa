import streamlit as st
from textbook_processor import TextbookProcessor
from hierarchical_indexer import HierarchicalIndexer
from retriever import Retriever
from rag_system import RAGSystem
from qa_system import QASystem


def main():
    st.title("LLM-powered Question Answering System")

    # Initialize components
    textbook_processor = TextbookProcessor()
    hierarchical_indexer = HierarchicalIndexer()
    retriever = Retriever()
    rag_system = RAGSystem()
    qa_system = QASystem()

    # Sidebar for textbook selection
    st.sidebar.header("Textbook Selection")
    selected_textbooks = st.sidebar.multiselect(
        "Select textbooks", ["Textbook 1", "Textbook 2", "Textbook 3"]
    )

    # Process selected textbooks
    if selected_textbooks:
        with st.spinner("Processing textbooks..."):
            for textbook in selected_textbooks:
                content = textbook_processor.extract_content(textbook)
                hierarchical_indexer.build_index(textbook, content)

        st.success("Textbooks processed and indexed successfully!")

    # User query input
    user_query = st.text_input("Enter your question:")

    if user_query:
        with st.spinner("Generating answer..."):
            # Retrieve relevant content
            relevant_content = retriever.retrieve(user_query, hierarchical_indexer)

            # Generate answer using RAG
            rag_answer = rag_system.generate_answer(user_query, relevant_content)

            # Get final answer from QA system
            final_answer = qa_system.get_answer(user_query, rag_answer)

        # Display results
        st.subheader("Answer:")
        st.write(final_answer)

        st.subheader("Relevant Context:")
        for context in relevant_content:
            st.write(f"- {context}")


if __name__ == "__main__":
    main()
