from langchain.chains import RetrievalQA
from langchain.llms import Groq
from langchain.prompts import PromptTemplate


class RAGSystem:
    def __init__(self):
        self.llm = Groq(api_key="your-groq-api-key")
        self.qa_chain = None

    def generate_answer(self, query, relevant_content):
        if self.qa_chain is None:
            self._initialize_qa_chain(relevant_content)

        return self.qa_chain.run(query)

    def _initialize_qa_chain(self, relevant_content):
        prompt_template = """
        Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=relevant_content,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT},
        )
