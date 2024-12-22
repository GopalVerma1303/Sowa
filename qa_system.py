from langchain.llms import Groq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


class QASystem:
    def __init__(self):
        self.llm = Groq(api_key="your-groq-api-key")
        self.qa_chain = self._create_qa_chain()

    def _create_qa_chain(self):
        prompt_template = """
        Given the following question and the generated answer from a Retrieval-Augmented Generation (RAG) system, 
        provide a final, refined answer. Ensure the answer is accurate, concise, and well-supported by the given information.

        Question: {question}
        RAG Answer: {rag_answer}

        Final Answer:
        """
        prompt = PromptTemplate(
            template=prompt_template, input_variables=["question", "rag_answer"]
        )
        return LLMChain(llm=self.llm, prompt=prompt)

    def get_answer(self, question, rag_answer):
        return self.qa_chain.run(question=question, rag_answer=rag_answer)
