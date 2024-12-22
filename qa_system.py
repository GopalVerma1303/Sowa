import groq
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


class QASystem:
    def __init__(self):
        self.groq_client = groq.Groq()
        self.openai_llm = OpenAI()
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
        return LLMChain(llm=self.openai_llm, prompt=prompt)

    def get_answer(self, question, rag_answer):
        # Use Groq for initial refinement
        groq_response = self.groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant tasked with refining answers.",
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\nInitial Answer: {rag_answer}\nPlease refine this answer.",
                },
            ],
            model="mixtral-8x7b-32768",
            max_tokens=150,
        )
        groq_refined_answer = groq_response.choices[0].message.content

        # Use OpenAI (through LangChain) for final refinement
        final_answer = self.qa_chain.run(
            question=question, rag_answer=groq_refined_answer
        )

        return final_answer
