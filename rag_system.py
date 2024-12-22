import groq
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate


class RAGSystem:
    def __init__(self):
        self.groq_client = groq.Groq()
        self.openai_llm = OpenAI()
        self.qa_chain = None

    def generate_answer(self, query, relevant_content):
        if self.qa_chain is None:
            self._initialize_qa_chain(relevant_content)

        # Use Groq for the initial answer generation
        groq_response = self.groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": f"Question: {query}\nContext: {relevant_content}",
                },
            ],
            model="mixtral-8x7b-32768",
            max_tokens=150,
        )
        groq_answer = groq_response.choices[0].message.content

        # Use OpenAI (through LangChain) for refinement
        refined_answer = self.qa_chain.run(query=query, context=groq_answer)

        return refined_answer

    def _initialize_qa_chain(self, relevant_content):
        prompt_template = """
        Use the following pieces of context to refine the answer to the question at the end. 
        If you can't refine the answer, just return the original answer.

        Context: {context}

        Question: {query}
        Refined Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "query"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.openai_llm,
            chain_type="stuff",
            retriever=relevant_content,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT},
        )
