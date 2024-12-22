import groq
from langchain.prompts import PromptTemplate


class RAGSystem:
    def __init__(self):
        self.groq_client = groq.Groq()

    def generate_answer(self, query, relevant_content):
        prompt_template = """
        Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context:
        {context}

        Question: {query}
        Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "query"]
        )

        formatted_prompt = PROMPT.format(
            context="\n".join([doc.page_content for doc in relevant_content]),
            query=query,
        )

        response = self.groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": formatted_prompt},
            ],
            model="mixtral-8x7b-32768",
            max_tokens=300,
        )

        return response.choices[0].message.content
