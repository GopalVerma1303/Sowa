import groq


class QASystem:
    def __init__(self):
        self.groq_client = groq.Groq()

    def get_answer(self, question, rag_answer):
        prompt = f"""
        Given the following question and the generated answer from a Retrieval-Augmented Generation (RAG) system, 
        provide a final, refined answer. Ensure the answer is accurate, concise, and well-supported by the given information.

        Question: {question}
        RAG Answer: {rag_answer}

        Final Answer:
        """

        response = self.groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant tasked with refining answers.",
                },
                {"role": "user", "content": prompt},
            ],
            model="mixtral-8x7b-32768",
            max_tokens=200,
        )

        return response.choices[0].message.content
