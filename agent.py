import groq
from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class AI_Agent:
    def __init__(self):
        self.client = groq.Client(api_key=GROQ_API_KEY)
    
    def answer(self, query, context):
        system_prompt = (
            "You are an AI assistant that strictly answers questions based only on the provided context. "
            "If the provided context does not contain relevant information to answer the user's query, "
            "respond strictly with: 'No relevant data found.' "
            "Do not make assumptions, provide external knowledge, or infer beyond the given context."
        )

        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context: {context}\n\nUser Query: {query}"}
            ]
        )

        return response.choices[0].message.content.strip()
