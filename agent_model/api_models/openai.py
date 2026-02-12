import os
from openai import OpenAI

class OpenAIModel:
    def __init__(self, base_url="http://localhost:8080/v1/", api_key=None, model_name=None):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url)
        self.model=model_name

    def response(self, query: str, stream=False) -> str:
        response = self.client.chat.completions.create(
            model=self.model, 
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": query},
            ],
            stream=stream
        )
        return response.choices[0].message.content