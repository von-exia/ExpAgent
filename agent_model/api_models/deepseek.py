import os
from openai import OpenAI

class DeepSeekModel:
    def __init__(self, enable_think=False):
        if not enable_think:
            self.model = "deepseek-chat"
            print("Using DeepSeek-V3.2 non-thinking ...")
        else:
            self.model = "deepseek-reasoner"
            print("Using DeepSeek-V3.2 thinking ...")
        
        self.client = OpenAI(
            api_key=os.environ.get('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com")

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