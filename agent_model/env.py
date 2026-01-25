class Environment:
    def __init__(self, critique):
        self.critique = critique
    
    def evaluate_response(self, query: str, response: str) -> dict:
        critique_prompt = f"""
Evaluate if the response resolves the user's query.
Query: {query}
Response: {response}
"""
        criti = self.critique.select(
            query=critique_prompt,
            options=["true", "false"]
            )
        # print("-"*15 + "Env Part" + "-"*15)
        # print(critique_prompt, criti)
        # print("-"*15 + "Env Part" + "-"*15)
        solved = True if criti == "true" else False
        return solved