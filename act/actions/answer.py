
from act.actions.action import Action
from agent_model.utils import extract_dict_from_text


class Answer(Action):
    def __init__(self):
        self._init_prompt()
    
    def _init_prompt(self):
        self.prompt = """
[ACTION GUIDELINES]
Respond the question/answer the query/improve the response
Note that the answer should be with its unit, if it has

## OUTPUT FORMAT:
Output STRICTLY according to this JSON Schema:
{{
    "answer": "string"
}}

Return only valid JSON.

[USER]
Query: 
{query}

[ASSISTANT]
/no_think
"""     
    
    def extract_answer_from_response(self, response):
        response_dict = extract_dict_from_text(response)
        answer = response_dict['answer']
        return answer
    
    def execute(self, agent, query: str) -> str:
        prompt = self.prompt.format(query=query)
        response = agent.response(prompt, stream=False)
        answer = self.extract_answer_from_response(response)
        return {
            'success': True,
            'response': answer
        }
        
    @classmethod
    def content(cls):
        return """
Function: When required information is enough, respond the question/answer the query/improve the response
Method: Extract the answer from context for current goal
Return: Answer
"""