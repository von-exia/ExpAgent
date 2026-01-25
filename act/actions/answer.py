
from act.actions.action import Action


class Answer(Action):
    def execute(self, agent, query: str) -> str:
        prompt = f"""
system
You are a helpful assistant to answer the question.

user
Query: {query}

assistant
/no_think
"""
        return agent.response(prompt, stream=False)
    @classmethod
    def content(cls):
        return "RESPOND the input question/ACHIEVE the goal/ANSWER the query/ANALYSE the information"