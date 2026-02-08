import MNN.llm as llm
import logging

from agent_model.utils import extract_dict_from_text

class Verifier:
    def __init__(self, 
                cfg_path: str="./Qwen3-0.6B-MNN/",
                model=None,):
        self.cfg_path = cfg_path
        if model is not None:
            self.model = model
            print("Verifier using provided LLM model.")
        else:
            self.model = llm.create(cfg_path)
            self.model.load()
            print("Verifier LLM model loaded from:", cfg_path)
            
        self._init_prompt()
    
    def _init_prompt(self):
        self.verify_prompt = """
[EVALUATION GUIDELINES]
Evaluate the RESPONSE whether accompulish the current goal, based on CONTEXT and QUERY only. Focus on the logits of RESPONSE.

## Evaluation criteria
1. COMPLETENESS: Does the RESPONSE solve all aspects of the QUERY?
2. ACCURACY: Is the information or the calculation in the RESPONSE correct?

## COMMON MISTAKES TO AVOID
- WRONG: `round(17000, -3) * 1000 / 1000`  # Unnecessary operations
- WRONG: `round(17000, -3)`  # Missing division by 1000
- WRONG: `(round(17000, -3) / 1000) * 1000`  # Undoes the division
- CORRECT: `round(17000, -3) / 1000`  # Simple and correct

## Output requirements:
- The status of the response: completed|incompleted
- If incompleted, give suggestion for improvement

[START OF CONTEXT]
{context}
[END OF CONTEXT]

[START OF QUERY]
{query}
[END OF QUERY]

[START OF RESPONSE]
{response}
[END OF RESPONSE]

## OUTPUT FORMAT:
Output STRICTLY according to this JSON Schema:
{{
    "status": "completed|incompleted",
    "suggestion": "string"
}}

Return only valid JSON.

[ASSISTANT]
/no_think
"""     

    def extract_evaluation(self, response: str):
        response_dict = extract_dict_from_text(response)
        status = response_dict['status']
        sug = response_dict['suggestion']
        return status, sug

    def evaluate_response(self, query, context: str, response: str) -> bool:
        verify_prompt = self.verify_prompt.format(query=query, context=context, response=response)
        # logging.debug("Verified Prompt:\n"+verify_prompt)
        response = self.model.response(verify_prompt, False)
        status, sug = self.extract_evaluation(response)
        logging.debug(f"Verifier response: status: {status}; sug:{sug}")
        completed = True if status == "completed" else False
        return completed, sug