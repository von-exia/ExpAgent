import MNN.llm as llm
import logging

from agent_model.base_model import BaseModel
from agent_model.utils import extract_dict_from_text

class Verifier(BaseModel):
    def __init__(self, 
                cfg_path: str="./Qwen3-0.6B-MNN/",
                model=None
                ):
        super().__init__(cfg_path, model)
            
        self._init_prompt()
    
    def _init_prompt(self):
        self.verify_prompt = self.load_prompt("./agent_model/verifier/sys_prompt.txt")

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