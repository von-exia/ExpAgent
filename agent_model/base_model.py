import MNN.llm as llm


class BaseModel:
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
            
    def load_prompt(self, path):
        with open(path, "r", encoding="utf-8") as f:
            prompt = f.read()
        return prompt