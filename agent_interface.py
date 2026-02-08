from agent_model.agent_model import AgentModel
from agent_model.rag import RealTimeRAG
from agent_model.api_models.deepseek import DeepSeekModel
from agent_model.api_models.aliclound import DashTextEmbeddingModel
from agent_model.verifier import Verifier
from act import *

######################################################################
#                        Initialize API Keys                         # 
######################################################################
import os
import json
with open("./api_keys.json", "r") as f:
    api_keys = json.load(f)
if api_keys.get("DEEPSEEK_API_KEY", False):
    os.environ["DEEPSEEK_API_KEY"] = api_keys["DEEPSEEK_API_KEY"]
if api_keys.get("DASHSCOPE_API_KEY", False):
    os.environ["DASHSCOPE_API_KEY"] = api_keys["DASHSCOPE_API_KEY"]


class AgentInterface:
    def __init__(self, planner_type="reactree"):
        ######################################################################
        #                  Initialize Actions/Tools/Skills                   # 
        ######################################################################

        emb_model = DashTextEmbeddingModel()
        rag = RealTimeRAG(embedding_model=emb_model, 
                        chunk_size=1500,
                        chunk_overlap=50, 
                        )

        # Prepare action and tool dictionaries
        action_dict = {}
        action_dict.update({
            "answer": Answer,
        })

        tool_dict = {}
        tool_dict.update({
            "shell": Shell,
            "wikipedia_search": WikipediaSearch,
            "calculator": Calculator,
            "view_text_file": ViewTextFile,
            "write_text_file": WriteTextFile, 
            # "browser": BrowserProcessor,
        })

        ######################################################################
        #                       Initialize LM & Planner                      # 
        ######################################################################
        if planner_type == "reactree":
            from planner.ReAcTree import Config, ReAcTreePlanner
            planner_cls = ReAcTreePlanner
        elif planner_type == "react":
            from planner.ReAct import Config, ReActPlanner
            planner_cls = ReActPlanner
        else:
            raise NotImplementedError(f"planner {planner_type} is not implemented")
        # Set up the verifier and planner
        cfg = Config()

        # Initialize the local model
        config_path = "./models/Qwen3-8B-MNN/"
        # config_path = "./models/Qwen3-4B-MNN/"
        # config_path = "./models/Qwen3-4B-Ins-2507-MNN/"
        # config_path = "./models/Qwen2_5-7B-Instruct-MNN/"
        # logging.info(f"LLM is initialized from {config_path}")

        # Initialize the local model API model
        model = DeepSeekModel(enable_think=False)
        agent = AgentModel(config_path,
                        model=model,
                        action_dict=action_dict, 
                        tool_dict=tool_dict, 
                        use_skills=False,
                        use_rag=rag,
                        planner_cfg=None,
                        planner_cls=None
                        )

        verifier = Verifier(
                # cfg_path=config_path,
                model=agent.model
            )

        self.planner = planner_cls(
                cfg=cfg,
                agent=agent,
                verifier=verifier
            )
        
    def response(self, query, file_path=None, extract_answer=False):
        terminate_info = self.planner.collect(query, extract_answer)
        if extract_answer:
            return terminate_info
        return terminate_info['response']
    
    def visualize(self, title=f"ReAcTree Visualization", save_path="./reactree.png"):
        self.planner.visualize(title=title, save_path=save_path)
