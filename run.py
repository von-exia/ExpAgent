from agent_model.agent_model import AgentModel
from agent_model.env import Environment
from planner.ReAcTree import Config, ReAcTreePlanner
from act import *

# Prepare action and tool dictionaries
action_dict = {}
action_dict.update({
    "answer": Answer,
})

tool_dict = {}
tool_dict.update({
    "shell": Shell,
    "wikipedia_search": WikipediaSearch,
    # "browser": BrowserProcessor,
})

# Set up the environment and planner
cfg = Config()

# Initialize the agent model
# config_path = "./Qwen3-8B-MNN/"
config_path = "./Qwen3-4B-MNN/"
# config_path = "./Qwen2_5-7B-Instruct-MNN/"
agent = AgentModel(config_path, 
                   action_dict=action_dict, 
                   tool_dict=tool_dict, 
                   use_skills=False,
                   use_rag=False,
                   planner_cfg=cfg,
                   planner_cls=ReAcTreePlanner
                   )

env = Environment(critique=agent)
rat_planner = ReAcTreePlanner(
    cfg=cfg,
    agent=agent,
    env=env
)

# Define the query
# query = """Search \"凡人修仙传\" with \"http://baidu.com\", then provide a brief summary."""
# query = """What is the first name of the only Malko Competition recipient from the 20th Century (after 1977) whose nationality on record is a country that no longer exists?"""
query = """Run the py_test.py file."""
terminate_info = rat_planner.collect(query)
print(terminate_info)