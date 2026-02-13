import re
from typing import Dict, List, Optional

from act import *
from agent_model.base_model import BaseModel
from agent_model.rag import RealTimeRAG
from agent_model.verifier.verifier import Verifier
from agent_model.utils import extract_dict_from_text


class AgentModel(BaseModel):
    def __init__(self, 
                 cfg_path: str="./Qwen3-0.6B-MNN/",
                 model=None,
                 action_dict={},
                 tool_dict={},
                 use_skills=False,
                 use_rag: Optional[bool|RealTimeRAG]=False,
                 planner_cls=None,
                 planner_cfg=None
                 ):
        super().__init__(cfg_path, model)
        
        self._init_act_components(action_dict, tool_dict, use_skills)
        print("AgentModel initialized with actions:", self.action_list)
        
        self.rag_generator = None
        rag_init_flag = False
        if isinstance(use_rag, RealTimeRAG):
            self.rag_generator = use_rag
            rag_init_flag = True
        elif isinstance(use_rag, bool) and use_rag:
            self._init_rag_generator()
            rag_init_flag = True
        # Initialize RAG generator if search tool is included  
        if any(["search" in key for key in tool_dict.keys()]) and not rag_init_flag:
            self._init_rag_generator()
        
        self.browser_processor = None  # Initialize browser processor to None
        self.selection_prompt = None
        self.plan_prompt = None
        self.reasoning_prompt = None
        self._init_prompts()
        
        # Ensure the planner of child (e.g., Skill and Browser-tool)
        if (planner_cls is None or planner_cfg is None) and use_skills:
            raise ValueError("The planner for the Skill or Browser-tool is not passed.")
        self.planner_cls = planner_cls
        self.planner_cfg = planner_cfg
        
        
    def _init_prompts(self):
        self.selection_prompt = self.load_prompt("./agent_model/agent_model/prompts/selection_prompt.txt")
        self.plan_prompt = self.load_prompt("./agent_model/agent_model/prompts/plan_prompt.txt")
        self.reasoning_prompt = self.load_prompt("./agent_model/agent_model/prompts/reasoning_prompt.txt")
        
    def _init_rag_generator(self):
        self.rag_generator = RealTimeRAG(chunk_size=1000)
    
    def _init_act_components(self, action_dict, tool_dict, use_skills) -> None:
        tool_registry = ToolFactory()
        action_registry = ActionFactory()
        
        for action_name, action_class in (action_dict or {}).items():
            action_registry.register(action_name, action_class)
        for tool_name, tool_class in (tool_dict or {}).items():
            tool_registry.register(tool_name, tool_class)
        
        act_loader = ActLoader(action_registry, tool_registry, use_skills)
        self.action_list = act_loader.list_acts()
        self.action_content = act_loader.acts_content()
        self.act_loader = act_loader
    
    def response(self, prompt: str, stream: bool = False) -> str:
        return self.model.response(prompt, stream=stream)
        
    def _perform_skill(self, skill_name: str, query: str) -> str:
        skill_prompt, allowed_tools = self.act_loader.get_skill_prompt(skill_name, query)
        
        tool_dict = {}
        if allowed_tools:
            for tool_name in allowed_tools:
                tool_dict[tool_name] = self.act_loader._tools[tool_name]
        else:
            tool_dict = self.act_loader._tools
  
        action_dict = self.act_loader._actions
        
        skill_agent = AgentModel(model=self.model, 
                               action_dict=action_dict, 
                               tool_dict=tool_dict, 
                               use_skills=False,
                               use_rag=self.rag_generator if self.rag_generator is not None else False
                                )
        skill_ver = Verifier(model=self.model)
        new_planner = self.planner_cls(
            cfg=self.planner_cfg,
            agent=skill_agent,
            verifier=skill_ver
        )
        
        terminate_info = new_planner.collect(skill_prompt)
        return terminate_info
    
    def _perform_browser(self, query: str) -> str:
        # Note this is the only entry point to use browser tool
        # the new angent will not enter this function again to avoid infinite loop
        
        print(f"Starting browser tool to achieve the goal.")
        # Initialize browser processor
        browser_processor = BrowserProcessor()
        browser_processor.setup()
        
        # Prepare new agent with browser tools
        action_dict = {}
        action_dict.update({
            "answer": Answer,
        })
        
        tool_dict = browser_processor.toolset
        print(f"Browser toolset: {tool_dict.keys()}")
        new_agent = AgentModel(model=self.model, 
                               action_dict=action_dict, 
                               tool_dict=tool_dict, 
                               use_skills=False,
                               use_rag=False)
        
        new_agent.browser_processor = browser_processor
        env = Verifier(model=self.model)
        new_planner = self.planner_cls(
            cfg=self.planner_cfg,
            agent=new_agent,
            env=env
        )
        

        browser_query = query
        # print(query)
        if "Current Subgoal (To be solved now):" in query:
            pattern = r'Current Subgoal \(To be solved now\):\n(.+)\n'
            # print(pattern)
            match = re.search(pattern, query)
            result = match.group(1)
            browser_query = result
        
        prompt = f"""
You have access to a browser toolset to help achieve the goal, with Browser toolset: {tool_dict.keys()}
You need to plan and use the browser toolset to achieve the goal step by step.
Goal: {browser_query}
"""
        # print(f"Browser Agent Prompt:\n{prompt}\n")
        terminate_info = new_planner.collect(prompt)['response']
        
        # Closedown browser after use
        browser_processor.closedown()
        print(f"Ending the browser tool.")
        result = new_planner.agent.browser_processor.browser_manager.current_page_text
        return result

        
    ####################################################################################
    #                         Implementations for act action                           #
    ####################################################################################

    def act(self, goal: str) -> str:
        query = f"""
ACT LIST:
{self.action_content}        
        
{goal}

Prioritize using acts/tools to obtain current information for your answer, avoiding reliance on internal knowledge or speculation.

From the ACT LIST, which is the most suitable act to achieve the current goal? 
"""
        selected_act = self.select(
            query=query,
            options=self.action_list
        )
        if selected_act is None:
            selected_act = "answer"
            
        if selected_act in ["baidu_search", "wikipedia_search"]:
            res = self.act_loader.create(selected_act).execute(self.model, goal, self.rag_generator)
        elif selected_act in self.act_loader._skills.keys():
            print(f"Performing skill: {selected_act}")
            res = self._perform_skill(selected_act, goal)
        elif selected_act == "browser":
            res = self._perform_browser(goal)
        elif "browser-" in selected_act:
            res = self.act_loader.create(selected_act).execute(self, goal)
        else:
            res = self.act_loader.create(selected_act).execute(self.model, goal)
        return selected_act, res

    ####################################################################################
    #                      Implementations for reasoning action                        #
    ####################################################################################
    
    def extract_think_from_json(self, response):
        response_dict = extract_dict_from_text(response)
        reasoning = response_dict['think']
        status = response_dict['status']
        return reasoning, status
    
    def reasoning(self, query: str) -> str:
        reasoning_prompt = self.reasoning_prompt.format(query=query, act_list=self.action_content)
        response = self.model.response(reasoning_prompt, stream=False)
        reasoning, status = self.extract_think_from_json(response)
        status = True if status == "completed" else False
        return reasoning, status

    ####################################################################################
    #                      Implementations for selection action                        #
    ####################################################################################
    
    def extract_option_from_json(self, response):
        response_dict = extract_dict_from_text(response)
        selected_index = response_dict['selected_index']
        selected_option = response_dict['selected_option_name']
        return selected_index, selected_option
    
    def set_selection_prompt(self, prompt: str):
        self.selection_prompt = prompt
    
    def select(self, 
                query: str,
                options: List[str]) -> str:
        """
        Select the most likely option based on the prompt and options.
        
        Args:
            query: the question prompt
            options: a list of option strings
            
        Returns:
            The selected option string.
        """
        
        ######## Organize the options text #########
        options_text = ""
        for ind, option in enumerate(options):
            options_text += f"({ind}) {option}\n"
        
        selection_prompt = self.selection_prompt.format(query=query, options_text=options_text)

        # print("######## Selection prompt:\n"+selection_prompt+"\n######## Selection prompt\n")
        response = self.model.response(selection_prompt, stream=False)
        # print("######## Selection response:\n"+response+"\n######## Selection response\n")
        selected_index, selected_option = self.extract_option_from_json(response)
        # print("Selection Response Text:\n", response, selected_id)
                
        if selected_index >= len(options) or options[selected_index] != selected_option: # If number in response exceeds the length of options
            print("Warning:: Selection Response Text:\n", response, selected_index, options)
            return selected_option
        selected = options[selected_index]
        return selected

    ####################################################################################
    #                      Implementations for plan action                             #
    ####################################################################################

    def extract_control_flow_from_json(self, response):
        response_dict = extract_dict_from_text(response)
        subgoals = []
        for subgoal_dict in response_dict['subgoals']:
            subgoals.append(subgoal_dict['subgoal'])
        return {
            'decision': response_dict['control_flow'],
            'subgoals': subgoals,
            'subgoal_count': len(subgoals)
            }
    
    def set_plan_prompt(self, prompt: str):
        self.plan_prompt = prompt
    
    def plan(self, query: str) -> str:
        plan_prompt = self.plan_prompt.format(query=query, act_list=self.action_content)
        response = self.model.response(plan_prompt, stream=False)
        response = self.extract_control_flow_from_json(response)
        return response

    ####################################################################################
    #                Implementations for answer extraction action                      #
    ####################################################################################

    def extract_answer_from_response(self, response):
        response_dict = extract_dict_from_text(response)
        answer = response_dict['answer']
        return answer
    
    def extract_answer(self, query, response):
        prompt = self.load_prompt("./agent_model/agent_model/prompts/extract_answer.txt").format(query)
        response = self.model.response(prompt, False)
        answer = self.extract_answer_from_response(response)
        print(f"Extracted answer: {answer}")
        return answer
