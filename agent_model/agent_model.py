
import re
import MNN.llm as llm
from typing import Dict, List, Optional

from act import *
from agent_model.rag import RealTimeRAG
from agent_model.env import Environment


class AgentModel:
    def __init__(self, 
                 cfg_path: str="./Qwen3-0.6B-MNN/",
                 model=None,
                 action_dict={},
                 tool_dict={},
                 use_skills=True,
                 use_rag: Optional[bool|RealTimeRAG]=False,
                 planner_cls=None,
                 planner_cfg=None
                 ):
        
        self.cfg_path = cfg_path
        if model is not None:
            self.model = model
            print("Using provided LLM model.")
        else:
            self.model = llm.create(cfg_path)
            self.model.load()
            print("LLM model loaded from:", cfg_path)
        
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
        self._init_prompts()
        
        # Ensure the planner of child (e.g., Skill and Browser-tool)
        if planner_cls is None or planner_cfg is None:
            raise "The planner for the Skill or Browser-tool is not decided."
        self.planner_cls = planner_cls
        self.planner_cfg = planner_cfg
        
    def _init_prompts(self):
        self.selection_prompt = """
Your task is to analyze the given question and options (actions), then choose ONLY the correct option index number or the action you want to do, like <answer>index</answer>.
Example 1
Query: 1 + 2 = ?
Options:
(0) 5
(1) 3
(2) 1
(3) -1
Output Format:
<answer>1</answer>

Example 2
Query: What is the capital of France?
Options:
(0) Berlin
(1) Madrid
(2) Paris
(3) Rome
Output Format:
<answer>2</answer>

Example 3
Query: Please bring one pudding and one juice on the coffee table.
Options:
(0) Act
(1) Expand
Output Format:
<answer>1</answer>

The task you need to achieve is:
Query: 
{query}
"""
        
        self.plan_prompt = """
Decompose the goal into subgoals and select the optimal control flow, output decision and subgoals enclosed within <decision>...</decision> and <subgoals>...</subgoals> tags respectively.
## Control Flow:
Sequence: Achieve subgoals sequentially. If any subgoal fails, the sequence is interrupted
Fallback: Attempt subgoals in order until one succeeds. If a subgoal is successful, the remaining subgoals are not attempted
Parallel: Achieve subgoals in parallel; this enables tasks to continue independently, even if one subgoal fails
## Example:
Query: Book a trip to Paris
Decision: <decision>sequence</decision>
Subgoals: 
<subgoals>Find and book available flight</subgoals>
<subgoals>Reserve accommodation</subgoals>
<subgoals>Plan itinerary for sightseeing</subgoals>

user
Query: {query}

assistant
/no_think
"""
        
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
  
        action_dict = {}
        for action_name in self.act_loader._actions:
            action_dict[action_name] = self.act_loader._actions[action_name]
        
        new_agent = AgentModel(model=self.model, 
                               action_dict=action_dict, 
                               tool_dict=tool_dict, 
                               use_skills=False,
                               use_rag=False)
        env = Environment(critique=self.model)
        # cfg = Config()
        # rat_planner = ReAcTreePlanner(
        # cfg=cfg,
        # agent=new_agent,
        # env=env
        # )
        new_planner = self.planner_cls(
            cfg=self.planner_cfg,
            agent=new_agent,
            env=env
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
        new_agent.set_plan_prompt(
f"""Decompose the goal into subgoals and select the optimal control flow, output decision and subgoals enclosed within <decision>...</decision> and <subgoals>...</subgoals> tags respectively.
You need to use the browser toolset to achieve the goal step by step: {tool_dict.keys()}."""
"""
## Control Flow:
Sequence: Achieve subgoals sequentially. If any subgoal fails, the sequence is interrupted
Fallback: Attempt subgoals in order until one succeeds. If a subgoal is successful, the remaining subgoals are not attempted
Parallel: Achieve subgoals in parallel; this enables tasks to continue independently, even if one subgoal fails
## Example:
Query: Search "Artificial Intelligence" with "http://wikipedia.org".
Decision: <decision>sequence</decision>
Subgoals: 
<subgoals>Navigate to http://wikipedia.org with browser-navigate tool</subgoals>
<subgoals>Input "Artificial Intelligence" into the search bar with browser-input_text tool</subgoals>
<subgoals>Click the search button with browser-click tool</subgoals>

user
Query: {query}

assistant
/no_think
""")
        new_agent.set_selection_prompt(
"""
You are a powerful assistant. Your task is to analyze the given question and options, then output ONLY the correct option index number like <answer>index</answer>.
## Example:
Query: 1 + 2 = Which is the correct answer?
Options:
(0) 5
(1) 3
(2) 1
(3) -1
Output Format:
<answer>1</answer>

user
You may need to plan first and use the browser toolset to achieve the goal step by step.
Query: 
{query}
""")
        new_agent.browser_processor = browser_processor
        env = Environment(critique=self)
        # cfg = Config()
        # rat_planner = ReAcTreePlanner(
        # cfg=cfg,
        # agent=new_agent,
        # env=env
        # )
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

    def act(self, goal: str) -> str:
        query = f"""
Goal: {goal}
Which action is more suitable to achieve this goal?
"""
        selected_action = self.select(
            query=query+self.action_content,
            options=self.action_list
        )
        print(f"Selected action: {selected_action}")
        if selected_action is None:
            selected_action = "answer"
        # selected_action = "browser"
            
        if selected_action in ["baidu_search", "wikipedia_search"]:
            res = self.act_loader.create(selected_action).execute(self.model, goal, self.rag_generator)
        elif selected_action in self.act_loader._skills.keys():
            print(f"Performing skill: {selected_action}")
            res = self._perform_skill(selected_action, goal)
        elif selected_action == "browser":
            res = self._perform_browser(goal)
        elif "browser-" in selected_action:
            res = self.act_loader.create(selected_action).execute(self, goal)
        else:
            res = self.act_loader.create(selected_action).execute(self.model, goal)
        return res
    
    
    def reasoning(self, query: str) -> str:
        prompt = f"""
system
You are a reasoning assistant. Provide simple reasoning steps enclosed within <reason> ... </reason>. 

## FORMAT
<reason>
...
</reason>

**Don't think too long! Max length 100 words!**

user
Query: {query}

assistant
/no_think
<reason>
"""
        # print(prompt)
        response = self.model.response(prompt, stream=False)
        def extract_reasoning_from_response(answer_text: str) -> Optional[int]:
            pattern = re.compile(r'<reason>\n(.*?)\n</reason>', re.DOTALL)
            match = pattern.search(answer_text)
            if match:
                try:
                    reasoning = match.group(1)
                    return reasoning
                except ValueError:
                    return None
            return None
        reasoning = extract_reasoning_from_response(response)
        return reasoning

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
        selection_prompt = self.selection_prompt.format(query=query)+ "\nOptions:"
        for ind, option in enumerate(options):
            selection_prompt += f"\n({ind}) {option}"
        selection_prompt += "\n\nassistant\n/no_think"
        # print("######## Selection prompt:\n"+selection_prompt+"\n######## Selection prompt\n")
        response = self.model.response(selection_prompt, stream=False)
        # response = self.model.response(prompt, stream=True)
        # print(response)
        def extract_option_from_answer(answer_text: str) -> Optional[int]:
            pattern = re.compile(r'<answer>(\d+)</answer>')
            match = pattern.search(answer_text)
            if match:
                try:
                    index = int(match.group(1))
                    return index
                except ValueError:
                    return None
            return None
        selected_id = extract_option_from_answer(response)
        # print("Selection Response Text:\n", response, selected_id)
        selected = options[selected_id] if selected_id is not None else None
        return selected
    
    
    def extract_decision_and_subgoals(self, xml_text: str, strict_mode: bool = True) -> Optional[Dict]:
        """
        健壮的提取函数，处理更多边界情况

        Args:
            xml_text: 包含<decision>和<subgoals>标签的文本
            strict_mode: 严格模式，如果为True，遇到错误会抛出异常

        Returns:
            包含decision和subgoals的字典，失败时返回None
        """
        try:
            # 清理输入文本
            xml_text = xml_text.strip()

            # 提取decision
            decision_match = re.search(r'<decision>(.*?)</decision>', xml_text, re.DOTALL | re.IGNORECASE)
            if not decision_match:
                if strict_mode:
                    raise ValueError("未找到有效的<decision>标签")
                return None

            decision = decision_match.group(1).strip()
            print(f"Extracted decision: {decision}")

            # 验证decision值
            valid_decisions = {'sequence', 'fallback', 'parallel'}
            if decision not in valid_decisions:
                if strict_mode:
                    raise ValueError(f"无效的decision值: {decision}，应为sequence/fallback/parallel之一")
                return None

            # 提取所有<subgoals>标签的内容（支持多个）
            subgoals_matches = re.findall(r'<subgoals>(.*?)</subgoals>', xml_text, re.DOTALL | re.IGNORECASE)

            if not subgoals_matches:
                if strict_mode:
                    raise ValueError("未找到有效的<subgoals>标签")
                return None

            # 处理每个匹配到的subgoals内容
            subgoals = []
            for match in subgoals_matches:
                subgoal_text = match.strip()
                if subgoal_text:  # 只添加非空的子目标
                    subgoals.append(subgoal_text)

            if not subgoals:
                if strict_mode:
                    raise ValueError("未找到有效的subgoals")
                return None

            return {
                'decision': decision,
                'subgoals': subgoals,
                'subgoal_count': len(subgoals)
            }

        except Exception as e:
            if strict_mode:
                raise
            print(f"提取过程中出错: {e}")
            return None

    def set_plan_prompt(self, prompt: str):
        self.plan_prompt = prompt
    
    def plan(self, query: str) -> str:
        plan_prompt = self.plan_prompt.format(query=query)
        # print(plan_prompt)
        response = self.model.response(plan_prompt, stream=False)
        # print("Planning Response:\n", response)
        response = self.extract_decision_and_subgoals(response, strict_mode=True)
        return response
    
    
if __name__ == "__main__":
    config_path = "./Qwen3-8B-MNN/"
    # config_path = "./Qwen3-4B-MNN/"
    # config_path = "./MiMo-7B-RL-MNN/"
    agent = AgentModel(config_path)
    
    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        sprompt = f.read()
    sprompt = sprompt.replace("<content_list>", agent.action_content)
    print(sprompt)
    
    query = """How to implement a Python function for addition?"""
    # query = """Help me to write a deepfake detection paper."""
    # query = """What is the first name of the only Malko Competition recipient from the 20th Century (after 1977) whose nationality on record is a country that no longer exists?"""
    
    # res = agent.plan(query)
    # res = agent.reasoning(query)
    # res = agent.select(sprompt+query, options=["Think", "Act", "Expand"])
    # res = agent.select(sprompt+query, options=["Act", "Expand"])
    res = agent.act(query)
    print(f"Response: {res}")