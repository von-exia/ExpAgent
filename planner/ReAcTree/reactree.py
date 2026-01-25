from typing import List, Optional
# from debug.aa import ActionFactory
import re

class Information:
    default_info_fields = [
        "step_id", 
        "decision_id", 
        "depth", 
        "success", 
        "query", 
        "response"
    ]
    
    def __init__(self, step_id, decision_id, depth, success, query, response, info_fields=None):
        self.step_id = step_id
        self.decision_id = decision_id
        self.depth = depth
        self.success = success
        self.query = query
        self.response = response
        # 如果初始化时指定了info_fields，则覆盖默认
        self.info_fields = info_fields if info_fields is not None else self.default_info_fields

    def to_dict(self, return_fields=None):
        # 如果指定了return_fields参数，则使用该参数，否则使用实例的info_fields
        fields = return_fields if return_fields is not None else self.info_fields
        return {k: getattr(self, k) for k in fields if hasattr(self, k)}


class TreeNode:
    def __init__(self, cfg, depth):
        self.cfg = cfg
        self.depth = depth
        self.children = []
        self.parent = None
        self.query = None

    def add_child(self, child_node):
        child_node.parent = self
        self.children.append(child_node)
    
    def collect(self, cur_step_id, cur_decision_id, ctx):
        raise NotImplementedError()


class ControlFlowNode(TreeNode):
    def collect(self, cur_step_id, cur_decision_id, ctx):
        if self.content == 'sequence':
            for ind, child in enumerate(self.children):
                terminate_info = child.collect(cur_step_id, cur_decision_id, ctx=ctx)
                print("-"*10+f" Sequence {ind}: {terminate_info}, ctx: {child.query}")
                cur_step_id, cur_decision_id = terminate_info['step_id'], terminate_info['decision_id']
                ctx += terminate_info['response']
                if not terminate_info['success']:
                    return Information(step_id=cur_step_id,
                                decision_id=cur_decision_id,
                                depth=self.depth,
                                success=False,
                                query=child.query,
                                response="").to_dict()
            return Information(step_id=cur_step_id,
                                decision_id=cur_decision_id,
                                depth=self.depth,
                                success=True,
                                query=self.query,
                                response=ctx).to_dict()
        elif self.content == 'fallback':
            for child in self.children:
                terminate_info = child.collect(cur_step_id, cur_decision_id, ctx)
                cur_step_id, cur_decision_id = terminate_info['step_id'], terminate_info['decision_id']
                ctx += terminate_info['response']
                if terminate_info['success']:
                    return Information(step_id=cur_step_id,
                                        decision_id=cur_decision_id,
                                        depth=self.depth,
                                        success=True,
                                        query=child.query,
                                        response=ctx).to_dict()
            return Information(step_id=cur_step_id,
                                decision_id=cur_decision_id,
                                depth=self.depth,
                                success=False,
                                query=self.query,
                                response="").to_dict()
        elif self.content == 'parallel':
            is_success = True
            for child in self.children:
                terminate_info = child.collect(cur_step_id, cur_decision_id, ctx)
                cur_step_id, cur_decision_id = terminate_info['step_id'], terminate_info['decision_id']
                if not terminate_info['success']:
                    is_success = False
                else:
                    ctx += terminate_info['response']
            return Information(step_id=cur_step_id,
                                decision_id=cur_decision_id,
                                depth=self.depth,
                                success=is_success,
                                query=self.query,
                                response=ctx).to_dict()
        else:
            raise NotImplementedError()
        
    def make_message(self):
        return None


class AgentNode(TreeNode):
    def __init__(self, cfg, depth, agent, env=None):
        super().__init__(cfg, depth)
        self.agent = agent
        self.env = env
        
        self.load_decision_prompt()
        self.query = None
        self.response = ""
        self.try_think = 0
        self.try_plan = 0
    
    def load_decision_prompt(self):
        action_content = self.agent.action_content
        with open("./planner/ReAcTree/system_prompt.txt", "r", encoding="utf-8") as f:
            sprompt = f.read()
        # sprompt = sprompt.replace("<content_list>", "["+", ".join(action_list)+"]")
        sprompt = sprompt.replace("<content_list>", action_content)
        self.decision_prompt = sprompt
        
    def collect(self, cur_step_id, cur_decision_id, query=None, ctx=""):
        
        if cur_step_id > self.cfg.max_steps or cur_decision_id > self.cfg.max_decisions or self.depth > self.cfg.max_depth:
            return Information(step_id=cur_step_id, 
                                decision_id=cur_decision_id,
                                depth=self.depth,
                                success=True,
                                query=self.query, 
                                response="Reached max steps/decisions/depth limit.").to_dict()
        
        if self.query is not None:
            query = self.query
        
        
        decision_prompt = self.decision_prompt + f"Goal: {query}\nHistory: {ctx}\n"
        # if self.try_think >= self.cfg.max_think:
        #     print("\n### Reached max **Think** limit, switching to Act or Expand. ###\n")
        #     decision = self.agent.select(decision_prompt,
        #                 ["Act", "Expand"])
        # elif self.try_plan >= self.cfg.max_expand:
        #     print("\n### Reached max **Expand** limit, switching to Act or Think. ###\n")
        #     decision = self.agent.select(decision_prompt,
        #                 ["Think", "Act"])
        # else:
        #     decision = self.agent.select(decision_prompt,
        #                 ["Think", "Act", "Expand"])

        if self.try_plan >= self.cfg.max_expand:
            print("\n### Reached max **Expand** limit, switching to Act. ###\n")
            decision = "Act"
        else:
            decision = self.agent.select(decision_prompt,
                        ["Act", "Expand"])
            
            
        print("------------------------------------------------")
        print(f"Decision at step {cur_step_id}, decision {cur_decision_id}: {decision}, depth: {self.depth}, query: {query}")
        print("------------------------------------------------")
        cur_decision_id += 1
        self.query = query
        
        if decision == "Think":
            print(f"Thinking: {query + ctx}")
            reasoning_result = self.agent.reasoning(query + ctx)
            if reasoning_result is not None:
                ctx += f"Considerations:\n{reasoning_result}"
            print("Thinking resul: ", reasoning_result)
            # Reuse the node
            cur_step_id += 1
            self.try_think += 1
            terminate_info = self.collect(cur_step_id, cur_decision_id, self.query, ctx)
            return terminate_info
        elif decision == "Act":
            print("Acting...")
            action_result = self.agent.act(query + ctx)
            # print("Action Result:\n", action_result)
            cur_step_id += 1
            
            if isinstance(action_result, dict):
                solved = action_result.get('success', False)
                action_result = action_result.get('result', '')
            else:
                solved = self.env.evaluate_response(query, action_result)
                
                
            if solved:
                return Information(step_id=cur_step_id, 
                                   decision_id=cur_decision_id, 
                                   depth=self.depth, 
                                   success=True, 
                                   query=query, 
                                   response=ctx+action_result).to_dict()
            else:
                # Return to the parent
                return Information(step_id=cur_step_id, 
                                   decision_id=cur_decision_id, 
                                   depth=self.depth, 
                                   success=False, 
                                   query=query, 
                                   response=action_result).to_dict()
        elif decision == "Expand":
            decision_dict = self.agent.plan(query+ctx)
            # print(f"Expansion Plan: {decision_dict}")
            
            control_flow = decision_dict['decision'].lower()
            child = ControlFlowNode(self.cfg, self.depth+1)
            child.content = control_flow
            self.add_child(child)
            
            subgoals = decision_dict['subgoals']
            expand_text = self.organize_control_flow_text(decision_dict)
            for subgoal in subgoals:
                grand_child = AgentNode(self.cfg, self.depth+2, self.agent, self.env)
                grand_child.try_plan += 1 # record the plan time
                if "Final Goal:" in query:
                    pattern = r'Final Goal:\n(.+)\n'
                    match = re.search(pattern, query)
                    result = match.group(1)
                    grand_child.query = f"\nCurrent Subgoal (To be solved now):\n{subgoal}\n" + f"Final Goal:\n{result}\n" + expand_text
                elif "Final Goal:" not in query:
                    grand_child.query = f"\nCurrent Subgoal (To be solved now):\n{subgoal}\n" + f"Final Goal:\n{query}\n" + expand_text
                else:
                    grand_child.query = subgoal
                child.add_child(grand_child)
            terminate_info = child.collect(cur_step_id, cur_decision_id, ctx)
            cur_step_id += 1
            return terminate_info
        else:
            raise NotImplementedError()
    
    def organize_control_flow_text(self, decision_dict):
        control_flow = decision_dict['decision'].lower()
        subgoals = decision_dict['subgoals']
        
        text = f"Previous Control Flow Pattern:\n{control_flow}\n"
        for ind, subgoal in enumerate(subgoals, 1):
            text += f"{ind}. {subgoal}\n"
        text = text[:-1]
        return text
        
    
    
class Reactree():
    def __init__(self, cfg, agent, env):
        self.cfg = cfg
        self.agent = agent
        self.env = env
        self.max_steps = cfg.max_steps
        self.max_decisions = cfg.max_decisions
        self.cur_step_id = 1
        self.cur_decision_id = 1
    def collect(self):
        raise NotImplementedError()


class ReAcTreePlanner(Reactree):
    def __init__(self, cfg, agent, env):
        super().__init__(cfg, agent, env)
        self.root = AgentNode(cfg, depth=0, agent=agent, env=env)
    
    def collect(self, query: str):
        terminate_info = self.root.collect(cur_decision_id=0, cur_step_id=0, query=query)
        return terminate_info