import logging

# 基本配置
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='reactree.log',  # 可选：输出到文件
    filemode='w'  # 'a'为追加模式，'w'为覆盖模式
)

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
        self.is_root = False
        # 添加可视化支持
        self.visualization_data = {}


    def add_child(self, child_node):
        child_node.parent = self
        self.children.append(child_node)

    def collect(self, cur_step_id, cur_decision_id, ctx):
        raise NotImplementedError()
    
    def get_tree_structure(self):
        """
        返回树的结构信息，用于可视化
        """
        structure = {
            'node_type': self.__class__.__name__,
            'depth': self.depth,
            'query': self.query,
            'visualization_data': self.visualization_data,
            'children': [child.get_tree_structure() for child in self.children]
        }
        return structure

START_OF_SEQ = "[START OF PREVIOUS SEQUENCE RESULTS]\n"
END_OF_SEQ = "[END OF PREVIOUS SEQUENCE RESULTS]\n"
START_OF_FAL = "[START OF PREVIOUS FALLBACK RESULTS]\n"
END_OF_FAL = "[END OF PREVIOUS FALLBACK RESULTS]\n"
START_OF_PAR = "[START OF PREVIOUS PARALLEL RESULTS]\n"
END_OF_PAR = "[END OF PREVIOUS PARALLEL RESULTS]\n"

class ControlFlowNode(TreeNode):
    def __init__(self, cfg, depth, query, plan_dict):
        super().__init__(cfg, depth)
        self.content = plan_dict['decision']
        self.subgoals = plan_dict['subgoals']

        if self.content == 'sequence':
            self.START = START_OF_SEQ
            self.END = END_OF_SEQ
        elif self.content == 'fallback':
            self.START = START_OF_FAL
            self.END = END_OF_FAL
        elif self.content == 'parallel':
            self.START = START_OF_PAR
            self.END = END_OF_PAR

        self.node_template = """
## {content} node {ind}
subgoal: {subgoal}
observation:\n{response}
"""
        self.cur_query_template = """
"""
        self.working_memory = f"Your primary goal is: {query}\n# Control Flow: {self.content}\n"
        # Track execution history for visualization
        self.execution_history = []
        self.query = query
    
    def update_working_memory(self, n_ind, n_subgoal, response=None):
        if response is not None:
            cur_node_memory = self.node_template.format(content=self.content, ind=n_ind, subgoal=n_subgoal, response=response)
            self.working_memory += cur_node_memory
            n_ind += 1
        wm = self.START + self.working_memory
        for ind, subgoal in enumerate(self.subgoals[n_ind:], start=n_ind):
            wm += f"## {self.content} node {ind}\nsubgoal: {subgoal}\n" 
        wm += self.END    
        # wm += "**Current subgoal to be solved now** is: "
        return wm
    
    def collect(self, cur_step_id, cur_decision_id, ctx):
        next_step_id = cur_decision_id
        if self.content == 'sequence':
            wm = self.update_working_memory(0, self.subgoals[0], None)
            for ind, child in enumerate(self.children):
                next_step_id += 1
                terminate_info = child.collect(next_step_id, 0, ctx=ctx+wm)
                logging.info("-"*20)
                logging.info(f"Control Flow Node: Sequence {ind}:")
                for k, v in terminate_info.items():
                    logging.info(f"{k}: {v}")
                logging.info("-"*20)
                cur_step_id, cur_decision_id = terminate_info['step_id'], terminate_info['decision_id']
                wm = self.update_working_memory(ind, self.subgoals[ind], terminate_info['response'])
                
                # 记录执行历史
                self.execution_history.append({
                    'step_id': next_step_id,
                    'subgoal_index': ind,
                    'subgoal': self.subgoals[ind],
                    'status': 'success' if terminate_info['success'] else 'failed',
                    'response': terminate_info['response'],
                    'control_flow': 'sequence'
                })
                
                if not terminate_info['success']:
                    return Information(step_id=next_step_id,
                                decision_id=cur_decision_id,
                                depth=self.depth,
                                success=False,
                                query=child.query,
                                response=wm).to_dict()
                                
            # 记录执行历史
            self.execution_history.append({
                'step_id': next_step_id,
                'status': 'completed',
                'control_flow': 'sequence',
                'result': 'success'
            })
            
            return Information(step_id=next_step_id,
                                decision_id=cur_decision_id,
                                depth=self.depth,
                                success=True,
                                query=child.query,
                                response=wm).to_dict()

        elif self.content == 'fallback':
            wm = self.update_working_memory(0, self.subgoals[0], None)
            for ind, child in enumerate(self.children):
                next_step_id += 1
                terminate_info = child.collect(next_step_id, 0, ctx=ctx+wm)
                logging.debug("-"*10+f"Control Flow Node:\nFallback {ind}: {terminate_info}")
                cur_step_id, cur_decision_id = terminate_info['step_id'], terminate_info['decision_id']
                wm = self.update_working_memory(ind, self.subgoals[ind], terminate_info['response'])
                logging.debug("-"*20)
                
                # 记录执行历史
                self.execution_history.append({
                    'step_id': next_step_id,
                    'subgoal_index': ind,
                    'subgoal': self.subgoals[ind],
                    'status': 'success' if terminate_info['success'] else 'failed',
                    'response': terminate_info['response'],
                    'control_flow': 'fallback'
                })
                
                if terminate_info['success']:
                    return Information(step_id=next_step_id,
                                        decision_id=cur_decision_id,
                                        depth=self.depth,
                                        success=True,
                                        query=child.query,
                                        response=wm).to_dict()
                                        
            # 记录执行历史
            self.execution_history.append({
                'step_id': next_step_id,
                'status': 'completed',
                'control_flow': 'fallback',
                'result': 'failed'
            })
            
            return Information(step_id=next_step_id,
                                decision_id=cur_decision_id,
                                depth=self.depth,
                                success=False,
                                query=child.query,
                                response="").to_dict()

        elif self.content == 'parallel':
            is_success = True
            wm = self.update_working_memory(0, self.subgoals[0], None)
            parallel_responses = []
            for ind, child in enumerate(self.children):
                next_step_id += 1
                terminate_info = child.collect(next_step_id, 0, ctx=ctx+wm)
                logging.debug("-"*10+f"Control Flow Node:\nParallel {ind}: {terminate_info}")
                cur_step_id, cur_decision_id = terminate_info['step_id'], terminate_info['decision_id']
                if not terminate_info['success']:
                    is_success = False
                parallel_responses.append(terminate_info['response'])
                # 更新工作记忆以包含所有并行结果
                wm = self.update_working_memory(ind, self.subgoals[ind], terminate_info['response'])
                logging.debug("-"*20)
                
                # 记录执行历史
                self.execution_history.append({
                    'step_id': next_step_id,
                    'subgoal_index': ind,
                    'subgoal': self.subgoals[ind],
                    'status': 'success' if terminate_info['success'] else 'failed',
                    'response': terminate_info['response'],
                    'control_flow': 'parallel'
                })

            # 合并所有并行结果
            combined_response = "\n".join(parallel_responses)
            
            # 记录执行历史
            self.execution_history.append({
                'step_id': next_step_id,
                'status': 'completed',
                'control_flow': 'parallel',
                'result': 'success' if is_success else 'failed'
            })
            
            return Information(step_id=cur_step_id,
                                decision_id=cur_decision_id,
                                depth=self.depth,
                                success=is_success,
                                query=child.query,
                                response=combined_response).to_dict()
        else:
            raise NotImplementedError()
        
    def make_message(self):
        return None


class AgentNode(TreeNode):
    def __init__(self, cfg, depth, agent, verifier=None):
        super().__init__(cfg, depth)
        self.agent = agent
        self.verifier = verifier

        self.response_template = "### Previous Action: {name}\n#### Status: {status}\n#### Action Response: {response}"
        self.goal_prefix = "\nCurrent Goal (to be completed now): {query}"
        self.sug_template = "\n### Verifier's Response:\n#### Status: {status}\n#### Improvement Suggestions: {sug}\n"+\
            "Improve the previous response based on Suggestions\n"

        self.load_decision_prompt()
        self.query = None
        self.response = ""
        self.try_think = 0
        self.try_plan = 0
        
        # Track execution history for visualization
        self.execution_history = []
       
    def reset(self):
        self.try_think = 0
        self.try_plan = 0
        self.query = None
        self.child = []
        self.execution_history = []
    
    def load_decision_prompt(self):
        action_content = self.agent.action_content
        with open("./planner/ReAcTree/system_prompt.txt", "r", encoding="utf-8") as f:
            sprompt = f.read()
        sprompt = sprompt.replace("<content_list>", action_content)
        # sprompt = sprompt.replace("<content_list>", ", ".join(self.agent.action_list))
        self.decision_prompt = sprompt
        
    def collect(self, cur_step_id, cur_decision_id, query=None, ctx=""):

        if cur_step_id > self.cfg.max_steps or self.depth > self.cfg.max_depth:
            # 记录执行历史
            self.execution_history.append({
                'step_id': cur_step_id,
                'decision_id': 0,
                'decision': 'max_limit_reached',
                'status': 'failed',
                'response': "Reached max steps/depth limit."
            })
            
            return Information(step_id=cur_step_id,
                                decision_id=0,
                                depth=self.depth,
                                success=False,
                                query=query if query is not None else self.query,
                                response="Reached max steps/depth limit.").to_dict()

        if self.query is not None:
            query = self.query

        working_memory = ctx # Record working memory in this node
        # final_response = "" # To save the context, only record the final response
        completed = False
        for i in range(self.cfg.max_decisions):
            if self.is_root and i == 0:
                decision="Plan"
            else:
                decision_prompt = self.decision_prompt + f"\n**Working Memory**:\n{working_memory}\n**Current Task**:\n{query}\nWhich decision is the best for current condition?\n"
                if self.try_think >= self.cfg.max_think:
                    logging.debug("------------------------------------------------")
                    logging.debug("### Reached max **Think** limit, switching to Act or Plan. ###")
                    logging.debug("------------------------------------------------")
                    decision = self.agent.select(decision_prompt,
                                ["Act", "Plan"])
                elif self.try_plan >= self.cfg.max_plan:
                    logging.debug("------------------------------------------------")
                    logging.debug("### Reached max **Plan** limit, switching to Act or Think. ###")
                    logging.debug("------------------------------------------------")
                    decision = self.agent.select(decision_prompt,
                                ["Think", "Act"])
                else:
                    decision = self.agent.select(decision_prompt,
                                ["Think", "Act", "Plan"])
                # if self.try_plan >= self.cfg.max_plan:
                #     logging.debug("\n### Reached max **Plan** limit, switching to Act. ###\n")
                #     decision = "Act"
                # else:
                #     decision = self.agent.select(decision_prompt,
                #                 ["Act", "Plan"])

                logging.info("------------------------------------------------")
                logging.info(f"**Agent Node** Decision at step {cur_step_id}, decision {i}: {decision}, depth: {self.depth}")
                logging.info(f"Query: {query}")
                # logging.info(f"**Decision Prompt**:\n{decision_prompt}")
            if i == 0 :
                working_memory += "\n[START OF WORKING MEMEORY]\n"
            act_prompt = working_memory+"\n[END OF WORKING MEMEORY]\n"+self.goal_prefix.format(query=query)
            # logging.info(f"**Node Prompt**:\n{act_prompt}")
            logging.info("------------------------------------------------")

            if decision == "Think":
                logging.info("Thinking...")
                think_result, completed = self.agent.reasoning(act_prompt)
                if think_result is not None:
                    think_text = f"\nThink:\n{think_result}\n"
                    logging.info(think_text)
                    working_memory += think_text
                else:
                    logging.warning("think is null!")
                self.try_think += 1
                
                # 记录执行历史
                self.execution_history.append({
                    'step_id': cur_step_id,
                    'decision_id': i,
                    'decision': 'Think',
                    'status': 'completed' if completed else 'incomplete',
                    'response': think_text if think_result else ''
                })
                
                if completed:
                    return Information(step_id=cur_step_id,
                                decision_id=cur_decision_id,
                                depth=self.depth,
                                success=True,
                                query=query,
                                response=think_text).to_dict()
                else:
                    continue

            elif decision == "Act":
                logging.info("Acting...")
                action_name, terminate_info = self.agent.act(act_prompt)
                logging.info(f"Selected action: {action_name}")
                cur_response = self.response_template.format(name=action_name,
                                                                status="success" if terminate_info['success'] else "failure",
                                                                response=terminate_info['response'])
                working_memory += cur_response
                # if current act fails, continue to next decision
                if not terminate_info['success']:
                    continue
                    
                # 记录执行历史
                self.execution_history.append({
                    'step_id': cur_step_id,
                    'decision_id': i,
                    'decision': 'Act',
                    'status': 'success' if terminate_info['success'] else 'failed',
                    'response': cur_response,
                    'action_name': action_name
                })

            elif decision == "Plan":
                logging.info("Planning...")
                plan_dict = self.agent.plan(act_prompt)
                child = ControlFlowNode(self.cfg, self.depth+1, query, plan_dict)
                self.add_child(child)
                subgoals = plan_dict['subgoals']
                for subgoal in subgoals:
                    grand_child = AgentNode(self.cfg, self.depth+2, self.agent, self.verifier)
                    grand_child.query = subgoal
                    child.add_child(grand_child)
                terminate_info = child.collect(cur_step_id+1, i, ctx)
                action_name = "Plan"
                cur_response = self.response_template.format(name=action_name,
                                                                status="success" if terminate_info['success'] else "failure",
                                                                response=terminate_info['response'])
                working_memory += cur_response
                self.try_plan += 1
                # If plan complete, it absolutely completed
                if terminate_info['success']:
                    # 记录执行历史
                    self.execution_history.append({
                        'step_id': cur_step_id,
                        'decision_id': i,
                        'decision': 'Plan',
                        'status': 'success',
                        'response': cur_response,
                        'plan_details': plan_dict
                    })
                    return Information(step_id=cur_step_id,
                                decision_id=cur_decision_id,
                                depth=self.depth,
                                success=True,
                                query=query,
                                response=cur_response).to_dict()
                else:
                    # 记录执行历史
                    self.execution_history.append({
                        'step_id': cur_step_id,
                        'decision_id': i,
                        'decision': 'Plan',
                        'status': 'failed',
                        'response': cur_response,
                        'plan_details': plan_dict
                    })
                    
                    continue
            else:
                raise NotImplementedError()

            # TO DO: Whether use verifier
            completed, sug = self.verifier.evaluate_response(query, act_prompt, cur_response)
            if completed:
                return Information(step_id=cur_step_id,
                                decision_id=cur_decision_id,
                                depth=self.depth,
                                success=True,
                                query=query,
                                response=cur_response).to_dict()
            else:
                working_memory += self.sug_template.format(status="incompleted",
                                                           sug=sug)
                # continue

        logging.debug("*"*20+"MAX Decisions Reached"+"*"*20)
        # Reach the maximum steps
        return Information(
            step_id=cur_step_id,
            decision_id=0,
            depth=0,
            success=False,
            query=query,
            response="Reach the max decisions" if terminate_info is None else terminate_info).to_dict()
        
        
class Reactree():
    def __init__(self, cfg, agent, verifier):
        self.cfg = cfg
        self.agent = agent
        self.verifier = verifier
        self.max_steps = cfg.max_steps
        self.max_decisions = cfg.max_decisions
        self.cur_step_id = 1
        self.cur_decision_id = 1
    def collect(self):
        raise NotImplementedError()


class ReAcTreePlanner(Reactree):
    def __init__(self, cfg, agent, verifier):
        super().__init__(cfg, agent, verifier)
        self.root = AgentNode(cfg, depth=0, agent=agent, verifier=verifier)
        self.root.is_root = True

    def collect(self, query: str, extract_answer=False):
        self.root.reset()
        terminate_info = self.root.collect(cur_decision_id=0, cur_step_id=0, query=query)
        if extract_answer:
            return self.agent.extract_answer(query, terminate_info['response'])
        return terminate_info
    
    def visualize(self, title="ReAcTree Visualization", save_path=None):
        """
        可视化ReAcTree
        """
        try:
            from .visualize_tree import visualize_reactree
            visualize_reactree(self.root, title, save_path)
        except ImportError as e:
            print(f"Visualization failed due to missing modules: {e}")
            print("Please install matplotlib: pip install matplotlib")