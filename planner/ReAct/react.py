from typing import List, Optional
import re

NO_THINK = "\n/no_think"

class Information:
    default_info_fields = [
        "step_id",
        "decision_id",
        "depth",
        "success",
        "query",
        "response",
        "thought",
        "action",
        "observation"
    ]

    def __init__(self, step_id, decision_id, depth, success, query, response, thought="", action="", observation="", info_fields=None):
        self.step_id = step_id
        self.decision_id = decision_id
        self.depth = depth
        self.success = success
        self.query = query
        self.response = response
        self.thought = thought
        self.action = action
        self.observation = observation
        # 如果初始化时指定了info_fields，则覆盖默认
        self.info_fields = info_fields if info_fields is not None else self.default_info_fields

    def to_dict(self, return_fields=None):
        # 如果指定了return_fields参数，则使用该参数，否则使用实例的info_fields
        fields = return_fields if return_fields is not None else self.info_fields
        return {k: getattr(self, k) for k in fields if hasattr(self, k)}


class ReActNode:
    """
    ReAct算法的核心节点类，实现思考(Reasoning)和行动(Action)的循环
    """
    def __init__(self, cfg, agent, env=None):
        self.cfg = cfg
        self.agent = agent
        self.env = env

        self.max_steps = cfg.max_steps
        # self.max_decisions = cfg.max_decisions
        # self.max_think = cfg.max_think

        self.step_count = 0
        self.think_count = 0
        self.history = ""  # 存储ReAct循环的历史记录

        self.load_system_prompt()

    def load_system_prompt(self):
        """加载系统提示词"""
        action_content = self.agent.action_content
        with open("./planner/ReAct/system_prompt.txt", "r", encoding="utf-8") as f:
            sprompt = f.read()
        sprompt = sprompt.replace("<content_list>", action_content)
        self.system_prompt = sprompt

    def react_loop(self, query: str) -> dict:
        """
        执行ReAct主循环
        ReAct算法的核心是交替进行思考(Reasoning)和行动(Action)，直到达到目标
        """
        current_query = query
        history = f"Initial Query: {query}\n"

        for step in range(self.cfg.max_steps):
            if step >= self.cfg.max_steps:
                return Information(
                    step_id=step,
                    decision_id=0,
                    depth=0,
                    success=False,
                    query=current_query,
                    response=f"Reached max steps limit ({self.cfg.max_steps}).",
                    thought="",
                    action="",
                    observation=""
                ).to_dict()

            # 思考阶段 - 分析当前状态和下一步应该做什么
            thought = self.generate_thought(current_query, history)
            if not thought:
                return Information(
                    step_id=step,
                    decision_id=0,
                    depth=0,
                    success=False,
                    query=current_query,
                    response="Failed to generate thought.",
                    thought="",
                    action="",
                    observation=""
                ).to_dict()

            # 行动阶段 - 根据思考结果选择并执行行动
            action_result = self.execute_action(thought, current_query, history)

            # 更新历史记录
            history += f"\nStep {step + 1}:\nThought: {thought}\nAction: {action_result.get('action', '')}\nObservation: {action_result.get('observation', '')}\n"

            # 检查是否完成任务
            if action_result.get('completed', False):
                return Information(
                    step_id=step,
                    decision_id=0,
                    depth=0,
                    success=True,
                    query=current_query,
                    response=history + f"\nFinal Answer: {action_result.get('result', '')}",
                    thought=thought,
                    action=action_result.get('action', ''),
                    observation=action_result.get('observation', '')
                ).to_dict()

            # 更新查询和历史
            current_query = action_result.get('next_query', current_query)
            self.history = history

            # 如果行动失败，可能需要重新思考
            if not action_result.get('success', True):
                continue

        # 达到最大步数限制
        return Information(
            step_id=self.cfg.max_steps,
            decision_id=0,
            depth=0,
            success=False,
            query=current_query,
            response=f"Reached max steps limit without completing the task. History: {history}",
            thought=thought,
            action=action_result.get('action', ''),
            observation=action_result.get('observation', '')
        ).to_dict()

    def generate_thought(self, query: str, history: str) -> str:
        """
        生成思考内容，决定下一步要采取什么行动
        """
        prompt = self.system_prompt + f"\n\nCurrent Query: {query}\nHistory: {history}\n\n"
        prompt += "Generate your next thought and action. Follow the format:\n"
        prompt += "Thought: [Your reasoning here]\nAction: [Action to take]" + NO_THINK

        response = self.agent.response(prompt, False)

        # 解析响应，提取Thought和Action
        thought_match = re.search(r'Thought:\s*(.*?)(?:\n|Action:|$)', response, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()
            return thought
        else:
            # 如果没有明确格式，将整个响应作为思考
            return response

    def execute_action(self, thought: str, query: str, history: str) -> dict:
        """
        执行指定的行动并返回结果
        """
        # 从思考中提取行动
        action_match = re.search(r'Action:\s*(.*?)(?:\n|Observation:|$)', thought, re.DOTALL)
        action_name = None
        if action_match:
            action_desc = action_match.group(1).strip()
            # 尝试解析具体的行动名称
            for action in self.agent.action_list:
                if action.lower() in action_desc.lower():
                    action_name = action
                    break

        # 如果无法从思考中提取行动，让代理选择一个合适的行动
        if not action_name:
            options = self.agent.action_list
            action_name = self.agent.select(
                query=f"Based on this thought: '{thought}' and query: '{query}', which action should be taken?",
                options=options
            )

        if not action_name:
            return {
                'success': False,
                'completed': False,
                'result': '',
                'action': 'No action selected',
                'observation': 'Failed to determine appropriate action',
                'next_query': query
            }

        # 执行行动
        try:
            action_input = f"Thought: {thought}\nQuery: {query}\nAction: {action_name}"
            action_result = self.agent.act(action_input)

            # 检查是否是最终答案
            is_final = self.is_final_answer(action_result, query)

            return {
                'success': True,
                'completed': is_final,
                'result': action_result if is_final else '',
                'action': action_name,
                'observation': str(action_result),
                'next_query': self.update_query(query, action_result)
            }
        except Exception as e:
            return {
                'success': False,
                'completed': False,
                'result': '',
                'action': action_name,
                'observation': f"Error executing action: {str(e)}",
                'next_query': query
            }

    def is_final_answer(self, result: str, query: str) -> bool:
        """
        判断结果是否是最终答案
        """
        # 简单的启发式判断：如果结果看起来像是对原始查询的直接回答
        if not result or not query:
            return False

        # 如果结果包含明确的结束标记
        if '[END]' in result or '[DONE]' in result.upper():
            return True

        # 检查结果是否看起来像一个完整的答案
        result_lower = result.lower()
        if any(keyword in result_lower for keyword in ['answer:', 'final answer:', 'the answer is']):
            return True

        # 如果结果长度适中且似乎回答了查询
        if len(result) > 10 and len(result) < 1000:
            # 这里可以加入更复杂的判断逻辑
            return True

        return False

    def update_query(self, original_query: str, action_result: str) -> str:
        """
        根据行动结果更新查询
        """
        # 在ReAct中，有时需要根据观察结果调整查询
        return original_query


class ReActPlanner:
    """
    ReAct (Reasoning and Acting) 规划器
    ReAct是一种通过交替进行推理(Reasoning)和行动(Action)来解决问题的方法
    """
    def __init__(self, cfg, agent, env=None):
        self.cfg = cfg
        self.agent = agent
        self.env = env
        self.node = ReActNode(cfg, agent, env)

    def collect(self, query: str):
        """
        执行ReAct算法来解决查询
        """
        return self.node.react_loop(query)