import networkx as nx
import matplotlib.pyplot as plt

from agent_model.utils import extract_dict_from_text

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='dag_react.log',  # 可选：输出到文件
    filemode='w'  # 'a'为追加模式，'w'为覆盖模式
)

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


def build_and_traverse_with_networkx(dag_dict, visualize=False):
    # 创建有向图
    G = nx.DiGraph()
    
    # 添加节点和属性
    for item in dag_dict['sub-questions']:
        G.add_node(item['index'], sub_question=item['sub-question'], action=item['act_name'])
        
        # 添加边
        for dep in item['dependences']:
            G.add_edge(dep, item['index'])
    
    # 1. 拓扑排序遍历
    # print("--- NetworkX 拓扑排序结果 ---")
    order = list(nx.topological_sort(G))
    # for idx in order:
    #     # node_data = G.nodes[idx]
    #     # print(f"Index {idx}: {node_data['action']} - {node_data['sub_question']}")
    #     print(str(dag_dict['sub-questions'][idx]))
    
    # 2. (可选) 可视化绘图
    # 注意：运行此部分需要系统支持 GUI 或保存为文件
    if visualize:
        try:
            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=True, node_color='lightblue', arrows=True)
            # plt.show() # 如果在本地运行可取消注释
            plt.savefig('dag_graph.png', 
                dpi=300,              # 高分辨率 (默认 100)
                bbox_inches='tight',  # 自动裁剪空白边缘
                facecolor='white')    # 背景色
            print("\n(图形已生成，如需显示请取消 plt.show() 注释)")
        except Exception as e:
            print(f"可视化失败: {e}")
            pass
    return G, order


class DAGReActLoop:
    """
    ReAct算法的核心节点类，实现思考(Reasoning)和行动(Action)的循环
    """
    def __init__(self, cfg, agent, verifier=None):
        self.cfg = cfg
        self.agent = agent
        self.verifier = verifier
        
        self.max_steps = cfg.max_steps

        self.load_system_prompt()
        self.observation_template = "\nObservation {idx}:\nStatus: {status}\nAct result: {result}\n"


    def load_system_prompt(self):
        """加载系统提示词"""
        action_content = self.agent.action_content
        with open("./planner/DAG_ReAct/system_prompt.txt", "r", encoding="utf-8") as f:
            sprompt = f.read()
        sprompt = sprompt.replace("<content_list>", action_content)
        self.system_prompt = sprompt


    def forward(self, query: str) -> dict:
        """
        Execute DAG-ReAct main loop
        The core of the ReAct algorithm is to alternate between reasoning and acting until the goal is reached
        """
        original_query = query
        traj = f"Your primary question:\n{query}\n"

        for step in range(self.cfg.max_steps):

            # traj += f"\n--- Step {step} ---\n"
            # Planning stage - decompose the main question into sub-questions and determine the execution order
            dag, order = self.generate_DAG(traj)
            # Execution stage - execute the sub-questions according to the DAG structure and update the trajectory with observations
            traj = self.execute_DAG(dag, order, traj)
            # Judement stage - determine whether the primary question is completed based on the trajectory, if not, provide improvement suggestions for the next step
            completed, sug = self.verifier.judge_and_contract(original_query, traj)
            if completed:
                return Information(step_id=step,
                                decision_id=step,
                                depth=0,
                                success=True,
                                query=original_query,
                                response=traj).to_dict()
            else:
                traj = f"Your primary question:\n{sug}\n"
                # traj += f"\nImprovement Suggestions:\nThe primary question is incomplete. {sug}\n"
                logging.debug(f"{'='*50}")
                logging.debug(f"Step {step} trajectory:\n{traj}\n{'='*50}")
            
        # 达到最大步数限制
        return Information(
            step_id=self.cfg.max_steps,
            decision_id=0,
            depth=0,
            success=False,
            query=original_query,
            response=f"Reached max steps limit without completing the task. Trajectory: {traj}",
        ).to_dict()


    def generate_DAG(self, traj: str):
        """
        生成思考内容，决定下一步要采取什么行动
        """
        prompt = self.system_prompt.format(traj=traj)
        response = self.agent.response(prompt, False)
        dag_dict = extract_dict_from_text(response)
        dag, order = build_and_traverse_with_networkx(dag_dict, visualize=False)
        
        return dag_dict['sub-questions'], order


    def execute_DAG(self, dag, order, traj) -> dict:
        """
        执行指定的行动并返回结果
        """
        # 记录每个节点的执行状态
        node_success = {}

        for idx in order:
            node_data = dag[idx]
            dependences = node_data.get('dependences', [])
            act_name = node_data['act_name']

            # Update the trajectory with the current sub-question
            traj += f"\nNode {idx}: {str(node_data)}"

            # 检查该节点的 dependences 是否都成功
            dep_failed = False
            failed_deps = []
            for dep_idx in dependences:
                if dep_idx not in node_success or not node_success[dep_idx]:
                    dep_failed = True
                    failed_deps.append(dep_idx)

            if dep_failed:
                observation = self.observation_template.format(
                    idx=idx,
                    status="failure",
                    result=f"Dependences {failed_deps} did not succeed, skipping execution."
                )
                traj += observation
                node_success[idx] = False
                break

            # 执行 act
            prompt = traj + f"\nCurrent task to be executed:\n{node_data['sub-question']}\n/no_think"
            terminate_info = self.agent.direct_act(act_name, prompt)
            observation = self.observation_template.format(idx=idx,
                                                            status="success" if terminate_info['success'] else "failure",
                                                            result=terminate_info['response'])
            traj += observation
            node_success[idx] = terminate_info['success']

        return traj
            


class DAGReActPlanner:
    def __init__(self, cfg, agent, verifier=None):
        self.cfg = cfg
        self.agent = agent
        self.verifier = verifier
        self.main_loop = DAGReActLoop(cfg, agent, verifier)

    def collect(self, query: str, file_path=None, extract_answer=False):
        terminate_info = self.main_loop.forward(query=query)
        if extract_answer:
            return self.agent.extract_answer(query, terminate_info['response'])
        return terminate_info
    
    def judge(self, query: str, gt: str, file_path=None):
        terminate_info = self.main_loop.forward(query=query)
        traj = terminate_info['response']
        answer, judge = self.agent.judge(query, traj, gt)
        return answer, judge