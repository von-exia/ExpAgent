import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from .reactree import TreeNode, AgentNode, ControlFlowNode


class TreeVisualizer:
    def __init__(self):
        self.node_positions = {}  # 存储节点的位置
        self.node_colors = {
            'AgentNode': '#90EE90',      # 浅绿色
            'ControlFlowNode': '#87CEEB', # 天蓝色
            'Sequence': '#FFB6C1',       # 浅粉色
            'Fallback': '#DDA0DD',       # 梅花色
            'Parallel': '#F0E68C'        # 卡其色
        }

    def calculate_positions(self, root_node, x=0, y=0, level_width=4, y_spacing=2, current_depth=0):
        """
        计算树中每个节点的位置
        """
        if root_node is None:
            return

        # 为当前节点分配位置
        self.node_positions[id(root_node)] = (x, y)

        # 如果是控制流节点，根据类型设置不同的颜色
        if isinstance(root_node, ControlFlowNode):
            node_type = root_node.content.capitalize() if hasattr(root_node, 'content') else 'ControlFlowNode'
        else:
            node_type = root_node.__class__.__name__

        # 递归计算子节点位置 - 从上到下排列（根在顶部）
        if hasattr(root_node, 'children') and root_node.children:
            num_children = len(root_node.children)
            
            # 动态调整水平间距以避免节点重叠
            # 增加最小间距以防止节点重叠
            adjusted_level_width = max(level_width, 3.0)  # 增大最小间距
            
            if num_children > 1:
                total_width = (num_children - 1) * adjusted_level_width
                start_x = x - total_width / 2
            else:
                start_x = x  # 单个子节点居中
                
            for i, child in enumerate(root_node.children):
                child_x = start_x + i * adjusted_level_width
                child_y = y - y_spacing  # 负向增加y值，使子节点在下方，根节点在上方
                self.calculate_positions(child, child_x, child_y, level_width, y_spacing, current_depth + 1)

    def get_tree_depth(self, node):
        """
        计算树的最大深度
        """
        if node is None:
            return 0
        
        if not hasattr(node, 'children') or not node.children:
            return 1
        
        max_child_depth = 0
        for child in node.children:
            child_depth = self.get_tree_depth(child)
            max_child_depth = max(max_child_depth, child_depth)
        
        return 1 + max_child_depth

    def draw_tree(self, root_node, title="ReAcTree Visualization", save_path=None):
        """
        绘制整个树结构
        """
        # 清空之前的位置信息
        self.node_positions = {}

        # 计算树的深度，以便确定根节点的初始y坐标
        tree_depth = self.get_tree_depth(root_node)
        initial_y = (tree_depth - 1) * 2  # 假设y_spacing为2

        # 计算节点位置，根节点在最上方
        self.calculate_positions(root_node, x=0, y=initial_y)

        # 创建图形
        fig, ax = plt.subplots(figsize=(20, 12))

        # 绘制连接线
        self._draw_edges(ax, root_node)

        # 绘制节点
        self._draw_nodes(ax, root_node)

        # 设置图形属性
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_aspect('equal')

        # 隐藏坐标轴
        ax.set_xticks([])
        ax.set_yticks([])

        # 添加图例
        legend_elements = []
        for node_type, color in self.node_colors.items():
            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, label=node_type))
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()

        # 保存图片到指定路径
        if save_path:
            plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
            print(f"Tree visualization saved to {save_path}")

        plt.show()

    def _draw_edges(self, ax, node):
        """
        递归绘制节点之间的连接线
        """
        if node is None:
            return

        current_pos = self.node_positions.get(id(node))
        if current_pos is None:
            return

        # 绘制到子节点的连线
        if hasattr(node, 'children'):
            for child in node.children:
                child_pos = self.node_positions.get(id(child))
                if child_pos:
                    ax.plot([current_pos[0], child_pos[0]],
                           [current_pos[1], child_pos[1]],
                           'k-', linewidth=1, alpha=0.6)

                # 递归绘制子节点的边
                self._draw_edges(ax, child)

    def _get_node_label(self, node):
        """
        获取节点的标签文本
        """
        if isinstance(node, AgentNode):
            # 对于Agent节点，显示决策类型或状态
            label_parts = ["Agent"]
            
            # 显示节点的查询
            if hasattr(node, 'query') and node.query:
                query_text = node.query
                if len(query_text) > 30:
                    query_text = query_text[:27] + "..."
                label_parts.append(f"Q: {query_text}")

            # 显示执行历史中的最后几次决策
            if hasattr(node, 'execution_history') and node.execution_history:
                last_decisions = []
                for exec_item in node.execution_history[-1:]:  # 最后一次决策
                    decision = exec_item.get('decision', 'Unknown')
                    status = exec_item.get('status', 'Unknown')
                    act_name = exec_item.get('action_name', '')
                    last_decisions.append(f"{decision}: {act_name}({status})")

                if last_decisions:
                    label_parts.extend(last_decisions)

            label = "\n".join(label_parts)

        elif isinstance(node, ControlFlowNode):
            # 对于控制流节点，显示控制类型和子目标数量
            # subgoals_count = len(node.subgoals) if hasattr(node, 'subgoals') else 0
            # label_parts = [f"{node.content.upper()}", f"{subgoals_count} subgoals"]
            label_parts = [f"{node.content.upper()}"]
            
            # 显示节点的查询
            if hasattr(node, 'query') and node.query:
                query_text = node.query
                if len(query_text) > 30:
                    query_text = query_text[:27] + "..."
                label_parts.append(f"Q: {query_text}")

            # 显示部分子目标内容（最多显示2个）
            # if hasattr(node, 'subgoals'):
            #     for i, subgoal in enumerate(node.subgoals[:2]):  # 只显示前两个子目标
            #         if len(subgoal) > 20:
            #             subgoal = subgoal[:17] + "..."
            #         label_parts.append(f"S{i+1}:{subgoal}")

            # 显示执行历史中的状态
            if hasattr(node, 'execution_history') and node.execution_history:
                last_status = node.execution_history[-1] if node.execution_history else {}
                result = last_status.get('result', 'Unknown')
                label_parts.append(f"Result:{result}")

            label = "\n".join(label_parts)
        else:
            label = node.__class__.__name__

        return label

    def _draw_nodes(self, ax, node):
        """
        递归绘制节点
        """
        if node is None:
            return

        pos = self.node_positions.get(id(node))
        if pos is None:
            return

        # 确定节点颜色
        if isinstance(node, ControlFlowNode):
            node_type = node.content.capitalize() if hasattr(node, 'content') else 'ControlFlowNode'
            color = self.node_colors.get(node_type, self.node_colors['ControlFlowNode'])
        else:
            node_type = node.__class__.__name__
            color = self.node_colors.get(node_type, '#FFFFFF')  # 默认白色

        # 获取节点标签
        label = self._get_node_label(node)

        # 绘制圆形节点
        circle = plt.Circle(pos, 0.4, color=color, ec='black', lw=1.5)
        ax.add_patch(circle)

        # 添加标签文本
        ax.text(pos[0], pos[1], label, ha='center', va='center',
               fontsize=7, fontweight='bold', wrap=True)

        # 递归绘制子节点
        if hasattr(node, 'children'):
            for child in node.children:
                self._draw_nodes(ax, child)


def visualize_reactree(root_node, title="ReAcTree Visualization", save_path=None):
    """
    便捷函数，用于可视化ReAcTree
    """
    visualizer = TreeVisualizer()
    visualizer.draw_tree(root_node, title, save_path)