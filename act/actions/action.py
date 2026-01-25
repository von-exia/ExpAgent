import re
from typing import List, Optional
import ast

import importlib
import sys

def reload_module_and_import_star(module_name, target_globals=None):
    """
    重新加载模块并执行 from module_name import *
    
    Args:
        module_name: 模块名称（字符串）
        target_globals: 要更新到的命名空间（默认为调用者的globals）
    """
    if target_globals is None:
        # 获取调用者的全局命名空间
        target_globals = sys._getframe(1).f_globals
    
    # 如果模块已经导入，重新加载它
    if module_name in sys.modules:
        module = importlib.reload(sys.modules[module_name])
    else:
        # 导入模块
        module = __import__(module_name, target_globals, target_globals, ['*'])
    
    # 获取模块中所有不以下划线开头的公共名称
    public_names = [name for name in dir(module) if not name.startswith('_')]
    
    # 将这些名称添加到目标命名空间
    for name in public_names:
        target_globals[name] = getattr(module, name)
    
    return module

class ActionFactory:
    """ActionFactory: implements a factory pattern for creating action instances."""
    
    _action_classes = {}
    
    @classmethod
    def register(cls, action_type):
        """注册动作类"""
        def decorator(action_class):
            cls._action_classes[action_type] = action_class
            return action_class
        return decorator
    
    @classmethod
    def register(cls, action_type, action_class):
        """注册动作类"""
        cls._action_classes[action_type] = action_class

    
    @classmethod
    def create(cls, action_type, *args, **kwargs):
        """创建动作实例"""
        action_class = cls._action_classes.get(action_type)
        if action_class:
            return action_class(*args, **kwargs)
        raise ValueError(f"Action type '{action_type}' not supported")
    
    @classmethod
    def list_actions(cls):
        """列出所有注册的动作类型"""
        return list(cls._action_classes.keys())
    
    @classmethod
    def actions_content(cls):
        # ac_cont = "available actions:\n"
        ac_cont = ""
        ind = 0
        for action_type, action_class in cls._action_classes.items():
            ac_cont += f"({ind}) {action_type}: {action_class.content()};\n"
            # ac_cont += f"-{action_type}: {action_class.content()};\n"
            # ac_cont += f"{action_type},"
            ind += 1
        return ac_cont[:-2]
    
    @classmethod
    def add_action_to_repository(cls, new_action_code, file_path="./action_repository.py"):
        """
        将新的 Action 类添加到已有的 Python 文件中
        
        Args:
            file_path: 目标文件路径
            new_action_code: 要添加的新 Action 类代码字符串
        """
        
        # 解析新代码为 AST
        new_action_ast = ast.parse(new_action_code)
        
        # 获取新类名和装饰器信息
        new_class_name = None
        new_decorators = []
        
        for node in ast.walk(new_action_ast):
            if isinstance(node, ast.ClassDef):
                new_class_name = node.name
                # 收集装饰器
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Call):
                        if isinstance(decorator.func, ast.Attribute):
                            if decorator.func.attr == 'register':
                                # 获取装饰器参数
                                if decorator.args:
                                    new_decorators.append(decorator.args[0].value)
                    elif isinstance(decorator, ast.Attribute):
                        new_decorators.append(decorator.attr)
        
        # 读取现有文件
        with open(file_path, 'r', encoding='utf-8') as f:
            existing_content = f.read()
        
        # 解析现有文件
        existing_ast = ast.parse(existing_content)
        
        # 检查是否已存在相同类名
        existing_classes = []
        for node in ast.walk(existing_ast):
            if isinstance(node, ast.ClassDef):
                existing_classes.append(node.name)
        
        # 检查是否已存在相同的装饰器注册
        existing_decorators = []
        for node in ast.walk(existing_ast):
            if isinstance(node, ast.ClassDef):
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Call):
                        if isinstance(decorator.func, ast.Attribute):
                            if decorator.func.attr == 'register':
                                if decorator.args:
                                    existing_decorators.append(decorator.args[0].value)
        
        # 如果类已存在或装饰器已注册，则更新现有类
        if new_class_name in existing_classes or any(d in existing_decorators for d in new_decorators):
            # 找到并替换现有类
            lines = existing_content.split('\n')
            
            # 查找类定义开始位置
            start_line = -1
            end_line = -1
            indent_level = 0
            in_target_class = False
            
            for i, line in enumerate(lines):
                stripped = line.strip()
                
                # 查找目标类开始
                if stripped.startswith(f'class {new_class_name}'):
                    start_line = i
                    in_target_class = True
                    indent_level = len(line) - len(line.lstrip())
                    continue
                
                if in_target_class:
                    # 检查是否仍在同一类中（通过缩进判断）
                    if stripped and (len(line) - len(line.lstrip())) <= indent_level and not stripped.startswith(' '):
                        end_line = i
                        break
            
            # 如果找到类定义，替换它
            if start_line != -1:
                if end_line == -1:
                    end_line = len(lines)
                
                # 移除旧类定义
                del lines[start_line:end_line]
                
                # 插入新类定义
                new_lines = new_action_code.strip().split('\n')
                lines[start_line:start_line] = new_lines
            else:
                # 如果类名存在但未找到定义，追加到文件末尾
                lines.append('')
                lines.extend(new_action_code.strip().split('\n'))
        else:
            # 类不存在，追加到文件末尾
            lines = existing_content.split('\n')
            if lines[-1].strip():  # 如果最后一行非空，添加空行
                lines.append('')
            lines.extend(new_action_code.strip().split('\n'))
        
        # 写入文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        return True


# Base Action class
class Action:
    def execute(self):
        raise NotImplementedError("Subclasses must implement execute()")
    @classmethod
    def content():
        raise NotImplementedError("Subclasses must implement content()")



    



    
# @ActionFactory.register("design")
# class Design(Action):
#     def execute(self, agent, query: str) -> str:
#         sys_prompt = f"""
# You are an action designer. Your task is to design a new action to achieve the goal, which should be **specific and simple**. Remember to construct the system prompt and disable the thinking mode via /no_think.

# ## Example:
# <action>
# @ActionFactory.register("response")
# class Response(Action):
#     def execute(self, agent, query: str) -> str:
#         system_prompt = f\"\"\"
# system
# You are a helpful assistant to answer the question.

# user
# Query: {{query}}

# assistant
# /no_think
# \"\"\"
#         return agent.response(system_promp+"\\n/no_think", stream=False)
#     @classmethod
#     def content(cls):
#         return "Response the input question/acheive the goal/answer the query."
# </action>

# user
# Query: {query}

# assistant
# /no_think
# """
#         prompt = sys_prompt
#         # print(prompt)
#         response = agent.response(prompt, stream=False)
#         # print(response)
#         def extract_creation_from_response(answer_text: str) -> Optional[int]:
#             pattern = re.compile(r'<action>\n(.*?)\n</action>', re.DOTALL)
#             match = pattern.search(answer_text)
#             if match:
#                 try:
#                     creatio = match.group(1)
                    
#                     # Extract Action's name
#                     pattern_name = r'@ActionFactory\.register\("([^"]+)"\)'
#                     match_name = re.search(pattern_name, answer_text)
#                     if match_name:
#                         name = match_name.group(1)  # 提取括号内的内容
#                     return creatio, name
#                 except ValueError:
#                     return None
#             return None
#         created, name = extract_creation_from_response(response)
        
#         ############################## UPDATE MODULE ##############################
#         from aa import ActionFactory
#         ActionFactory.add_action_to_repository(created)
#         reload_module_and_import_star("action_repository")
        
#         print("New Action name:", name)
#         res = ActionFactory.create(name).execute(agent, query)
#         # return created, res
#         return res
#     @classmethod
#     def content(cls):
#         return "Check the actions list. When the current actions are not enough to achieve the goal, design a new action as much specific as possible"