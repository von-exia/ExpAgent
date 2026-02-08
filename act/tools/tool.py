from typing import List, Optional
import ast


class ToolFactory:
    """ToolFactory: implements a factory pattern for creating tool instances."""
    _tool_classes = {}
    def __init__(self):
        self._tool_classes.clear()
    
    @classmethod
    def register(cls, tool_type):
        """注册动作类"""
        def decorator(tool_class):
            cls._tool_classes[tool_type] = tool_class
            return tool_class
        return decorator
    
    @classmethod
    def create(cls, tool_type, *args, **kwargs):
        """创建动作实例"""
        tool_class = cls._tool_classes.get(tool_type)
        if tool_class:
            return tool_class(*args, **kwargs)
        raise ValueError(f"Tool type '{tool_type}' not supported")
    
    @classmethod
    def register(cls, tool_type, tool_class):
        """注册动作类"""
        cls._tool_classes[tool_type] = tool_class
    
    
    @classmethod
    def list_tools(cls):
        """列出所有注册的动作类型"""
        return list(cls._tool_classes.keys())
    
    @classmethod
    def tools_content(cls):
        # ac_cont = "available tools:\n"
        ac_cont = ""
        ind = 0
        for tool_type, tool_class in cls._tool_classes.items():
            ac_cont += f"({ind}) {tool_type}: {tool_class.content()};\n"
            # ac_cont += f"-{tool_type}: {tool_class.content()};\n"
            # ac_cont += f"{tool_type},"
            ind += 1
        return ac_cont[:-2]
    
    @classmethod
    def add_tool_to_repository(cls, new_tool_code, file_path="./tool_repository.py"):
        """
        将新的 Tool 类添加到已有的 Python 文件中
        
        Args:
            file_path: 目标文件路径
            new_tool_code: 要添加的新 Tool 类代码字符串
        """
        
        # 解析新代码为 AST
        new_tool_ast = ast.parse(new_tool_code)
        
        # 获取新类名和装饰器信息
        new_class_name = None
        new_decorators = []
        
        for node in ast.walk(new_tool_ast):
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
                new_lines = new_tool_code.strip().split('\n')
                lines[start_line:start_line] = new_lines
            else:
                # 如果类名存在但未找到定义，追加到文件末尾
                lines.append('')
                lines.extend(new_tool_code.strip().split('\n'))
        else:
            # 类不存在，追加到文件末尾
            lines = existing_content.split('\n')
            if lines[-1].strip():  # 如果最后一行非空，添加空行
                lines.append('')
            lines.extend(new_tool_code.strip().split('\n'))
        
        # 写入文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        return True


# Base Tool class
class Tool:
    def execute(self):
        raise NotImplementedError("Subclasses must implement execute()")
    @classmethod
    def content():
        raise NotImplementedError("Subclasses must implement content()")