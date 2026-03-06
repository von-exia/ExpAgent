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


# Base Tool class
class Tool:
    def execute(self):
        raise NotImplementedError("Subclasses must implement execute()")
    @classmethod
    def content():
        raise NotImplementedError("Subclasses must implement content()")