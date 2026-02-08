from typing import List, Optional
import ast


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


# Base Action class
class Action:
    def execute(self):
        raise NotImplementedError("Subclasses must implement execute()")
    @classmethod
    def content():
        raise NotImplementedError("Subclasses must implement content()")