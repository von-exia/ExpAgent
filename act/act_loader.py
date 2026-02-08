from act.actions.action import ActionFactory
from act.tools.tool import ToolFactory
from act.skills.skill_loader import SkillLoader
from typing import List

class ActLoader:
    _act_classes = {}
    _actions = {}  # 存储action类型到类的映射
    _tools = {}    # 存储tool类型到类的映射
    _skills = {}   # 存储skill类型到类的映射

    def __init__(self, action_registry: ActionFactory, tool_registry: ToolFactory=None, use_skills: bool=True):
        # 清空之前的注册
        self._act_classes.clear()
        self._actions.clear()
        self._tools.clear()
        self._skills.clear()

        # 注册所有actions
        if action_registry is not None:
            for action_type in action_registry.list_actions():
                action_class = action_registry._action_classes[action_type]
                self._act_classes[action_type] = action_class
                self._actions[action_type] = action_class

        # 注册所有tools
        if tool_registry is not None:
            for tool_type in tool_registry.list_tools():
                tool_class = tool_registry._tool_classes[tool_type]
                self._act_classes[tool_type] = tool_class
                self._tools[tool_type] = tool_class

        # 加载所有skills
        if use_skills:
            loader = SkillLoader()
            skill_registry = loader.load_all()
            # print(registry._skills)  
            # skill = registry.get("data-analysis")
            # skill_registry = get_skill_registry()
            for skill in skill_registry.list_all():
                self._skills[skill.name] = skill
                self._act_classes[skill.name] = skill
            print(f"Loaded skills: {list(self._skills.keys())}")

    @classmethod
    def create(cls, act_type, *args, **kwargs):
        """创建动作实例"""
        act_class = cls._act_classes.get(act_type)
        if act_class:
            return act_class()
        raise ValueError(f"Act type '{act_type}' not supported")

    @classmethod
    def list_acts(cls):
        """列出所有注册的动作类型，按actions和tools分类"""
        result = []
        # 先添加actions
        result.extend([f"{act_type}" for act_type in cls._actions.keys()])
        # 再添加tools
        result.extend([f"{act_type}" for act_type in cls._tools.keys()])
        # 最后添加skills
        result.extend([f"{act_type}" for act_type in cls._skills.keys()])
        return result

    @classmethod
    def acts_content(cls):
        """返回格式化的可用actions和tools列表"""
        # content = "actions:\n"
        content = ""
        ind = 0
        for action_type, action_class in cls._actions.items():
            # content += f"({ind}) {action_type}: {action_class.content()};\n"
            content += f"- {action_type} ({action_class.content()})\n"
            ind += 1

        for tool_type, tool_class in cls._tools.items():
            # content += f"({ind}) {tool_type}: {tool_class.content()};\n"
            content += f"- {tool_type} ({tool_class.content()})\n"
            ind += 1

        for skill_name, skill in cls._skills.items():
            # content += f"({ind}) {skill_name}: {skill.description};\n"
            content += f"- {skill_name} ({skill.description})\n"
            ind += 1

        return content.rstrip("; ")

    def _get_combined_skill_content(self, skill_name: str):
        """
        获取技能的组合内容，将allowed_tools和content结合起来

        Args:
            skill_name: 技能名称

        Returns:
            包含技能描述、允许的工具和内容的组合字符串
        """
        if skill_name not in self._skills:
            return None
        skill = self._skills[skill_name]
        combined_content = f"Skill Name: {skill.name}\n"
        combined_content += f"Content:\n{skill.content}"
        if skill.allowed_tools:
            allowed_tools: List[str] = skill.allowed_tools
        return combined_content, allowed_tools
    
    def get_skill_prompt(self, skill_name, query):
        skill_prompt, allowed_tools = self._get_combined_skill_content(skill_name)
        prompt = f"""
You need to solve the following query:\n{query}\n with the instructions of the skill below:
{skill_prompt}
"""
        return prompt, allowed_tools