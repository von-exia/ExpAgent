import requests
from baidusearch.baidusearch import search
from bs4 import BeautifulSoup
import html

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


def extract_webpage_content(url: str, max_chars: int = 500) -> str:
    """
    提取网页主要内容并返回文本
    
    Args:
        url: 网页URL
        max_chars: 返回的最大字符数
        
    Returns:
        网页的文本内容摘要
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # 检查请求是否成功
        
        # 检测编码
        if response.encoding is None:
            response.encoding = 'utf-8'
        
        # 使用BeautifulSoup解析HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 移除不需要的标签
        for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
            element.decompose()
        
        # 尝试提取主要内容
        content = ""
        content = soup.get_text(separator=' ', strip=True)

        return content
        
    except requests.exceptions.RequestException as e:
        print(f"Request error for {url}: {e}")
        return ""
    except Exception as e:
        print(f"Error parsing {url}: {e}") 
        return ""


# @ToolFactory.register("baidu_search")
# class BaiduSearch(Tool):
#     def execute(self, agent, query: str, rag_generator) -> str:
#         self.rag = rag_generator
#         self.query = query

#         # Improved prompt for keyword extrtool
#         extrtool_prompt = """Given the user query below, extract 1-3 most relevant and concise keywords for web searching to respond the query, generate between <key_words> and </key_words>. 
 
#         Query: {query}
        
#         Keywords: <key_words> 1. kw1 2. kw2 </key_words>"""
        
#         # Send to agent for keyword extrtool
#         formatted_prompt = extrtool_prompt.format(query=query)
#         response = agent.response(formatted_prompt + "\n/no_think\n<key_words> 1.", stream=False).strip()
        
#         def extract_keywords(response):
#             # 提取<key_words>标签内的内容
#             match = re.search(r"<key_words>(.+?)</key_words>", response, re.DOTALL)
#             if match:
#                 content = match.group(1).strip()
#                 # print(f"标签内容: {content}")
#                 keywords = re.findall(r"\d+\.\s*(.+?)(?=\s*\d+\.|$)", content)
#                 # print(f"提取的关键词: {keywords}")
#                 return keywords
#             return None
#         # print(response)
#         keywords = extract_keywords(response)[0]
#         # print(f"Extracted keywords: {keywords}")
        
#         # Use keywords for searching
#         results = search(keywords, num_results=10)
#         res = "Search Results:\n"
#         content_list = []
#         for idx, r in enumerate(results[:3], 1):  # Show top 5 results
#             # res += f"{idx}. {r['title']}\n   URL: {r['url']}\n"
#             url = r['url']
#             try:
#                 page_content = extract_webpage_content(url)
#                 if page_content or len(page_content) > 0:
#                     # res += f"   📄 **Content Preview:**\n"
#                     # res += f"   {page_content}\n"
#                     content_list.append(page_content)
#                     # print("URL: ", url)
#                     # print("Success:")
#                     # print(page_content)
#             except Exception as e:
#                 # print(f"Error extracting content from {url}: {e}")
#                 # res += f"   ⚠️ Could not extract content\n"
#                 pass
                
#         ret = ""
#         res = self.rag.execute(self.query, content_list, k=3)
#         for i, r in enumerate(res, 1):
#             ret += f"\nRetrival result {i}:\n{r}\n"
            
#         # print("*"*20 + " RAG " + "*"*20)
#         # print(ret)
#         # print("*"*20 + " RAG " + "*"*20)
            
#         res_prompt = f"""Given the user query below, acheive the goal based on the Retrival reslts. 
# Goal: {query}
# Retrival results: {ret}
# """
#         formatted_prompt = res_prompt.format(query=query, ret = ret)
#         response = agent.response(formatted_prompt + "\n/no_think", stream=False).strip()
        
#         ret = f"""
# ## After Baidu search for key words: **{keywords}**
# Retrival results: 
# {ret}
# Agent's answer:
# {response}
# """
#         return ret
    
#     @classmethod
#     def content(cls):
#         return "Extracts keywords from queries and uses Baidu search to find relevant information"