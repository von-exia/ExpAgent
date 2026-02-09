# -*- coding: utf-8 -*-
# pylint: disable=unused-argument
"""Glob tool for finding files matching glob patterns."""

import asyncio
import platform
from typing import Any
import os
import glob as py_glob
import pathlib

from act.tools.tool import Tool
from agent_model.utils import extract_dict_from_text


def execute_glob_search(
    pattern: str,
    path: str = ".",
    recursive: bool = True,
    **kwargs: Any,
):
    """Execute glob search and return the matched files within <matches></matches> tags.

    Args:
        pattern (`str`):
            The glob pattern to search for (e.g. "*.py", "**/*.js").
        path (`str`, defaults to `"."`):
            The directory to search in.
        recursive (`bool`, defaults to `True`):
            Whether to search recursively in subdirectories.

    Returns:
        `tuple`: A tuple containing success status, matched files, and error message.
    """
    
    try:
        # Convert path to pathlib.Path object
        search_path = pathlib.Path(path)
        
        if not search_path.exists():
            return False, "", f"Path does not exist: {path}"
        
        if not search_path.is_dir():
            return False, "", f"Path is not a directory: {path}"
        
        # Construct the full glob pattern
        if recursive and "**" not in pattern:
            full_pattern = str(search_path / "**" / pattern)
        else:
            full_pattern = str(search_path / pattern)
        
        # Use glob to find matching files
        if recursive:
            # Use rglob for recursive search
            if pattern.startswith("**/"):
                # Pattern already includes recursive part
                matches = list(search_path.rglob(pattern[3:]))  # Remove "**/" prefix
            else:
                matches = list(search_path.rglob(pattern))
        else:
            # Use glob for non-recursive search
            matches = list(search_path.glob(pattern))
        
        # Convert Path objects to strings
        matches_str = "\n".join([str(match) for match in matches])
        
        return True, matches_str, ""
        
    except Exception as e:
        return False, "", str(e)


def extract_glob_params_from_response(response):
    response_dict = extract_dict_from_text(response)
    pattern = response_dict.get('pattern', '*')
    path = response_dict.get('path', '.')
    recursive = response_dict.get('recursive', True)
    
    return pattern, path, recursive


def generate_glob_query(query: str, agent, os_name: str) -> tuple:
    """
    根据自然语言查询和操作系统生成适当的glob搜索参数

    Args:
        query: 自然语言查询
        agent: agent对象，用于生成命令
        os_name: 操作系统名称

    Returns:
        生成的搜索参数元组 (pattern, path, recursive)
    """
    # 根据操作系统提供特定的指令
    if os_name.lower() == "windows":
        system_specific_instruction = (
            "For Windows systems, use Python glob functionality. "
            "The pattern should be a valid glob pattern (e.g., '*.txt', '**/*.py'). "
            "The path specifies the directory to search in. "
            "Recursive determines whether to search subdirectories."
        )
    else:
        system_specific_instruction = (
            "For Unix-like systems (Linux/macOS), use Python glob functionality. "
            "The pattern should be a valid glob pattern (e.g., '*.txt', '**/*.py'). "
            "The path specifies the directory to search in. "
            "Recursive determines whether to search subdirectories."
        )

    # 构建提示词，让agent生成适当的搜索参数
    prompt = f"""
[GLOB SEARCH PARAMETER GENERATION GUIDELINES]
Generate appropriate parameters for glob search to complete the current goal, consider:
{system_specific_instruction}

## OUTPUT FORMAT:
Output STRICTLY according to this JSON Schema:
{{
    "pattern": "string",
    "path": "string",
    "recursive": "boolean"
}}

Return only valid JSON.

[USER]
Query:
{query}

[ASSISTANT]
/no_think
"""
    response = agent.response(prompt, stream=False)
    pattern, path, recursive = extract_glob_params_from_response(response)

    return pattern, path, recursive


class Glob(Tool):
    def execute(self, agent, query):
        # 获取当前操作系统
        os_name = platform.system()

        # 让agent根据查询生成适当的搜索参数
        pattern, path, recursive = generate_glob_query(query, agent, os_name)
        
        print(f"Searching for files with pattern: '{pattern}' in path: '{path}', recursive: {recursive}")
        
        # 执行搜索
        success, matches, error = execute_glob_search(pattern, path, recursive)
        
        if not success and error:
            return {
                "success": False,
                "response": f"Glob search failed with error: {error}"
            }
        
        if not matches:
            return {
                "success": True,
                "response": f"No files found matching pattern '{pattern}' in '{path}'"
            }
        
        return {
            "success": True,
            "response": f"Glob search for pattern '{pattern}' in '{path}' succeeded.\nMatching files:\n{matches}"
        }

    @classmethod
    def content(cls):
        return """
Function: Find files matching glob patterns
Method: [
    Parse the query to extract glob pattern, search path, and recursive flag,
    Execute file search using glob patterns in specified path,
    Return matched file paths
]
Return: [
    matched file paths,
    error message if search fails
]
"""