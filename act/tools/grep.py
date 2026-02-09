# -*- coding: utf-8 -*-
# pylint: disable=unused-argument
"""Grep tool for searching patterns in files."""

import asyncio
import platform
from typing import Any
import re
import os

from act.tools.tool import Tool
from agent_model.utils import extract_dict_from_text


async def execute_grep_search(
    pattern: str,
    path: str = ".",
    glob: str = None,
    limit: int = None,
    **kwargs: Any,
):
    """Execute grep search and return the matched lines within <matches></matches> tags.

    Args:
        pattern (`str`):
            The regex pattern to search for.
        path (`str`, defaults to `"."`):
            The file or directory to search in.
        glob (`str`, optional):
            Glob pattern to filter files (e.g. "*.js", "*.{ts,tsx}").
        limit (`int`, optional):
            Limit output to first N lines/entries.

    Returns:
        `tuple`: A tuple containing success status, matched lines, and error message.
    """
    
    # Determine the operating system
    os_name = platform.system()
    
    try:
        if os_name.lower() == "windows":
            # For Windows, we'll use a Python-based approach since Windows doesn't have grep by default
            matches = []
            
            # Handle glob pattern if provided
            import fnmatch
            import pathlib
            
            search_path = pathlib.Path(path)
            
            # Determine files to search
            if glob:
                # If glob is provided, find all matching files
                if search_path.is_dir():
                    files_to_search = [f for f in search_path.rglob('*') if f.is_file() and fnmatch.fnmatch(f.name, glob)]
                else:
                    files_to_search = [search_path] if fnmatch.fnmatch(search_path.name, glob) else []
            else:
                # If no glob, just search in the specified path
                if search_path.is_dir():
                    files_to_search = [f for f in search_path.rglob('*') if f.is_file()]
                else:
                    files_to_search = [search_path] if search_path.exists() else []
            
            # Search for the pattern in each file
            for file_path in files_to_search:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        for line_num, line in enumerate(lines, start=1):
                            if re.search(pattern, line):
                                matches.append(f"{file_path}:{line_num}: {line.rstrip()}")
                                
                                # Apply limit if specified
                                if limit and len(matches) >= limit:
                                    break
                        if limit and len(matches) >= limit:
                            break
                except Exception as e:
                    # Skip files that can't be read
                    continue
            
            matches_str = "\n".join(matches[:limit]) if limit else "\n".join(matches)
            return True, matches_str, ""
            
        else:
            # For Unix-like systems, use the grep command
            cmd_parts = ["grep", "-rn", "--color=never"]
            
            if glob:
                cmd_parts.extend(["--include", glob])
                
            if limit:
                cmd_parts.append(f"--max-count={limit}")
                
            cmd_parts.extend([pattern, path])
            
            command = " ".join(cmd_parts)
            
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                bufsize=0,
            )

            try:
                stdout, stderr = await proc.communicate()
                try:
                    stdout_str = stdout.decode("utf-8")
                    stderr_str = stderr.decode("utf-8")
                except Exception as e:
                    print("Decoding error with UTF-8:", e)
                    print("Attempting to decode with GBK encoding for compatibility.")
                    stdout_str = stdout.decode("gbk")
                    stderr_str = stderr.decode("gbk")
                    
                return proc.returncode == 0 or bool(stdout_str.strip()), stdout_str.strip(), stderr_str.strip()

            except Exception as e:
                return False, "", str(e)
                
    except Exception as e:
        return False, "", str(e)


def extract_search_params_from_response(response):
    response_dict = extract_dict_from_text(response)
    pattern = response_dict.get('pattern', '')
    path = response_dict.get('path', '.')
    glob = response_dict.get('glob', None)
    limit = response_dict.get('limit', None)
    
    return pattern, path, glob, limit


def generate_grep_query(query: str, agent, os_name: str) -> tuple:
    """
    根据自然语言查询和操作系统生成适当的grep搜索参数

    Args:
        query: 自然语言查询
        agent: agent对象，用于生成命令
        os_name: 操作系统名称

    Returns:
        生成的搜索参数元组 (pattern, path, glob, limit)
    """
    # 根据操作系统提供特定的指令
    if os_name.lower() == "windows":
        system_specific_instruction = (
            "For Windows systems, use Python-based file searching. "
            "The pattern should be a valid regex. "
            "The path specifies the directory or file to search in. "
            "The glob parameter filters files by pattern (e.g., '*.txt'). "
            "The limit parameter restricts the number of results."
        )
    else:
        system_specific_instruction = (
            "For Unix-like systems (Linux/macOS), use grep command. "
            "The pattern should be a valid regex. "
            "The path specifies the directory or file to search in. "
            "The glob parameter filters files by pattern (e.g., '*.txt'). "
            "The limit parameter restricts the number of results."
        )

    # 构建提示词，让agent生成适当的搜索参数
    prompt = f"""
[GREP SEARCH PARAMETER GENERATION GUIDELINES]
Generate appropriate parameters for grep search to complete the current goal, consider:
{system_specific_instruction}

## OUTPUT FORMAT:
Output STRICTLY according to this JSON Schema:
{{
    "pattern": "string",
    "path": "string",
    "glob": "string or null",
    "limit": "integer or null"
}}

Return only valid JSON.

[USER]
Query:
{query}

[ASSISTANT]
/no_think
"""
    response = agent.response(prompt, stream=False)
    pattern, path, glob, limit = extract_search_params_from_response(response)

    return pattern, path, glob, limit


class Grep(Tool):
    def execute(self, agent, query):
        # 获取当前操作系统
        os_name = platform.system()

        # 让agent根据查询生成适当的搜索参数
        pattern, path, glob, limit = generate_grep_query(query, agent, os_name)
        
        print(f"Searching for pattern: '{pattern}' in path: '{path}', glob: '{glob}', limit: {limit}")
        
        # 执行搜索
        success, matches, error = asyncio.run(
            execute_grep_search(pattern, path, glob, limit)
        )
        
        if not success and error:
            return {
                "success": False,
                "response": f"Grep search failed with error: {error}"
            }
        
        if not matches:
            return {
                "success": True,
                "response": f"No matches found for pattern '{pattern}' in '{path}'"
            }
        
        return {
            "success": True,
            "response": f"Grep search for pattern '{pattern}' in '{path}' succeeded.\nMatches:\n{matches}"
        }

    @classmethod
    def content(cls):
        return """
Function: Search for patterns in files using grep functionality
Method: [
    Parse the query to extract search pattern, path, file glob pattern, and result limit,
    Execute pattern search in specified path with optional file filtering,
    Return matched lines with file names and line numbers
]
Return: [
    matched lines with file names and line numbers,
    error message if search fails
]
"""