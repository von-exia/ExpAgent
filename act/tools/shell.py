# -*- coding: utf-8 -*-
# pylint: disable=unused-argument
"""The shell command tool in agentscope."""

import asyncio
import platform
from typing import Any
import re

from act.tools.tool import Tool

async def execute_shell_command(
        command: str,
        timeout: int = 300,
        **kwargs: Any,
    ):
        """Execute given command and return the return code, standard output and
        error within <returncode></returncode>, <stdout></stdout> and
        <stderr></stderr> tags.

        Args:
            command (`str`):
                The shell command to execute.
            timeout (`float`, defaults to `300`):
                The maximum time (in seconds) allowed for the command to run.

        Returns:
            `ToolResponse`:
                The tool response containing the return code, standard output, and
                standard error of the executed command.
        """

        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            bufsize=0,
        )

        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout)
            stdout, stderr = await proc.communicate()
            try:
                stdout_str = stdout.decode("utf-8")
                stderr_str = stderr.decode("utf-8")
            except Exception as e:
                print("Decoding error with UTF-8:", e)
                print("Attempting to decode with GBK encoding for Windows.")
                stdout_str = stdout.decode("gbk")
                stderr_str = stderr.decode("gbk")
            returncode = proc.returncode

        except asyncio.TimeoutError:
            stderr_suffix = (
                f"TimeoutError: The command execution exceeded "
                f"the timeout of {timeout} seconds."
            )
            returncode = -1
            try:
                proc.terminate()
                stdout, stderr = await proc.communicate()
                stdout_str = stdout.decode("utf-8")
                stderr_str = stderr.decode("utf-8")
                if stderr_str:
                    stderr_str += f"\n{stderr_suffix}"
                else:
                    stderr_str = stderr_suffix
            except ProcessLookupError:
                stdout_str = ""
                stderr_str = stderr_suffix

        return f"<returncode>{returncode}</returncode>" + f"<stdout>{stdout_str}</stdout>" + f"<stderr>{stderr_str}</stderr>"


def generate_shell_command(query: str, agent, os_name: str) -> str:
    """
    根据自然语言查询和操作系统生成适当的shell命令

    Args:
        query: 自然语言查询
        agent: agent对象，用于生成命令
        os_name: 操作系统名称

    Returns:
        生成的shell命令字符串
    """
    # 根据操作系统选择合适的命令格式
    if os_name.lower() == "windows":
        system_specific_instruction = (
            "For Windows systems, use PowerShell or cmd commands. "
            "Use 'dir' for listing files, 'where' for finding executables, "
            "and 'type' for displaying file contents. Use 'ping' for network checks."
        )
    else:
        system_specific_instruction = (
            "For Unix-like systems (Linux/macOS), use bash commands. "
            "Use 'ls' for listing files, 'which' or 'whereis' for finding executables, "
            "and 'cat' for displaying file contents. Use 'ping' for network checks."
        )

    # 构建提示词，让agent生成适当的shell命令
    prompt = f"""
Generate a shell command to perform the following task: {query}
{system_specific_instruction}
Only output the command without any explanation or additional text. 
Command:"""

    # 使用agent生成命令
    # 注意：这里假设agent有response方法，根据实际agent接口调整
    try:
        # 尝试使用agent生成命令
        command = agent.response(prompt+ "\n/no_think", stream=False).strip()
        def get_last_command(text):
            """
            从字符串中提取最后一个命令，处理多种格式
            """
            # 移除<think>标签及其内容
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
            
            # 移除多余空白字符
            text = re.sub(r'\s+', ' ', text).strip()
            
            # 如果字符串为空
            if not text:
                return ""
            
            # 分割命令
            commands = re.split(r'\s*&&\s*', text)
            
            # 返回最后一个命令
            return commands[-1]
        command = get_last_command(command)
    except Exception as e:
        # 如果agent调用失败，尝试直接解释查询
        command = e

    return command.strip()


class Shell(Tool):
    def execute(self, agent, query):
        # 获取当前操作系统
        os_name = platform.system()

        # 让agent根据查询和操作系统生成适当的shell命令
        command = generate_shell_command(query, agent, os_name)
        print("Generated command:", command)
        # 执行生成的命令
        return asyncio.run(execute_shell_command(command))

    @classmethod
    def content(cls):
        return "Perform shell commands, based on the query and the operating system"