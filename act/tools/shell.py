# -*- coding: utf-8 -*-
# pylint: disable=unused-argument
"""Borrowed and modified the shell command tool from agentscope."""

import asyncio
import platform
from typing import Any
import re

from act.tools.tool import Tool
from agent_model.utils import extract_dict_from_text

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

        # return f"<returncode>{returncode}</returncode>", f"<stdout>{stdout_str}</stdout>", f"<stderr>{stderr_str}</stderr>"
        return returncode, stdout_str, stderr_str

def extract_command_from_response(response):
    response_dict = extract_dict_from_text(response)
    command = response_dict['command']
    return command

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
[SHELL COMMAND GENERATION GUIDELINES]
Generate a shell command and perform it to complete the currentgoal, consider: 
{system_specific_instruction}

## OUTPUT FORMAT:
Output STRICTLY according to this JSON Schema:
{{
    "command": "string"
}}

Return only valid JSON.

[USER]
Query:
{query}

[ASSIATANT]
/no_think
"""
    response = agent.response(prompt, stream=False)
    command = extract_command_from_response(response)

    return command


class Shell(Tool):
    def execute(self, agent, query):
        # 获取当前操作系统
        os_name = platform.system()

        # 让agent根据查询和操作系统生成适当的shell命令
        command = generate_shell_command(query, agent, os_name)
        print("Generated command:", command)
        # 执行生成的命令
        return_code, std_out, std_eer = asyncio.run(execute_shell_command(command))
        if return_code == -1:
            return {
                "success": False,
                "response": std_out+std_eer
            }
        return {
                "success": True,
                "response": f"Shell command:\"{command}\" succeed.\nCommand result:\nstd_out:\n{std_out}\nstd_error:\n{std_eer}"
            }

    @classmethod
    def content(cls):
        # return "Perform the shell commands based on the query and operating system, like: \"python3 test.py\", \"ls\""
        return """
Function: Perform shell commands based on the query and operating system
Method: [
    Generate appropriate shell command for the given query and OS,
    Execute the command with timeout protection,
    Capture return code, stdout and stderr
]
Return: [
    standard output of the command,
    standard error of the command
]
"""