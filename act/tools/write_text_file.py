# -*- coding: utf-8 -*-
# flake8: noqa: E501
# pylint: disable=line-too-long
"""Borrowed and modified the write text file tool from agentscope."""
import os
import asyncio
from act.tools.tool import Tool
from agent_model.utils import extract_dict_from_text


async def write_text_file(
    file_path: str,
    content: str,
):
    """Create a new text file and write content to it.

    Args:
        file_path (`str`):
            The target file path.
        content (`str`):
            The content to be written.

    Returns:
        `ToolResponse`:
            The tool response containing the result of the writing operation.
    """

    if os.path.exists(file_path):
        return {"success": False, "response": f"FileExistsError: The file {file_path} already exists. "}

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)

    return {"success": True, "response": f"Create and write {file_path} successfully. The new content is:\n{content}\n"}


class WriteTextFile(Tool):
    """A tool to create a new text file and write content to it."""

    def __init__(self):
        self._init_prompt()

    def _init_prompt(self):
        self.write_prompt = """
[EXTRACTION GUIDELINES]
Extract file path and content from the query. Follow these principles:
1. Identify the file path from the query
2. Extract the content to be written
3. Return the extracted information in JSON format

## OUTPUT FORMAT:
Output STRICTLY according to this JSON Schema:
{{
    "file_path": "string",
    "content": "string"
}}

Return only valid JSON.

[USER]
Query:
{query}

[ASSISTANT]
/no_think
"""

    def extract_kw_from_response(self, response):
        response_dict = extract_dict_from_text(response)
        file_path = response_dict['file_path']
        content = response_dict['content']
        return file_path, content

    def execute(self, agent, query: str) -> str:
        """
        Execute the write text file operation based on the query.

        Args:
            agent: The agent object (not used in this tool).
            query (str): The query containing the file path and content.

        Returns:
            str: The result of the write operation or an error message.
        """
        # Extract file path and content from the query using JSON format
        key_prompt = self.write_prompt.format(query=query)
        response = agent.response(key_prompt, stream=False)
        file_path, content = self.extract_kw_from_response(response)

        # Validate file path
        if not file_path:
            return {"success": False, "response": f"Error: File path is empty."}

        try:
            result = asyncio.run(write_text_file(file_path, content))
            return result
        except Exception as e:
            return {"success": False, "response": f"Error: {str(e)}"}

    @classmethod
    def content(cls):
        return """
Function: Create a new text file and write content to it
Method: [
    Determine the file path and content from query,
    Create a new file using 'w' mode and write the content,
    Return the result of the operation
]
Return: Result of the write operation
"""