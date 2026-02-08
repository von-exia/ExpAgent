# -*- coding: utf-8 -*-
# flake8: noqa: E501
# pylint: disable=line-too-long
"""Borrowed and modified the view text file tool from agentscope."""
import os
import re
from act.tools.tool import Tool
from agent_model.utils import extract_dict_from_text


def _assert_ranges(
    ranges: list[int],
) -> None:
    """Check if the ranges are valid.

    Raises:
        ToolInvalidArgumentsError: If the ranges are invalid.
    """
    if (
        isinstance(ranges, list)
        and len(ranges) == 2
        and all(isinstance(i, int) for i in ranges)
    ):
        start, end = ranges
        if start > end:
            raise ValueError(f"InvalidArgumentError: The start line is greater than the " +
                f"end line in the given range {ranges}.")
    else:
        raise ValueError(f"InvalidArgumentError: Invalid range format. Expected a list of " +
            f"two integers, but got {ranges}.")


def _view_text_file(
    file_path: str,
    ranges: list[int] | None = None,
) -> str:
    """Return the file content in the specified range with line numbers."""
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    if ranges:
        _assert_ranges(ranges)
        start, end = ranges

        if start > len(lines):
            raise ValueError(f"InvalidArgumentError: The range '{ranges}' is out of bounds " +
                f"for the file '{file_path}', which has only {len(lines)} " +
                f"lines.")

        # Handle negative indices for end of file
        if start < 0:
            start = len(lines) + start + 1
        if end < 0:
            end = len(lines) + end + 1

        # Adjust start and end to be within valid range
        start = max(1, start)
        end = min(len(lines), end)

        view_content = [
            f"{index + start}: {line}"
            for index, line in enumerate(lines[start - 1 : end])
        ]

        return "".join(view_content)

    return "".join(f"{index + 1}: {line}" for index, line in enumerate(lines))


class ViewTextFile(Tool):
    """A tool to view the content of a text file with line numbers."""

    def __init__(self):
        self._init_prompt()

    def _init_prompt(self):
        self.key_prompt = """
[EXTRACTION GUIDELINES]
Extract file path and ranges from the query. Follow these principles:
1. Identify the file path from the query
2. If line ranges are specified, extract them as a list of two integers
3. Return the extracted information in JSON format

## OUTPUT FORMAT:
Output STRICTLY according to this JSON Schema:
{{
    "file_path": "string",
    "ranges": ["int", "int"] | null
}}

Return only valid JSON.

[USER]
Query:
{query}

[ASSISTANT]
/no_think
"""

    def extract_arg_from_response(self, response):
        response_dict = extract_dict_from_text(response)
        file_path = response_dict['file_path']
        ranges = response_dict['ranges']
        return file_path, ranges


    def execute(self, agent, query: str) -> str:
        """
        Execute the view text file operation based on the query.

        Args:
            agent: The agent object (not used in this tool).
            query (str): The query containing the file path and optional line ranges.

        Returns:
            str: The content of the file or an error message.
        """
        # Extract file path and ranges from the query using JSON format
        key_prompt = self.key_prompt.format(query=query)
        response = agent.response(key_prompt, stream=False)
        file_path, ranges = self.extract_arg_from_response(response)

        # Validate file path
        if not os.path.exists(file_path):
            return {"success": False, "response": f"Error: The file {file_path} does not exist."}
        if not os.path.isfile(file_path):
            return {"success": False, "response": f"Error: The path {file_path} is not a file."}

        try:
            content = _view_text_file(file_path, ranges)
        except Exception as e:
            return {"success": False, "response": f"Error: {str(e)}"}

        if ranges is None:
            return {"success": True,
                   "response": f"""The content of {file_path}:
```
{content}```"""}
        else:
            return {"success": True,
                   "response": f"""The content of {file_path} in {ranges} lines:
```
{content}```"""}

    @classmethod
    def content(cls):
        return """
Function: View the content of a text file with line numbers
Method: [
    Extract file path and ranges from query,
    Read the file content with specified ranges,
    Return the content with line numbers
]
Return: File content with line numbers
"""