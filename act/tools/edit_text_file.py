# -*- coding: utf-8 -*-
# flake8: noqa: E501
# pylint: disable=line-too-long
"""A tool to edit content in a text file in-place."""
import os
import asyncio
from act.tools.tool import Tool
from agent_model.utils import extract_dict_from_text
from act.tools.view_text_file import _view_text_file
import logging

def _calculate_view_ranges(
    old_n_lines: int,
    new_n_lines: int,
    start: int,
    end: int,
    extra_view_n_lines: int = 5,
) -> tuple[int, int]:
    """Calculate after writing the new content, the view ranges of the file.

    Args:
        old_n_lines (`int`):
            The number of lines before writing the new content.
        new_n_lines (`int`):
            The number of lines after writing the new content.
        start (`int`):
            The start line of the writing range.
        end (`int`):
            The end line of the writing range.
        extra_view_n_lines (`int`, optional):
            The number of extra lines to view before and after the range.
    """

    view_start = max(1, start - extra_view_n_lines)

    delta_lines = new_n_lines - old_n_lines
    view_end = min(end + delta_lines + extra_view_n_lines, new_n_lines)

    return view_start, view_end


async def edit_text_file(
    file_path: str,
    start_line: int,
    end_line: int,
    new_content: str,
):
    """Edit content in a text file by replacing lines in the specified range with new content.

    Args:
        file_path (`str`):
            The target file path.
        start_line (`int`):
            The starting line number (1-indexed) of the range to be replaced.
        end_line (`int`):
            The ending line number (1-indexed, inclusive) of the range to be replaced.
        new_content (`str`):
            The new content to replace the specified line range.

    Returns:
        `dict`:
            A dictionary containing success status and response message.
    """
    if not os.path.exists(file_path):
        return {
            "success": False,
            "response": f"InvalidArgumentsError: The target file {file_path} does not exist."
        }

    with open(file_path, "r", encoding="utf-8") as file:
        original_lines = file.readlines()

    total_lines = len(original_lines)

    # Validate line range
    # Allow start_line to be 0 for inserting at the beginning of file
    # Allow start_line to be total_lines + 1 for appending content at the end of file
    if start_line < 0 or start_line > total_lines + 1:
        return {
            "success": False,
            "response": f"InvalidArgumentsError: start_line {start_line} is out of range. File has {total_lines} lines."
        }

    # When appending (start_line == total_lines + 1), end_line should equal start_line
    # When inserting at beginning (start_line == 0), end_line should be 0 or 1
    # Otherwise, end_line must be within the file range
    if start_line == total_lines + 1:
        if end_line < start_line:
            return {
                "success": False,
                "response": f"InvalidArgumentsError: end_line {end_line} must be >= start_line {start_line} when appending."
            }
    elif start_line == 0:
        # Inserting at the beginning: end_line can be 0 (insert before line 1) or 1 (replace line 1)
        if end_line < 0 or end_line > total_lines:
            return {
                "success": False,
                "response": f"InvalidArgumentsError: end_line {end_line} is out of range. File has {total_lines} lines."
            }
    else:
        if end_line < start_line or end_line > total_lines:
            return {
                "success": False,
                "response": f"InvalidArgumentsError: end_line {end_line} is out of range. File has {total_lines} lines."
            }

    # Build new content by replacing the specified line range
    # Lines before the range + new content + lines after the range
    # When start_line is 0, lines_before is empty (insert at beginning)
    lines_before = original_lines[:start_line - 1] if start_line > 0 else []
    lines_after = original_lines[end_line:]
    
    # Ensure new_content ends with newline if there are lines after it
    new_content_lines = new_content.splitlines(keepends=True)
    if lines_after and new_content_lines and not new_content_lines[-1].endswith('\n'):
        new_content_lines[-1] = new_content_lines[-1] + '\n'
    
    new_lines = lines_before + new_content_lines + lines_after
    new_file_content = ''.join(new_lines)

    # Write the new content to the file
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(new_file_content)

    # Calculate view ranges to show the edited section
    view_start, view_end = _calculate_view_ranges(
        total_lines,
        len(new_lines),
        start_line,
        end_line,
        extra_view_n_lines=5,
    )

    show_content = _view_text_file(file_path, [view_start, view_end])

    return {
        "success": True,
        "response": f"Edit content in {file_path} successfully. "
                   f"The content between lines {view_start}-{view_end} is:\n"
                   f"```\n{show_content}```"
    }


class EditTextFile(Tool):
    """A tool to edit content in a text file in-place by replacing old content with new content."""

    def __init__(self):
        self._init_prompt()

    def _init_prompt(self):
        self.edit_prompt = """
[EXTRACTION GUIDELINES]
Extract file path, line range, and new content from the query. Follow these principles:
1. Identify the file path from the query
2. Extract the starting line number (start_line) and ending line number (end_line) of the range to be replaced, like: [1, 2]
3. Extract the new content that will replace the specified line range (new_content)

[USER]
Query:
{query}

## OUTPUT FORMAT:
Output STRICTLY according to this JSON Schema:
{{
    "file_path": "string",
    "edit_range": ["integer", "integer"],
    "new_content": "string"
}}

Return only valid JSON.

[ASSISTANT]
/no_think
"""

    def extract_kw_from_response(self, response):
        response_dict = extract_dict_from_text(response)
        file_path = response_dict.get('file_path')
        edit_range = response_dict.get('edit_range')
        new_content = response_dict.get('new_content')
        return file_path, edit_range[0], edit_range[1], new_content

    def execute(self, agent, query: str) -> str:
        """
        Execute the edit text file operation based on the query.

        Args:
            agent: The agent object (not used in this tool).
            query (str): The query containing the file path, line range, and new content.

        Returns:
            str: The result of the edit operation or an error message.
        """
        # Extract file path, line range, and new content from the query using JSON format
        key_prompt = self.edit_prompt.format(query=query)
        response = agent.response(key_prompt, stream=False)
        file_path, start_line, end_line, new_content = self.extract_kw_from_response(response)

        # Validate file path
        if not file_path:
            return {
                "success": False,
                "response": f"Error: File path is empty."
            }

        # Validate line range
        if not start_line or not end_line:
            return {
                "success": False,
                "response": f"Error: Line range (start_line/end_line) is not specified."
            }

        # Validate new content
        if not new_content:
            return {
                "success": False,
                "response": f"Error: New content is empty."
            }

        try:
            # Call edit_text_file to perform the edit operation
            result = asyncio.run(edit_text_file(file_path, start_line, end_line, new_content))
            return result
        except Exception as e:
            return {
                "success": False,
                "response": f"Error: {str(e)}"
            }

    @classmethod
    def content(cls):
        return """
Function: Edit (modify) content in an existing text file in-place by replacing lines in a specified range with new content
Method: [
    Extract an existing file path, line range (start_line, end_line), and new content from query,
    Replace the lines in the specified range with new content in the specified file,
    Return the result of the edit operation
]
Return: Result of the edit operation
"""