# -*- coding: utf-8 -*-
# flake8: noqa: E501
# pylint: disable=line-too-long
"""A tool to edit content in a text file in-place."""
import os
import asyncio
from act.tools.tool import Tool
from agent_model.utils import extract_dict_from_text
from act.tools.write_text_file import _calculate_view_ranges, _view_text_file


async def edit_text_file(
    file_path: str,
    old_content: str,
    new_content: str,
    exact_match: bool = True,
):
    """Edit content in a text file by replacing old content with new content.

    Args:
        file_path (`str`):
            The target file path.
        old_content (`str`):
            The content to be replaced.
        new_content (`str`):
            The new content to replace the old content.
        exact_match (`bool`, optional):
            Whether to perform an exact match for the content to be replaced.
            If True, will replace only exact matches of old_content.
            If False, will replace content based on position/range information.

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
        original_content = file.read()
        original_lines = file.readlines()

    # Find the position of old_content in the file
    if exact_match:
        if old_content not in original_content:
            return {
                "success": False,
                "response": f"Error: The content to be replaced was not found in {file_path}."
            }

        # Replace the first occurrence of old_content with new_content
        new_file_content = original_content.replace(old_content, new_content, 1)

        # Calculate the line numbers where the change occurred
        old_start_pos = original_content.find(old_content)
        lines_before_change = original_content[:old_start_pos].count('\n')
        start_line = lines_before_change + 1
        
        # Count lines in the old content to determine end line
        old_line_count = old_content.count('\n') + 1
        end_line = start_line + old_line_count - 1
    else:
        # If not exact match, we'll need to implement position-based editing
        # For now, we'll use the same approach as exact match
        if old_content not in original_content:
            return {
                "success": False,
                "response": f"Error: The content to be replaced was not found in {file_path}."
            }

        new_file_content = original_content.replace(old_content, new_content, 1)
        
        old_start_pos = original_content.find(old_content)
        lines_before_change = original_content[:old_start_pos].count('\n')
        start_line = lines_before_change + 1
        old_line_count = old_content.count('\n') + 1
        end_line = start_line + old_line_count - 1

    # Write the new content to the file
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(new_file_content)

    # Read the file again to get the new lines for calculating view ranges
    with open(file_path, "r", encoding="utf-8") as file:
        new_lines = file.readlines()

    # Calculate view ranges to show the edited section
    view_start, view_end = _calculate_view_ranges(
        len(original_lines),
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
Extract file path, old content, and new content from the query. Follow these principles:
1. Identify the file path from the query
2. Extract the content that needs to be replaced (old_content)
3. Extract the new content that will replace the old content (new_content)
4. Determine if an exact match is required for the replacement
5. Return the extracted information in JSON format

## OUTPUT FORMAT:
Output STRICTLY according to this JSON Schema:
{{
    "file_path": "string",
    "old_content": "string",
    "new_content": "string",
    "exact_match": "boolean"
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
        file_path = response_dict.get('file_path')
        old_content = response_dict.get('old_content')
        new_content = response_dict.get('new_content')
        exact_match = response_dict.get('exact_match', True)
        return file_path, old_content, new_content, exact_match

    def execute(self, agent, query: str) -> str:
        """
        Execute the edit text file operation based on the query.

        Args:
            agent: The agent object (not used in this tool).
            query (str): The query containing the file path, old content, and new content.

        Returns:
            str: The result of the edit operation or an error message.
        """
        # Extract file path, old content, new content, and exact_match flag from the query using JSON format
        key_prompt = self.edit_prompt.format(query=query)
        response = agent.response(key_prompt, stream=False)
        file_path, old_content, new_content, exact_match = self.extract_kw_from_response(response)

        # Validate file path
        if not file_path:
            return {
                "success": False,
                "response": f"Error: File path is empty."
            }

        # Validate old and new content
        if not old_content:
            return {
                "success": False,
                "response": f"Error: Old content is empty."
            }

        try:
            # Call edit_text_file to perform the edit operation
            result = asyncio.run(edit_text_file(file_path, old_content, new_content, exact_match))
            return result
        except Exception as e:
            return {
                "success": False,
                "response": f"Error: {str(e)}"
            }

    @classmethod
    def content(cls):
        return """
Function: Edit content in a text file in-place by replacing old content with new content
Method: [
    Extract file path, old content, new content, and exact_match flag from query,
    Replace the old content with new content in the specified file,
    Return the result of the edit operation
]
Return: Result of the edit operation
"""