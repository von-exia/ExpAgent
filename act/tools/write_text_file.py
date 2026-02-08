# -*- coding: utf-8 -*-
# flake8: noqa: E501
# pylint: disable=line-too-long
"""Borrowed and modified the write text file tool from agentscope."""
import os
import re
import asyncio
from act.tools.tool import Tool
from agent_model.utils import extract_dict_from_text


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

        view_content = [
            f"{index + start}: {line}"
            for index, line in enumerate(lines[start - 1 : end])
        ]

        return "".join(view_content)

    return "".join(f"{index + 1}: {line}" for index, line in enumerate(lines))


async def insert_text_file(
    file_path: str,
    content: str,
    line_number: int,
):
    """Insert the content at the specified line number in a text file.

    Args:
        file_path (`str`):
            The target file path.
        content (`str`):
            The content to be inserted.
        line_number (`int`):
            The line number at which the content should be inserted, starting
            from 1. If exceeds the number of lines in the file, it will be
            appended to the end of the file.

    Returns:
        `ToolResponse`:
            The tool response containing the result of the insertion operation.
    """
    if line_number <= 0:
        return {"success": False, "response": f"InvalidArgumentsError: " + \
                f"The line number {line_number} is invalid. "}

    if not os.path.exists(file_path):
        return {"success": False, "response": f"InvalidArgumentsError: The target file " +\
                    f"{file_path} does not exist. "}

    with open(file_path, "r", encoding="utf-8") as file:
        original_lines = file.readlines()

    if line_number == len(original_lines) + 1:
        new_lines = original_lines + ["\n" + content]
    elif line_number < len(original_lines) + 1:
        new_lines = (
            original_lines[: line_number - 1]
            + [content + "\n"]
            + original_lines[line_number - 1 :]
        )
    else:
        return {"success": False, "response": "InvalidArgumentsError: The given line_number " +\
                f"({line_number}) is not in the valid range " +\
                f"[1, {len(original_lines) + 1}]."}

    with open(file_path, "w", encoding="utf-8") as file:
        file.writelines(new_lines)

    with open(file_path, "r", encoding="utf-8") as file:
        new_lines = file.readlines()

    start, end = _calculate_view_ranges(
        len(original_lines),
        len(new_lines),
        line_number,
        line_number,
        extra_view_n_lines=5,
    )

    show_content = _view_text_file(file_path, [start, end])

    return {"success": True, "response": f"Insert content into {file_path} at line " +\
            f"{line_number} successfully. The new content " +\
            f"between lines {start}-{end} is:\n" +\
            f"```\n{show_content}```"}


async def append_text_file(
    file_path: str,
    content: str,
):
    """Append content to the end of a text file.

    Args:
        file_path (`str`):
            The target file path.
        content (`str`):
            The content to be appended.

    Returns:
        `ToolResponse`:
            The tool response containing the result of the append operation.
    """
    
    with open(file_path, "a", encoding="utf-8") as file:
        file.write("\n" + content)

    return {"success": True, "response": f"Append to {file_path} successfully. The new content is:\n{content}\n"}


async def write_text_file(
    file_path: str,
    content: str,
    ranges: None | list[int] = None,
):
    """Create/Replace/Overwrite content in a text file. When `ranges` is provided, the content will be replaced in the specified range. Otherwise, the entire file will be overwritten.

    Args:
        file_path (`str`):
            The target file path.
        content (`str`):
            The content to be written.
        ranges (`list[int] | None`, defaults to `None`):
            The range of lines to be replaced. If `None`, the entire file will
            be overwritten.

    Returns:
        `ToolResponse`:
            The tool response containing the result of the writing operation.
    """

    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)

        if ranges:
            return {"success": True, "response": f"Create and write {file_path} successfully. " +\
                    f"The ranges {ranges} is ignored because the " +\
                    f"file does not exist."}

        return {"success": True, "response": f"Create and write {file_path} successfully. The new content is {content}"}

    with open(file_path, "r", encoding="utf-8") as file:
        original_lines = file.readlines()

    if ranges is not None:
        if (
            isinstance(ranges, list)
            and len(ranges) == 2
            and all(isinstance(i, int) for i in ranges)
        ):
            # Replace content in the specified range
            start, end = ranges
            if start > len(original_lines):
                return {"success": False, "response": f"Error: The start line {start} is invalid. " +\
                        f"The file only has {len(original_lines)} " +\
                        f"lines."}

            new_content = (
                original_lines[: start - 1]
                + [
                    content,
                ]
                + original_lines[end:]
            )

            with open(file_path, "w", encoding="utf-8") as file:
                file.write("".join(new_content))

            # The written content may contain multiple "\n", to avoid mis
            # counting the lines, we read the file again to get the new content
            with open(file_path, "r", encoding="utf-8") as file:
                new_lines = file.readlines()

            view_start, view_end = _calculate_view_ranges(
                len(original_lines),
                len(new_lines),
                start,
                end,
            )

            show_content = "".join(
                [
                    f"{index + view_start}: {line}"
                    for index, line in enumerate(
                        new_lines[view_start - 1 : view_end],
                    )
                ],
            )

            return {"success": True, "response": f"""Write {file_path} successfully. The new content snippet:
```
{show_content}```"""}

        else:
            return {"success": False, "response": f"Error: Invalid range format. Expected a list " +\
                    f"of two integers, but got {ranges}."}

    # If ranges is None, overwrite the entire file (keeping previous behavior)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)

    return {"success": True, "response": f"Overwrite {file_path} successfully. The new content is:\n{content}\n"}


class WriteTextFile(Tool):
    """A tool to write content to a text file with optional line ranges."""

    def __init__(self):
        self._init_prompt()

    def _init_prompt(self):
        self.write_prompt = """
[EXTRACTION GUIDELINES]
Extract file path, content and ranges from the query. Follow these principles:
1. Identify the file path from the query
2. Extract the content to be written
3. If line ranges are specified for replacement, extract them as a list of two integers [start, end]
4. If a single line number is specified for insertion, extract it as [line_num, line_num]
5. Determine the operation type based on query context:
   - "overwrite": Replace content in a range or overwrite entire file
   - "append": Add content to the end of the file
   - "insert": Insert content at a specific line number
6. Return the extracted information in JSON format

## OUTPUT FORMAT:
Output STRICTLY according to this JSON Schema:
{{
    "file_path": "string",
    "content": "string",
    "ranges": ["int", "int"] | null,
    "mode": "overwrite" | "append" | "insert"
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
        ranges = response_dict['ranges']
        mode = response_dict['mode']
        return file_path, content, ranges, mode

    def execute(self, agent, query: str) -> str:
        """
        Execute the write text file operation based on the query.

        Args:
            agent: The agent object (not used in this tool).
            query (str): The query containing the file path, content and optional line ranges.

        Returns:
            str: The result of the write operation or an error message.
        """
        # Extract file path, content, ranges and mode from the query using JSON format
        key_prompt = self.write_prompt.format(query=query)
        response = agent.response(key_prompt, stream=False)
        file_path, content, ranges, mode = self.extract_kw_from_response(response)

        # Validate file path
        if not file_path:
            return {"success": False, "response": f"Error: File path is empty."}

        try:
            if mode == "overwrite":
                # For overwrite mode, call write_text_file with ranges if provided
                result = asyncio.run(write_text_file(file_path, content, ranges))
            elif mode == "append":
                # For append mode, call append_text_file to append content to the end of file
                result = asyncio.run(append_text_file(file_path, content))
            elif mode == "insert" and ranges is not None:
                # For insert mode (specific line number), call insert_text_file
                # Note: insert_text_file expects a single line number, not a range
                # So we'll use the start of the range as the insertion point
                start_line = ranges[0] if isinstance(ranges, list) and len(ranges) >= 1 else 1
                result = asyncio.run(insert_text_file(file_path, content, start_line))
            else:
                return {"success": False, "response": f"Error: Invalid combination of mode '{mode}' and ranges {ranges}."}

            return result
        except Exception as e:
            return {"success": False, "response": f"Error: {str(e)}"}

    @classmethod
    def content(cls):
        return """
Function: Write content to a text file with modify, overwrite, append or insert mode
Method: [
    Extract file path, content, ranges and mode from query,
    Write the content to the file using the specified mode (modify, overwrite, append, or insert),
    Return the result of the operation
]
Return: Result of the write operation
"""