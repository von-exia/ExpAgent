# -*- coding: utf-8 -*-
# flake8: noqa: E501
# pylint: disable=line-too-long
"""Tool for reading multiple text files or files matching glob patterns."""
import os
import re
import glob as py_glob
import pathlib
from typing import List, Union
from act.tools.tool import Tool
from agent_model.utils import extract_dict_from_text


def _read_single_file(
    file_path: str,
    ranges: list[int] | None = None,
) -> str:
    """Return the file content in the specified range with line numbers."""
    with open(file_path, "r", encoding="utf-8", errors='ignore') as file:
        lines = file.readlines()

    if ranges:
        if (
            isinstance(ranges, list)
            and len(ranges) == 2
            and all(isinstance(i, int) for i in ranges)
        ):
            start, end = ranges
            if start > end:
                raise ValueError(f"InvalidArgumentError: The start line is greater than the " +
                    f"end line in the given range {ranges}.")

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
        else:
            raise ValueError(f"InvalidArgumentError: Invalid range format. Expected a list of " +
                f"two integers, but got {ranges}.")

    return "".join(f"{index + 1}: {line}" for index, line in enumerate(lines))


def _get_files_by_glob(
    pattern: str,
    path: str = ".",
    recursive: bool = True,
) -> List[str]:
    """Get a list of files matching the glob pattern."""
    search_path = pathlib.Path(path)

    if not search_path.exists():
        raise ValueError(f"Path does not exist: {path}")

    if not search_path.is_dir():
        raise ValueError(f"Path is not a directory: {path}")

    # Construct the full glob pattern
    if recursive and not pattern.startswith("**"):
        full_pattern = str(search_path / "**" / pattern)
    else:
        full_pattern = str(search_path / pattern)

    # Use glob to find matching files
    if recursive and pattern.startswith("**/"):
        matches = list(search_path.rglob(pattern[3:]))  # Remove "**/" prefix
    elif recursive:
        matches = list(search_path.rglob(pattern))
    else:
        matches = list(search_path.glob(pattern))

    # Convert Path objects to strings
    return [str(match) for match in matches]


class ReadManyFiles(Tool):
    """A tool to read the content of multiple text files or files matching glob patterns."""

    def __init__(self):
        self._init_prompt()

    def _init_prompt(self):
        self.key_prompt = """
[EXTRACTION GUIDELINES]
Extract file paths/glob patterns and ranges from the query. Follow these principles:
1. Identify the file paths or glob patterns from the query
2. If line ranges are specified, extract them as a list of two integers
3. If recursive search is needed, set the recursive flag to true
4. Return the extracted information in JSON format

## OUTPUT FORMAT:
Output STRICTLY according to this JSON Schema:
{{
    "file_paths_or_patterns": ["string"],
    "ranges": ["int", "int"] | null,
    "recursive": "boolean"
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
        file_paths_or_patterns = response_dict['file_paths_or_patterns']
        ranges = response_dict['ranges']
        recursive = response_dict.get('recursive', True)
        return file_paths_or_patterns, ranges, recursive


    def execute(self, agent, query: str) -> str:
        """
        Execute the read many files operation based on the query.

        Args:
            agent: The agent object (not used in this tool).
            query (str): The query containing the file paths/patterns and optional line ranges.

        Returns:
            str: The content of the files or an error message.
        """
        # Extract file paths/patterns and ranges from the query using JSON format
        key_prompt = self.key_prompt.format(query=query)
        response = agent.response(key_prompt, stream=False)
        file_paths_or_patterns, ranges, recursive = self.extract_arg_from_response(response)

        all_contents = []
        
        for path_or_pattern in file_paths_or_patterns:
            # Check if it's a glob pattern (contains *, ?, [, ])
            is_glob_pattern = any(char in path_or_pattern for char in ['*', '?', '[', ']'])
            
            if is_glob_pattern:
                # It's a glob pattern, expand it to get actual file paths
                try:
                    matching_files = _get_files_by_glob(path_or_pattern, ".", recursive)
                except ValueError as e:
                    return {"success": False, "response": f"Error: {str(e)}"}
                
                for file_path in matching_files:
                    # Validate file path
                    if not os.path.exists(file_path):
                        all_contents.append(f"Error: The file {file_path} does not exist.")
                        continue
                    if not os.path.isfile(file_path):
                        all_contents.append(f"Error: The path {file_path} is not a file.")
                        continue

                    try:
                        content = _read_single_file(file_path, ranges)
                        all_contents.append(f"--- START OF FILE: {file_path} ---\n{content}--- END OF FILE: {file_path} ---")
                    except Exception as e:
                        all_contents.append(f"Error reading {file_path}: {str(e)}")
            else:
                # It's a single file path
                file_path = path_or_pattern
                
                # Validate file path
                if not os.path.exists(file_path):
                    all_contents.append(f"Error: The file {file_path} does not exist.")
                    continue
                if not os.path.isfile(file_path):
                    all_contents.append(f"Error: The path {file_path} is not a file.")
                    continue

                try:
                    content = _read_single_file(file_path, ranges)
                    all_contents.append(f"--- START OF FILE: {file_path} ---\n{content}--- END OF FILE: {file_path} ---")
                except Exception as e:
                    all_contents.append(f"Error reading {file_path}: {str(e)}")

        if not all_contents:
            return {"success": False, "response": "No files matched the given patterns."}

        combined_content = "\n\n".join(all_contents)
        
        if ranges is None:
            return {"success": True,
                   "response": f"""The content of the requested files:
```
{combined_content}```"""}
        else:
            return {"success": True,
                   "response": f"""The content of the requested files in {ranges} lines:
```
{combined_content}```"""}

    @classmethod
    def content(cls):
        return """
Function: Read the content of multiple text files or files matching glob patterns with line numbers
Method: [
    Extract file paths/patterns and ranges from query,
    Expand glob patterns to actual file paths,
    Read each file content with specified ranges,
    Return the content with line numbers and file separators
]
Return: Combined file content with line numbers and file separators
"""