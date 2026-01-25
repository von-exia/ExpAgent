
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
            raise f"InvalidArgumentError: The start line is greater than the " +\
                f"end line in the given range {ranges}."
    else:
        raise f"InvalidArgumentError: Invalid range format. Expected a list of " +\
            f"two integers, but got {ranges}."


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
            raise f"InvalidArgumentError: The range '{ranges}' is out of bounds " +\
                f"for the file '{file_path}', which has only {len(lines)} " +\
                f"lines."

        view_content = [
            f"{index + start}: {line}"
            for index, line in enumerate(lines[start - 1 : end])
        ]

        return "".join(view_content)

    return "".join(f"{index + 1}: {line}" for index, line in enumerate(lines))


# -*- coding: utf-8 -*-
# flake8: noqa: E501
# pylint: disable=line-too-long
"""The view text file tool in agentscope."""
import os



async def view_text_file(
    file_path: str,
    ranges: list[int] | None = None,
):
    """View the file content in the specified range with line numbers. If `ranges` is not provided, the entire file will be returned.

    Args:
        file_path (`str`):
            The target file path.
        ranges:
            The range of lines to be viewed (e.g. lines 1 to 100: [1, 100]), inclusive. If not provided, the entire file will be returned. To view the last 100 lines, use [-100, -1].

    Returns:
        `ToolResponse`:
            The tool response containing the file content or an error message.
    """
    if not os.path.exists(file_path):
        return f"Error: The file {file_path} does not exist."
    if not os.path.isfile(file_path):
        return f"Error: The path {file_path} is not a file."

    try:
        content = _view_text_file(file_path, ranges)
    except Exception as e:
        return e

    if ranges is None:
        return f"""The content of {file_path}:
```
{content}```"""
    else:
        return f"""The content of {file_path} in {ranges} lines:
```
{content}```"""