# -*- coding: utf-8 -*-
"""A tool to list directory contents (LS)"""
import os
from typing import Dict, Any
from act.tools.tool import Tool
from agent_model.utils import extract_dict_from_text


def _list_directory(path: str = ".", show_hidden: bool = False) -> Dict[str, Any]:
    """
    List the contents of a directory.

    Args:
        path (str): The directory path to list. Defaults to current directory.
        show_hidden (bool): Whether to show hidden files/folders. Defaults to False.

    Returns:
        A dictionary containing the directory listing results.
    """
    try:
        # Check if path exists and is a directory
        if not os.path.exists(path):
            return {
                "success": False,
                "error": f"Directory '{path}' does not exist."
            }
        
        if not os.path.isdir(path):
            return {
                "success": False,
                "error": f"'{path}' is not a directory."
            }
        
        # List directory contents
        if show_hidden:
            items = os.listdir(path)
        else:
            items = [item for item in os.listdir(path) if not item.startswith('.')]
        
        # Separate files and directories
        files = []
        directories = []
        
        for item in items:
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                directories.append(item)
            else:
                files.append(item)
        
        return {
            "success": True,
            "path": os.path.abspath(path),
            "directories": sorted(directories),
            "files": sorted(files),
            "total_directories": len(directories),
            "total_files": len(files)
        }
    
    except PermissionError:
        return {
            "success": False,
            "error": f"Permission denied to access directory '{path}'."
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"An error occurred while listing directory '{path}': {str(e)}"
        }


class ListDirConts(Tool):
    """A tool to list directory contents."""

    def __init__(self):
        self._init_prompt()

    def _init_prompt(self):
        self.key_prompt = """
[EXTRACTION GUIDELINES]
Extract directory path and show_hidden flag from the query. Follow these principles:
1. Identify the directory path from the query (default to current directory if not specified)
2. Determine if hidden files should be shown (default to false)
3. Return the extracted information in JSON format

## OUTPUT FORMAT:
Output STRICTLY according to this JSON Schema:
{{
    "path": "string",
    "show_hidden": "boolean"
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
        path = response_dict.get('path', '.')
        show_hidden = response_dict.get('show_hidden', False)
        return path, show_hidden

    def execute(self, agent, query: str) -> Dict[str, Any]:
        """
        Execute the list directory operation based on the query.

        Args:
            agent: The agent object (not used in this tool).
            query (str): The query containing the directory path and options.

        Returns:
            Dict containing success status and directory listing results.
        """
        # Extract path and show_hidden from the query using JSON format
        key_prompt = self.key_prompt.format(query=query)
        response = agent.response(key_prompt, stream=False)
        path, show_hidden = self.extract_arg_from_response(response)

        # List the directory
        result = _list_directory(path, show_hidden)

        if result["success"]:
            # Format the response
            dir_list = result["directories"]
            file_list = result["files"]
            
            response_text = f"Contents of directory: {result['path']}\n\n"
            
            if dir_list:
                response_text += f"Directories ({result['total_directories']}):\n"
                for directory in dir_list:
                    response_text += f"  [DIR] {directory}\n"
                response_text += "\n"
            
            if file_list:
                response_text += f"Files ({result['total_files']}):\n"
                for file in file_list:
                    response_text += f"  [FILE] {file}\n"
                response_text += "\n"
            
            if not dir_list and not file_list:
                response_text += "Directory is empty.\n"
            
            response_text += f"Total: {result['total_directories']} directories, {result['total_files']} files"
            
            return {
                "success": True,
                "response": response_text
            }
        else:
            return {
                "success": False,
                "response": result["error"]
            }

    @classmethod
    def content(cls):
        return """
Function: List directory contents
Method: [
    Extract directory path and show_hidden flag from query,
    List the directory contents separating files and directories,
    Return the contents with counts
]
Return: Directory listing with files and directories separated
"""