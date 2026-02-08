# -*- coding: utf-8 -*-

import ast
import operator
from typing import Union
import re
import logging

from act.tools.tool import Tool
from agent_model.utils import extract_dict_from_text


class Calculator(Tool):
    """
    A calculator tool that evaluates mathematical expressions safely.
    Supports basic arithmetic operations (+, -, *, /, //, %, **) and parentheses.
    """
    
    # Define allowed operators
    _allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,  # Unary minus
        ast.UAdd: operator.pos,  # Unary plus
    }
    
    # Define allowed functions
    _allowed_functions = {
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'sum': sum,
    }
    
    def __init__(self):
        self._init_prompt()
    
    def _init_prompt(self):
        self.expr_prompt = """
[CALCULATOR GUIDELINES]
Generate valid Python arithmetic expressions to complete current goal. Follow these requirements:

## TIME CALCULATION HANDLING
When dealing with time values:
1. **Convert time formats to calculable numeric values**
   - Time format "HH:MM:SS" or "HH:MM" must be converted to a single numeric unit (seconds, minutes, or hours)
   - Example: "2:02:01" (2 hours, 2 minutes, 1 second) should be converted to:
     - In hours: `2 + 2/60 + 1/3600`
     - In minutes: `2*60 + 2 + 1/60`
     - In seconds: `2*3600 + 2*60 + 1`
   - Always specify the unit in your description

2. **Time duration calculations**:
   - For time differences: Convert both times to the same unit and subtract
   - For adding/subtracting time: Use consistent units
   - For time averages: Sum all time values in consistent units, then divide

## VALID EXPRESSION EXAMPLES
### Basic operations
2 + 3
10 - 4
5 * 6
15 / 3

### Exponentiation and modulo
2 ** 3
17 % 5

### Expressions with parentheses
(4 + 5) * 2

### Function calls
abs(-10)
round(3.14159, 2)
max(5, 10, 3)
min(5, 10, 3)

### List operations
sum([1, 2, 3, 4])

### To express in thousands after rounding: 
round(17456, -3) / 1000
17456 // 1000

## OUTPUT REQUIREMENTS
- Output the expression to only complete the current goal. DO NOT MAKE THE EXPRESSION TOO LONG. 
- Provide a concise explanation including:
    i) Meaning and Unit of each variable
    ii) Expression meaning
    
## OUTPUT FORMAT:
Output STRICTLY according to this JSON Schema:
{{
    "expression": "string"
    "explanation": {{
        "variables_meaning_and_unit": "string",
        "expression_meaning": "string"
    }}
}}

Return only valid JSON.

[USER]
Query:
{query}

Only focus on the current goal.

[ASSISTANT]
/no_think
"""

    def extract_expr_from_response(self, response):
        response_dict = extract_dict_from_text(response)
        expression = response_dict['expression']
        explanation = response_dict['explanation']
        vmu = explanation['variables_meaning_and_unit']
        expression_meaning = explanation['expression_meaning']
        explanation = f"Variables_meaning_and_unit: {vmu}\nExpression_meaning: {expression_meaning}"
        return expression, explanation

    def execute(self, agent, query: str) -> str:
        """
        Safely evaluate a mathematical expression.

        Args:
            agent: The agent object (not used in this tool).
            expression (str): The mathematical expression to evaluate.

        Returns:
            str: The result of the calculation or an error message.
        """
        prompt = self.expr_prompt.format(query=query)
        response = agent.response(prompt, False)
        expression, explanation = self.extract_expr_from_response(response)
        
        try:
            # Parse the expression into an Abstract Syntax Tree (AST)
            tree = ast.parse(expression.strip(), mode='eval')
            
            # Evaluate the AST safely
            result = self._eval_node(tree.body)
            
            return {
                "success": True,
                "response": f"\n[START OF CALCULATOR RESULT]\nThe result of \"{expression}\" is:\n{result}\nThe explanation of the expression and results:\n{explanation}\n[END OF CALCULATOR RESULT]\n"
            }
        
        except SyntaxError:
            return {"success": False, "response": f"Error: Invalid syntax in expression '{expression}'. Please check your mathematical expression."}
        
        except ZeroDivisionError:
            return {"success": False, "response": f"Error: Division by zero in expression '{expression}'."}
        
        except OverflowError:
            return {"success": False, "response": f"Error: Result too large in expression '{expression}'."}
        
        except Exception as e:
            return {"success": False, "response": f"Error: Failed to evaluate expression '{expression}'. Reason: {str(e)}"}

    def _eval_node(self, node):
        """
        Recursively evaluate AST nodes.
        """
        if isinstance(node, ast.Constant):  # Numbers (Python 3.8+)
            return node.value
        elif isinstance(node, ast.Num):  # Numbers (older Python versions)
            return node.n
        elif isinstance(node, ast.Str):  # Strings (older Python versions)
            return node.s
        elif isinstance(node, ast.BinOp):  # Binary operations (e.g., +, -, *, /)
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op = self._allowed_operators.get(type(node.op))
            
            if op is None:
                raise ValueError(f"Operator {type(node.op).__name__} not allowed")
            
            return op(left, right)
        
        elif isinstance(node, ast.UnaryOp):  # Unary operations (e.g., -, +)
            operand = self._eval_node(node.operand)
            op = self._allowed_operators.get(type(node.op))
            
            if op is None:
                raise ValueError(f"Unary operator {type(node.op).__name__} not allowed")
            
            return op(operand)
        
        elif isinstance(node, ast.Call):  # Function calls
            func_name = node.func.id if isinstance(node.func, ast.Name) else None
            
            if func_name not in self._allowed_functions:
                raise ValueError(f"Function '{func_name}' not allowed")
            
            args = [self._eval_node(arg) for arg in node.args]
            return self._allowed_functions[func_name](*args)
        
        elif isinstance(node, ast.List):  # Lists
            return [self._eval_node(item) for item in node.elts]
        
        elif isinstance(node, ast.Tuple):  # Tuples
            return tuple(self._eval_node(item) for item in node.elts)
        
        else:
            raise ValueError(f"Node type {type(node).__name__} not allowed")

    @classmethod
    def content(cls):
        return """
Function: Calculate expression that derived from current goal, through python function (e.g., +, -, *, /, //, abs, round, min, max)
Method: [
    Derive expression from query,
    Evaluate and calculate the expression,
    Explain the expression and result
]
Return: [
    expression result,
    explaination of the expression   
]
"""
