# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Safe arithmetic calculator tool for multi-turn GSM8K training."""

import ast
import operator
from functools import lru_cache
from typing import Callable

from rl.agentic.tools import ToolRegistry


_BINARY_OPERATORS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _evaluate(node: ast.AST) -> float:
    """Evaluate one bounded arithmetic syntax tree."""
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 12:
            raise ValueError("calculator exponent magnitude must not exceed 12")
        return _BINARY_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate(node.operand))
    raise ValueError("calculator accepts numbers and arithmetic operators only")


def calculate(expression: str) -> int | float:
    """Evaluate one short arithmetic expression without names or calls."""
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("calculator expression must be non-empty text")
    if len(expression) > 256:
        raise ValueError("calculator expression must contain at most 256 characters")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ValueError("calculator expression is invalid") from error
    value = _evaluate(tree)
    if not -1e100 < value < 1e100:
        raise ValueError("calculator result is outside the supported range")
    return int(value) if value.is_integer() else value


@lru_cache(maxsize=1)
def build_calculator_registry() -> ToolRegistry:
    """Build the calculator tool definition used by GSM8K multi-turn mode."""
    registry = ToolRegistry()
    registry.register(
        "calculator",
        description="Evaluate a numeric arithmetic expression",
        parameters={
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    )(calculate)
    return registry
