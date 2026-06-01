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

"""Defensive YAML loading helpers (caps nesting depth to guard against malicious input)."""
from typing import Union

import yaml
from yaml.nodes import MappingNode, Node

YAML_MAX_NESTING_DEPTH = 10


def _get_yaml_ast_depth(node: Node, depth: int = 0) -> int:
    """Recursively return the maximum nesting depth of a YAML AST."""
    if isinstance(node, MappingNode):
        return max(
            (_get_yaml_ast_depth(v, depth + 1) for _, v in node.value), default=depth
        )
    return depth


def check_yaml_depth_before_loading(yaml_str: Union[str, bytes],
                                    max_depth: int = YAML_MAX_NESTING_DEPTH) -> None:
    """Reject YAML documents whose nesting depth exceeds ``max_depth``.

    Args:
        yaml_str: Raw YAML text.
        max_depth: Maximum permitted nesting depth.

    Raises:
        ValueError: If the document exceeds ``max_depth`` or fails to parse.
    """
    try:
        node = yaml.compose(yaml_str)
        if node is None:
            return
        depth = _get_yaml_ast_depth(node)
        if depth > max_depth:
            raise ValueError(
                f"YAML nesting depth {depth} exceeds the maximum allowed value of {max_depth}"
            )
    except yaml.YAMLError as e:
        raise ValueError(f"YAML parse error: {e}") from e
