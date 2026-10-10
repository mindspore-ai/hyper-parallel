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
"""Conservative generation-time recognition of supported MoE structures."""

import ast
from dataclasses import dataclass
import inspect
import textwrap
from typing import Any

from torch import nn


class UnsupportedModuleStructure(ValueError):
    """A matched module has no proven parallel recipe."""


@dataclass(frozen=True)
class MoeStructure:
    """Serializable structural decisions frozen before distributed execution."""

    router: str
    shared_experts: str
    expert_storage: str
    jitter: bool = False


def _unsupported(module: Any, detail: str) -> UnsupportedModuleStructure:
    return UnsupportedModuleStructure(
        f"{type(module).__name__}: {detail}. Declare an explicit "
        "local_compute_fn._target_ to opt out of structure detection."
    )


def _forward_tree(module: Any) -> ast.FunctionDef:
    try:
        source = textwrap.dedent(inspect.getsource(type(module).forward))
        tree = ast.parse(source)
    except (AttributeError, OSError, TypeError, SyntaxError) as exc:
        raise _unsupported(module, "forward source is unavailable") from exc
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1:
        raise _unsupported(module, "forward source is ambiguous")
    return functions[0]


def detect_router_kind(module: Any) -> str:
    """Recognize a router without invoking it or consulting a model identity.

    Tuple routers keep their own scoring algorithm, including grouped sigmoid
    routing. Recomputing scores from their first return value would lose gate
    configuration and risks applying scaling twice.
    """
    gate = getattr(module, "gate", None)
    if gate is None:
        gate = getattr(module, "router", None)
    if gate is None:
        raise _unsupported(module, "missing gate/router")
    if type(gate) is nn.Linear:  # pylint: disable=C0123  # exact match: subclasses take the AST path
        config = getattr(module, "config", None)
        if not any(getattr(owner, name, None) is not None
                   for owner in (module, config) for name in ("top_k", "num_experts_per_tok")):
            raise _unsupported(module, "linear router has no explicit top-k setting")
        # A bare-linear gate feeds either softmax top-k routing or
        # group-limited sigmoid routing; both gates are literally nn.Linear.
        # The merge contract is told apart by the sigmoid/group markers the
        # MOE declares on itself or its config (DeepSeek-V3 / GLM-4-MoE keep
        # routed_scaling_factor + n_group + topk_group; plain softmax routers
        # carry none). This mirrors the runtime router adapters, which key on
        # the same attributes.
        if any(getattr(owner, name, None) is not None
               for owner in (module, config) for name in ("routed_scaling_factor", "n_group", "topk_group")):
            return "sigmoid_group"
        return "softmax_topk"
    forward = _forward_tree(gate)
    returns = [node.value for node in ast.walk(forward) if isinstance(node, ast.Return)]
    calls = {node.func.attr for node in ast.walk(forward)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    index_names = _topk_index_names(forward)
    if (returns and all(isinstance(value, ast.Tuple) and len(value.elts) == 3
                        and isinstance(value.elts[2], ast.Name)
                        and value.elts[2].id in index_names for value in returns)
            and "topk" in calls and "linear" in calls):
        return "topk_router_module"
    raise _unsupported(module, "router does not expose a supported logits/scores/indices contract")


def _topk_index_names(forward: ast.FunctionDef) -> set[str]:
    """Prove the final tuple item comes from a top-k index result."""
    names = set()
    for node in ast.walk(forward):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target, value = node.targets[0], node.value
        if (isinstance(target, (ast.Tuple, ast.List)) and len(target.elts) == 2
                and isinstance(target.elts[1], ast.Name) and _is_topk_call(value)):
            names.add(target.elts[1].id)
        elif (isinstance(target, ast.Name) and isinstance(value, ast.Subscript)
              and isinstance(value.slice, ast.Constant) and value.slice.value == 1
              and _is_topk_call(value.value)):
            names.add(target.id)
    return names


def _is_topk_call(value: ast.AST) -> bool:
    return (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
            and value.func.attr == "topk")


def detect_moe_structure(module: Any) -> MoeStructure:
    """Freeze the router, shared branch and expert parameter storage contract."""
    experts = getattr(module, "experts", None)
    if experts is None:
        raise _unsupported(module, "missing experts")
    if isinstance(experts, nn.ModuleList):
        if not experts or not all(all(hasattr(expert, name) for name in
                                     ("gate_proj", "up_proj", "down_proj")) for expert in experts):
            raise _unsupported(module, "expert list is not a SwiGLU projection list")
        storage = "module_list"
    elif all(isinstance(getattr(experts, name, None), nn.Parameter)
             and getattr(experts, name).ndim == 3 for name in ("gate_up_proj", "down_proj")):
        storage = "batched_parameters"
    else:
        raise _unsupported(module, "unsupported expert parameter storage")
    shared_names = {name for name in ("shared_expert", "shared_expert_gate", "shared_experts")
                    if getattr(module, name, None) is not None}
    if not shared_names:
        shared = "none"
    elif shared_names == {"shared_expert", "shared_expert_gate"}:
        shared = "gated"
    elif shared_names == {"shared_experts"}:
        shared = "additive"
    else:
        raise _unsupported(module, f"ambiguous shared expert branches {sorted(shared_names)}")
    return MoeStructure(detect_router_kind(module), shared, storage, hasattr(module, "jitter_noise"))
