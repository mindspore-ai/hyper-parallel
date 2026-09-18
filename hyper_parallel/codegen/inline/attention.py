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
"""Generate a self-contained attention class from real component source.

The generated artifact preserves the genuine construction and fused QKV weight
layout (``linear_qkv`` + ``InterleaveQKV`` checkpoint conversion) by copying the
real ``GQAAttention`` methods verbatim, and only rewrites ``forward`` to place
TP/CP orchestration around the attention kernel. The kernel entry function that
the real class normally receives as ``attention_interface`` is inlined from its
real source (plan §6), so the artifact reads as orchestration down to the single
hardware kernel ``torch_npu.npu_fusion_attention`` instead of a black-box import.

This generator is platform-agnostic ``core`` code: the module type and kernel
entry are handed in, never imported here.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any, Callable

from hyper_parallel.codegen.inline.expansion import ExpandedSource, expand_function


_TP_ALL_GATHER = ast.parse(
    """
ps = get_parallel_state()
if ps.tp_enabled:
    hidden_states = ps.tp.all_gather(hidden_states, dim=1)
"""
).body


_CP_DISPATCH = ast.parse(
    """
if ps.cp_enabled:
    cp_mesh = ps.cp_mesh
    query_length = query_states.shape[-2]
    query_offset = cp_mesh.get_local_rank() * query_length
    key_states, value_states = flex_cp_allgather(
        key_states.contiguous(), value_states.contiguous(), 2, cp_mesh
    )
    attention_mask = _cp_offset_causal_mask(
        query_length, key_states.shape[-2], query_offset, query_states.device
    )
"""
).body


_TP_REDUCE_SCATTER = ast.parse(
    """
if ps.tp_enabled:
    attn_output = ps.tp.reduce_scatter(attn_output, dim=1)
"""
).body


_GET_PARALLEL_STATE = '''
def get_parallel_state():
    """Return the externally installed codegen parallel state."""
    return get_inline_parallel_state(__name__)
'''


def _is_attention_interface_call(statement: ast.AST) -> bool:
    for node in ast.walk(statement):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
            and node.func.attr == "attention_interface"
        ):
            return True
    return False


def _rewrite_attention_interface_calls(statement: ast.AST) -> None:
    for node in ast.walk(statement):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
            and node.func.attr == "attention_interface"
        ):
            node.func = ast.Name(id="run_qwen3_moe_flash_attention", ctx=ast.Load())


def _build_forward(module_type: type) -> ast.FunctionDef:
    """Derive the forward from real source with TP/CP orchestration inserted."""
    source = textwrap.dedent(inspect.getsource(module_type.forward))
    tree = ast.parse(source)
    forward = tree.body[0]
    if not isinstance(forward, ast.FunctionDef) or forward.name != "forward":
        raise ValueError(f"codegen attention: {module_type.__name__}.forward is not a plain def")

    body = list(_TP_ALL_GATHER)
    kernel_boundary_inserted = False
    for statement in forward.body:
        if _is_attention_interface_call(statement):
            if kernel_boundary_inserted:
                raise ValueError("codegen attention: source contains multiple kernel call sites")
            if not isinstance(statement, ast.Assign):
                raise ValueError("codegen attention: kernel call must be a plain assignment")
            _rewrite_attention_interface_calls(statement)
            body.extend(_CP_DISPATCH)
            body.append(statement)
            kernel_boundary_inserted = True
        elif isinstance(statement, ast.Return):
            if not kernel_boundary_inserted:
                raise ValueError("codegen attention: return appears before the kernel call")
            body.extend(_TP_REDUCE_SCATTER)
            body.append(statement)
        else:
            body.append(statement)
    if not kernel_boundary_inserted:
        raise ValueError("codegen attention: source has no attention_interface call to bound")

    forward.body = body
    _strip_annotations(forward)
    ast.fix_missing_locations(forward)
    return forward


def _strip_annotations(statement: ast.FunctionDef) -> None:
    """Drop type hints so the artifact has no import-time resolution burden.

    Only annotations are cleared; ``*args`` and ``**kwargs`` parameters are kept
    because real forward bodies may reference them.
    """
    statement.returns = None
    for arg in (*statement.args.posonlyargs, *statement.args.args, *statement.args.kwonlyargs):
        arg.annotation = None
    if statement.args.vararg is not None:
        statement.args.vararg.annotation = None
    if statement.args.kwarg is not None:
        statement.args.kwarg.annotation = None


def _class_methods(module_type: type) -> list[ast.stmt]:
    """Copy every real method except ``forward``, dropping the module decorator."""
    source = textwrap.dedent(inspect.getsource(module_type))
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == module_type.__name__:
            methods = [
                statement
                for statement in node.body
                if not (
                    isinstance(statement, ast.FunctionDef) and statement.name == "forward"
                )
            ]
            for statement in methods:
                statement.decorator_list = []
                if isinstance(statement, ast.FunctionDef):
                    _strip_annotations(statement)
            return methods
    raise ValueError(f"codegen attention: no class {module_type.__name__!r} in its source module")


def render_attention_class(module_type: type, *, interface: Callable[..., Any]) -> ExpandedSource:
    """Render a self-contained attention class with visible TP/CP orchestration.

    Args:
        module_type: The real grouped-query attention component (e.g. ``modules.GQAAttention``).
        interface: The attention kernel entry function to inline (e.g.
            ``run_qwen3_moe_flash_attention``); its closure is expanded into the
            artifact and it must stop at the hardware kernel.
    """
    expanded_interface = expand_function(interface)
    forward = _build_forward(module_type)
    class_node = ast.ClassDef(
        name=module_type.__name__,
        bases=[ast.Name(id="nn.Module", ctx=ast.Load())],
        keywords=[],
        body=_class_methods(module_type) + [forward],
        decorator_list=[],
    )
    built_class = ast.unparse(class_node)
    shared_defs = (
        expanded_interface.source + "\n\n" + _GET_PARALLEL_STATE.strip()
    )
    imports = (
        *expanded_interface.imports,
        "from torch import nn",
        "from hyper_parallel.components.checkpoint.weight_conversion import WeightConverter",
        "from hyper_parallel.components.checkpoint import InterleaveQKV",
        "from hyper_parallel.components.functional import apply_rotary_pos_emb",
        "from hyper_parallel.components.functional import apply_rotary_pos_emb_interleave",
        "from hyper_parallel.components.functional import npu_fusion_attention_forward",
        "from hyper_parallel.distributed.context_parallel import flex_cp_allgather",
        "from hyper_parallel.distributed.context_parallel.attention import _cp_offset_causal_mask",
    )
    return ExpandedSource(
        imports,
        "def rotate_half(x):\n"
        "    first = x[..., : x.shape[-1] // 2]\n"
        "    second = x[..., x.shape[-1] // 2 :]\n"
        "    return torch.cat((-second, first), dim=-1)\n\n\n"
        "def _projections_can_fuse(projections):\n"
        "    biases = tuple(projection.bias for projection in projections)\n"
        "    return (\n"
        "        len({projection.in_features for projection in projections}) == 1\n"
        "        and len({projection.weight.requires_grad for projection in projections}) == 1\n"
        "        and (all(bias is None for bias in biases) or all(bias is not None for bias in biases))\n"
        "        and (biases[0] is None or len({bias.requires_grad for bias in biases}) == 1)\n"
        "    )\n\n\n"
        + shared_defs
        + "\n\n\n"
        + built_class,
    )