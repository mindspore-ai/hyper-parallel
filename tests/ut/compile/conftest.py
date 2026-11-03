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
"""Temporary compatibility surface for the unadapted graph-compile UT suite."""

from __future__ import annotations

from dataclasses import dataclass
import fnmatch
from typing import Any, Callable, Optional

import torch
from torch import nn

from hyper_parallel.distributed import activation_checkpoint as activation_checkpoint_module
from hyper_parallel.models import adapter_spec as adapter_spec_module


@dataclass(frozen=True)
class RecomputePolicy:
    """Test-only copy of the graph pass's retired adapter declaration."""

    region_patterns: tuple[str, ...]
    exclude_patterns: tuple[str, ...] = ()
    exclude_predicate: Optional[Callable[[str, Any], bool]] = None
    on_applied: Optional[Callable[[Any], None]] = None


def _resolve_torch_op(dotted_path: str) -> Any:
    """Resolve an optional torch operator to its default overload."""
    if dotted_path.count(".") == 1:
        dotted_path = f"{dotted_path}.default"
    obj = torch.ops
    try:
        for part in dotted_path.split("."):
            obj = getattr(obj, part)
    except AttributeError:
        return None
    return obj


def _resolve_op_attr(root: object, dotted_path: str) -> Any:
    """Resolve an optional dotted attribute."""
    obj = root
    try:
        for part in dotted_path.split("."):
            obj = getattr(obj, part)
    except AttributeError:
        return None
    return obj


def _existing_ops(*ops: Any) -> frozenset[Any]:
    """Drop unavailable optional operators."""
    return frozenset(op for op in ops if op is not None)


_SELECTIVE_AC_FORCE_RECOMPUTE_OPS = _existing_ops(_resolve_torch_op("aten.topk"))
_SELECTIVE_AC_COMM_OP_NAMES = (
    "aten.all_to_all_single",
    "aten.reduce_scatter_tensor",
    "_c10d_functional.all_to_all_single",
    "_c10d_functional.reduce_scatter_tensor",
    "c10d.allreduce_",
)
_SELECTIVE_AC_MUST_SAVE_OPS = _existing_ops(
    *(
        _resolve_torch_op(name)
        for name in (
            "aten.mm",
            "aten.addmm",
            "aten.bmm",
            "aten.linear",
            "aten._scaled_mm",
            "aten._scaled_dot_product_cudnn_attention",
            "aten._scaled_dot_product_efficient_attention",
            "aten._scaled_dot_product_flash_attention",
            "aten._scaled_dot_product_flash_attention_for_cpu",
            "aten._scaled_dot_product_fused_attention_overrideable",
            "aten.scaled_dot_product_attention",
            "npu.npu_fusion_attention_v3",
            "aten._flex_attention",
            "aten.max",
        )
    ),
    *(_resolve_torch_op(name) for name in _SELECTIVE_AC_COMM_OP_NAMES),
    _resolve_op_attr(torch, "_higher_order_ops.flex_attention"),
    _resolve_op_attr(torch, "_higher_order_ops.inductor_compiled_code"),
) - _SELECTIVE_AC_FORCE_RECOMPUTE_OPS


def _adapter_recompute_spec(model: nn.Module, *, required: bool = True) -> Optional[RecomputePolicy]:
    """Return no legacy policy unless a test patches the compile-pass helper."""
    del model
    if required:
        raise ValueError("test model has no legacy recompute adapter")


def _selected_checkpoint_layer_fqns(containers: list[Any], selection: Any) -> set[str]:
    """Convert the retired layer selector into discovered layer FQNs."""
    blocks = [block for container in containers for block in container.blocks]
    layer_count = getattr(selection, "layer_count", None)
    layer_indices = getattr(selection, "layer_indices", None)
    if (layer_count is None) == (layer_indices is None):
        raise ValueError("exactly one layer selector is required")
    if layer_indices is not None:
        if not isinstance(layer_indices, (list, tuple)) or not layer_indices:
            raise ValueError("layer_indices must be a non-empty sequence")
        if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in layer_indices):
            raise ValueError("layer_indices must contain non-negative integers")
        if len(set(layer_indices)) != len(layer_indices) or max(layer_indices) >= len(blocks):
            raise ValueError("layer_indices are duplicated or out of range")
        return {blocks[index].fqn for index in layer_indices}
    if isinstance(layer_count, bool) or not isinstance(layer_count, int) or not 0 <= layer_count <= len(blocks):
        raise ValueError("layer_count is invalid")
    return {blocks[index].fqn for index in range(layer_count)}


def _select_layer_region_roots(
    module_by_fqn: dict[str, nn.Module], selected_layers: set[str], recompute_spec: RecomputePolicy,
) -> list[tuple[str, nn.Module]]:
    """Select outermost legacy adapter regions under selected layers."""
    matched = [
        (fqn, module)
        for fqn, module in module_by_fqn.items()
        if fqn and any(fnmatch.fnmatchcase(fqn, pattern) for pattern in recompute_spec.region_patterns)
    ]
    roots = [
        (fqn, module)
        for fqn, module in matched
        if not any(fqn != other and fqn.startswith(f"{other}.") for other, _ in matched)
    ]
    return [
        (fqn, module)
        for fqn, module in roots
        if any(fqn == layer or fqn.startswith(f"{layer}.") for layer in selected_layers)
    ]


def _select_excluded_targets(
    module_by_fqn: dict[str, nn.Module], roots: list[tuple[str, nn.Module]], recompute_spec: RecomputePolicy,
) -> list[tuple[str, nn.Module]]:
    """Select legacy adapter exclusions below region roots."""
    root_fqns = tuple(fqn for fqn, _ in roots)
    selected = []
    for fqn, module in module_by_fqn.items():
        if not any(fqn == root or fqn.startswith(f"{root}.") for root in root_fqns):
            continue
        pattern_match = any(fnmatch.fnmatchcase(fqn, pattern) for pattern in recompute_spec.exclude_patterns)
        predicate = recompute_spec.exclude_predicate
        if pattern_match or (predicate is not None and predicate(fqn, module)):
            selected.append((fqn, module))
    return selected


def _validate_region_exclusions(
    roots: list[tuple[str, nn.Module]], exclusions: list[tuple[str, nn.Module]],
) -> None:
    """Reject overlapping legacy adapter regions and exclusions."""
    root_fqns = {fqn for fqn, _ in roots}
    root_ids = {id(module) for _, module in roots}
    for fqn, module in exclusions:
        if fqn in root_fqns:
            raise ValueError("region root is excluded")
        if any(child is not module and id(child) in root_ids for child in module.modules()):
            raise ValueError("excluded region contains a region root")


adapter_spec_module.RecomputePolicy = RecomputePolicy
for _name, _value in {
    "_SELECTIVE_AC_COMM_OP_NAMES": _SELECTIVE_AC_COMM_OP_NAMES,
    "_SELECTIVE_AC_FORCE_RECOMPUTE_OPS": _SELECTIVE_AC_FORCE_RECOMPUTE_OPS,
    "_SELECTIVE_AC_MUST_SAVE_OPS": _SELECTIVE_AC_MUST_SAVE_OPS,
    "_adapter_recompute_spec": _adapter_recompute_spec,
    "_existing_ops": _existing_ops,
    "_resolve_op_attr": _resolve_op_attr,
    "_resolve_torch_op": _resolve_torch_op,
    "_select_excluded_targets": _select_excluded_targets,
    "_select_layer_region_roots": _select_layer_region_roots,
    "_selected_checkpoint_layer_fqns": _selected_checkpoint_layer_fqns,
    "_validate_region_exclusions": _validate_region_exclusions,
}.items():
    setattr(activation_checkpoint_module, _name, _value)
