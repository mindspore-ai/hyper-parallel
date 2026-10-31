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
"""Operator and effect classification shared by the overlap schedulers."""

import operator

from torch import fx
from torch.ops import _c10d_functional, aten


def is_op(node: fx.Node, packet: object) -> bool:
    """Match both functional collective packets and their default overloads."""
    return node.op == "call_function" and node.target in (packet, packet.default)


def is_view(node: fx.Node) -> bool:
    """Recognize immutable aliases, including tuple projections of split views."""
    if node.op != "call_function":
        return False
    if node.target is operator.getitem:
        return True
    schema = getattr(node.target, "_schema", None)
    return schema is not None and not schema.is_mutable and any(
        result.alias_info is not None for result in schema.returns
    )


def is_collective(node: fx.Node) -> bool:
    """Recognize functional collective launches, excluding their waits."""
    return node.op == "call_function" and (
        str(node.target).startswith(("_c10d_functional.", "_c10d_functional_autograd."))
        and not is_op(node, _c10d_functional.wait_tensor)
    )


def is_barrier(node: fx.Node) -> bool:
    """Bound reordering at mutations and unknown effects.

    Tagged FSDP storage releases are handled by the owning scheduler's data
    dependencies; unrelated parameter releases need not serialize communication.
    """
    if node.op in ("placeholder", "get_attr", "output") or is_collective(node):
        return False
    if is_op(node, _c10d_functional.wait_tensor):
        return False
    if node.meta.get("comm_type") in ("fsdp_reshard_free", "fsdp_reshard_free_backward"):
        return False
    if node.op != "call_function":
        return True
    schema = getattr(node.target, "_schema", None)
    if schema is not None:
        return schema.is_mutable
    return node.target is not operator.getitem


def rs_layout_preparation(launch: fx.Node) -> list[fx.Node]:
    """Recognize a private default contiguous input owned by one backward RS.

    This read-only predicate is shared by metadata, planning, rewrite and overlap.
    Shared, explicitly forward, non-default or differently owned nodes stay put.

    Args:
        launch: Reduce-scatter launch whose input preparation is inspected.
    """
    inputs = launch.all_input_nodes
    if len(inputs) != 1:
        return []
    source = inputs[0]
    if set(source.users) != {launch} or source.meta.get("autograd_backward") is False:
        return []
    owner = source.meta.get("fsdp_reduce_scatter_prepare")
    if owner is not None and owner != launch.name:
        return []
    if len(source.args) != 1 or not isinstance(source.args[0], fx.Node) or source.kwargs:
        return []
    if source.op == "call_method" and source.target == "contiguous":
        return [source]
    if is_op(source, aten.contiguous) and owner == launch.name:
        return [source]
    return []


def bucket_preparation(launch: fx.Node, order: dict[fx.Node, int]) -> list[fx.Node]:
    """Move only the private, pure pack DAG emitted by the AG/RS bucket transforms.

    Args:
        launch: Bucket collective owning the pack nodes.
        order: Current graph node positions for topological ordering.
    """
    bucket_id = launch.meta.get("fsdp_bucket_id")
    if not bucket_id:
        return []
    preparation = set()
    stack = list(launch.all_input_nodes)
    while stack:
        node = stack.pop()
        if node.meta.get("fsdp_bucket_prepare") != bucket_id:
            continue
        schema = getattr(node.target, "_schema", None)
        if schema is None or schema.name not in ("aten::reshape", "aten::cat", "aten::contiguous"):
            return []
        if node not in preparation:
            preparation.add(node)
            stack.extend(node.all_input_nodes)
    if any(set(node.users) - preparation - {launch} for node in preparation):
        return []
    return sorted(preparation, key=order.get)
