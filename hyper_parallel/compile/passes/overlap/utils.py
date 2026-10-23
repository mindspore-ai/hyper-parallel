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
from torch.ops import _c10d_functional


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
