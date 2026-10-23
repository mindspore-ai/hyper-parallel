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
"""Prefetch FSDP parameters in a joint forward/backward FX graph.

Adapted from https://gitcode.com/Ascend/pytorch/pull/38283/diffs:
group gathers by their first real consumer, launch ahead of compute, and sink
waits through parameter views. HyperParallel identifies gathers by FSDPPass
metadata rather than by placeholder inputs (which would also match TP).
Forward and backward are scheduled separately; storage-release boundaries
remain explicit constraints. Like manual overlap scheduling, collective
reordering is part of this pass and requires consistent participating graphs.
"""

__all__ = ["overlap_param_gather"]

import logging
from dataclasses import dataclass

from torch import fx
from torch.ops import _c10d_functional

from ...pass_config import PassConfig
from .utils import is_barrier, is_op, is_view

_LOG = logging.getLogger(__name__)
_GATHER_TYPES = ("fsdp_all_gather", "fsdp_all_gather_backward")
_HEAVY_OPS = {"aten.mm", "aten.addmm", "aten.bmm", "aten.matmul", "aten.linear"}

# Mutable module policies keep graph consumer lookahead separate from eager FSDP configuration.
fsdp_forward_prefetch_depth: int = 1
fsdp_backward_prefetch_depth: int = 1


@dataclass
class _Gather:
    """One tagged FSDP gather and its complete wait/view realization DAG."""

    launch: fx.Node
    realization: list[fx.Node]
    first_user: fx.Node


def _indices(graph: fx.Graph) -> dict[fx.Node, int]:
    return {node: index for index, node in enumerate(graph.nodes)}


def _is_attention(node: fx.Node) -> bool:
    return node.op == "call_function" and any(
        name in str(node.target)
        for name in ("npu_fusion_attention", "scaled_dot_product", "flash_attention")
    )


def _is_heavy(node: fx.Node) -> bool:
    name = str(node.target)
    return node.op == "call_function" and (
        name.rsplit(".", 1)[0] in _HEAVY_OPS or _is_attention(node)
    )


def _collect_gathers(graph: fx.Graph) -> list[_Gather]:
    """Collect tagged parameter gathers, wait/alias DAGs, and first consumers."""
    order = _indices(graph)
    matches = []
    for launch in graph.nodes:
        if launch.meta.get("comm_type") not in _GATHER_TYPES:
            continue
        if not is_op(launch, _c10d_functional.all_gather_into_tensor):
            continue
        waits = list(launch.users)
        if len(waits) != 1 or not is_op(waits[0], _c10d_functional.wait_tensor):
            continue
        realization = {waits[0]}
        stack = [waits[0]]
        consumers = set()
        while stack:
            current = stack.pop()
            for user in current.users:
                if is_view(user):
                    if user not in realization:
                        realization.add(user)
                        stack.append(user)
                else:
                    consumers.add(user)
        if consumers:
            matches.append(_Gather(launch, sorted(realization, key=order.get), min(consumers, key=order.get)))
    return matches


def _launch_chain(launch: fx.Node) -> list[fx.Node]:
    """A private shard cast may move with the launch, as in the reference PR."""
    shard = launch.args[0]
    if isinstance(shard, fx.Node) and len(shard.users) == 1 and shard.op == "call_function":
        schema = getattr(shard.target, "_schema", None)
        if schema is not None and schema.name in ("prims::convert_element_type", "aten::_to_copy"):
            if shard.all_input_nodes and all(node.op == "placeholder" for node in shard.all_input_nodes):
                return [shard, launch]
    return [launch]


def _gather_nbytes(launch: fx.Node) -> int | None:
    nbytes = launch.meta.get("fsdp_unsharded_nbytes")
    return nbytes if isinstance(nbytes, int) and not isinstance(nbytes, bool) and nbytes >= 0 else None


def _within_budget(
    match: _Gather,
    target: fx.Node,
    prefetched: list[_Gather],
    order: dict[fx.Node, int],
    budget: int,
) -> bool:
    """Bound full replicas launched early but not yet consumed, across anchors."""
    nbytes = _gather_nbytes(match.launch)
    if nbytes is None:
        return False
    events = [(2 * order[target] - 1, nbytes), (2 * order[match.first_user], -nbytes)]
    for previous in prefetched:
        if previous is match:
            continue
        previous_bytes = _gather_nbytes(previous.launch)
        if previous_bytes is None:
            return False
        events.extend(((2 * order[previous.launch], previous_bytes),
                       (2 * order[previous.first_user], -previous_bytes)))
    live_bytes = 0
    for _, delta in sorted(events):
        live_bytes += delta
        if live_bytes > budget:
            return False
    return True


def _add_attention_interval_anchors(
    groups: dict[fx.Node, list[_Gather]],
    nodes: list[fx.Node],
    order: dict[fx.Node, int],
    anchors: dict[fx.Node, fx.Node],
) -> None:
    """Merge earlier attention-interval candidates into the lookahead anchors."""
    users = list(groups)
    attention = [node for node in nodes if _is_attention(node)]
    boundaries = [-1, *[order[node] for node in attention], len(order)]
    for index in range(len(boundaries) - 1):
        left, right = boundaries[index:index + 2]
        heavy = [node for node in nodes if _is_heavy(node) and max(left, 0) <= order[node] <= right]
        if not attention or not heavy:
            continue
        end = boundaries[index + 2] if index + 2 < len(boundaries) else right
        targets = [user for user in users if right < order[user] <= end]
        if index == 0:
            targets = [user for user in users if left < order[user] <= right] + targets
        for target_index, user in enumerate(targets):
            anchor = heavy[target_index * len(heavy) // len(targets)]
            for match in groups[user]:
                existing = anchors.get(match.launch, user)
                if order[anchor] < order[existing]:
                    anchors[match.launch] = anchor


def _prefetch_anchors(
    matches: list[_Gather], nodes: list[fx.Node], depth: int, order: dict[fx.Node, int]
) -> dict[fx.Node, fx.Node]:
    """Combine consumer lookahead with the PR's attention-interval anchors."""
    groups: dict[fx.Node, list[_Gather]] = {}
    for match in sorted(matches, key=lambda item: order[item.first_user]):
        groups.setdefault(match.first_user, []).append(match)
    users = list(groups)
    anchors = {}
    for index, user in enumerate(users):
        if index > 0:
            for match in groups[user]:
                anchors[match.launch] = users[max(0, index - depth)]

    _add_attention_interval_anchors(groups, nodes, order, anchors)
    return anchors


def _launch_constraints(graph: fx.Graph) -> dict[fx.Node, list[fx.Node]]:
    """Preserve effects and parameter lifetimes during collective reordering."""
    constraints = {}
    previous_effect = None
    frees = {}
    for node in graph.nodes:
        if node.meta.get("comm_type") in _GATHER_TYPES:
            predecessors = [previous_effect, frees.get(node.meta.get("param_name"))]
            constraints[node] = [item for item in predecessors if item is not None]
        if is_barrier(node):
            previous_effect = node
        if node.meta.get("comm_type") in ("fsdp_reshard_free", "fsdp_reshard_free_backward"):
            frees[node.meta.get("param_name")] = node
    return constraints


def _prefetch_phase(
    graph: fx.Graph, matches: list[_Gather], depth: int, budget: int, constraints: dict[fx.Node, list[fx.Node]]
) -> int:
    """Prefetch one phase within input, effect, collective, and byte constraints."""
    if not matches or depth == 0 or budget == 0:
        return 0
    backward = bool(matches[0].launch.meta.get("autograd_backward", False))
    nodes = [node for node in graph.nodes if node.op not in ("placeholder", "get_attr")
             and bool(node.meta.get("autograd_backward", False)) == backward]
    order = _indices(graph)
    phase_start = nodes[0]
    anchors = _prefetch_anchors(matches, nodes, depth, order)
    prefetched = [match for match in matches if "fsdp_prefetch_anchor" in match.launch.meta]
    changed = 0
    for match in matches:
        anchor = anchors.get(match.launch)
        if anchor is None:
            continue
        order = _indices(graph)
        chain = _launch_chain(match.launch)
        predecessors = list(constraints[match.launch])
        predecessors.extend(arg for node in chain for arg in node.all_input_nodes if arg not in chain)
        position = max(order[anchor], order[phase_start], *(order[node] + 1 for node in predecessors))
        if position >= order[chain[0]]:
            continue
        target = list(graph.nodes)[position]
        if not _within_budget(match, target, prefetched, order, budget):
            continue
        for node in chain:
            target.prepend(node)
        match.launch.meta["fsdp_prefetch_anchor"] = anchor.name
        if match not in prefetched:
            prefetched.append(match)
        changed += 1
    return changed


def _sink_realization(graph: fx.Graph, matches: list[_Gather]) -> int:
    """Sink every branch in reverse topological order, keeping all users valid."""
    changed = 0
    for match in matches:
        for node in reversed(match.realization):
            if not node.users:
                continue
            nodes = list(graph.nodes)
            order = {item: index for index, item in enumerate(nodes)}
            target = min(node.users, key=order.get)
            # Other parameter frees may be crossed, but this storage's free
            # is already a user of its realization and bounds the sinking.
            for between in nodes[order[node] + 1:order[target]]:
                if is_barrier(between) or (
                    bool(between.meta.get("autograd_backward", False))
                    != bool(match.launch.meta.get("autograd_backward", False))
                ):
                    target = between
                    break
            if order[target] > order[node] + 1:
                target.prepend(node)
                changed += 1
    return changed


def overlap_param_gather(graph_module: fx.GraphModule, pass_config: PassConfig) -> fx.GraphModule:
    """Schedule tagged FSDP gathers without changing tensor or collective semantics.

    Args:
        graph_module: Joint graph after FSDPPass (including reshard/rematerialization).
        pass_config: Independent prefetch switch and maximum early replicas in GiB.

    Returns:
        The transformed graph. Unknown replica sizes are conservatively not prefetched.
        The budget covers early launches until first use, not total model/activation memory.

    Note:
        Collective reordering requires identical static graphs and
        configuration across participating ranks. Only FSDP gathers move; other
        launches retain their relative order. Input, effect and free boundaries
        still apply. Module variables fsdp_forward_prefetch_depth and
        fsdp_backward_prefetch_depth set consumer-group lookahead (default 1).
        Patch them consistently on all ranks before compilation; existing
        compiled graphs are unaffected. Zero disables early gathers in that
        phase, including attention-interval prefetch, but still sinks waits/views.
    """
    if not pass_config.fsdp_allgather_prefetch:
        return graph_module
    depths = (fsdp_forward_prefetch_depth, fsdp_backward_prefetch_depth)
    for phase, depth in zip(("forward", "backward"), depths):
        if not isinstance(depth, int) or isinstance(depth, bool) or depth < 0:
            raise ValueError(f"fsdp_{phase}_prefetch_depth must be a non-negative integer")
    graph = graph_module.graph
    graph.lint()
    matches = _collect_gathers(graph)
    constraints = _launch_constraints(graph)
    budget = int(pass_config.max_in_flight_gb * 1024 * 1024 * 1024)
    counts = []
    for backward, depth in zip((False, True), depths):
        phase_matches = [match for match in matches
                         if bool(match.launch.meta.get("autograd_backward", False)) == backward]
        counts.append(_prefetch_phase(graph, phase_matches, depth, budget, constraints))
    sunk = _sink_realization(graph, matches)
    graph.lint()
    graph_module.recompile()
    graph_module.meta["fsdp_prefetch"] = {"forward": counts[0], "backward": counts[1], "sunk": sunk}
    _LOG.info("FSDP parameter prefetch: forward=%s backward=%s sunk=%s budget_bytes=%s depths=%s",
              counts[0], counts[1], sunk, budget, depths)
    return graph_module
