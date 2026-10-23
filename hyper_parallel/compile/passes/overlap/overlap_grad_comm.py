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
"""Overlap FSDP gradient reduce-scatter with the remaining backward compute.

Adapted from https://gitcode.com/Ascend/pytorch/pull/37311: defer gradient
waits until consumption. Parameter all-gather lookahead is already provided
by overlap_param_gather. FSDPPass emits reductions at the joint graph's tail,
so this pass also launches each reduction as soon as its gradient is ready.
FSDP metadata handles decomposed matmuls, norm and accumulated/tied gradients
without guessing which output of an operator is a weight gradient.
"""

__all__ = ["overlap_grad_comm"]

import logging
from dataclasses import dataclass

from torch import fx
from torch.ops import _c10d_functional

from ...pass_config import PassConfig
from .utils import is_barrier, is_collective, is_op, is_view

_LOG = logging.getLogger(__name__)
_PARAM_GATHERS = ("fsdp_all_gather", "fsdp_all_gather_backward")
_STORAGE_RELEASES = ("fsdp_reshard_free", "fsdp_reshard_free_backward")


@dataclass
class _Reduction:
    """A tagged launch and the full alias DAG materializing its waited result."""

    launch: fx.Node
    realization: list[fx.Node]
    input_aliases: set[fx.Node]


def _alias_ancestors(inputs: list[fx.Node]) -> set[fx.Node]:
    """Track input storage through immutable aliases, without traversing compute."""
    aliases = set(inputs)
    stack = list(inputs)
    while stack:
        node = stack.pop()
        if is_view(node):
            for arg in node.all_input_nodes:
                if arg not in aliases:
                    aliases.add(arg)
                    stack.append(arg)
    return aliases


def _is_input_release(node: fx.Node, match: _Reduction) -> bool:
    return node.meta.get("comm_type") in _STORAGE_RELEASES and bool(
        _alias_ancestors(node.all_input_nodes) & match.input_aliases
    )


def _collect_reductions(graph: fx.Graph) -> list[_Reduction]:
    """Collect tagged backward reductions and their complete wait/alias DAGs."""
    order = {node: index for index, node in enumerate(graph.nodes)}
    matches = []
    for launch in graph.nodes:
        if launch.meta.get("comm_type") != "fsdp_reduce_scatter":
            continue
        if not launch.meta.get("autograd_backward", False):
            continue
        if not is_op(launch, _c10d_functional.reduce_scatter_tensor):
            continue
        users = list(launch.users)
        if len(users) != 1 or not is_op(users[0], _c10d_functional.wait_tensor):
            continue
        realization = {users[0]}
        stack = [users[0]]
        while stack:
            node = stack.pop()
            for user in node.users:
                if is_view(user) and user not in realization:
                    realization.add(user)
                    stack.append(user)
        matches.append(_Reduction(launch, sorted(realization, key=order.get),
                                  _alias_ancestors(launch.all_input_nodes)))
    return matches


def _group(node: fx.Node) -> object:
    return node.meta.get("comm_group", node.kwargs.get("group_name", node.args[-1] if node.args else None))


def _launch_boundaries(nodes: list[fx.Node], matches: list[_Reduction]) -> dict[fx.Node, int]:
    """Find the last effect, phase boundary or unowned same-group collective.

    Owned FSDP gathers and reductions may interleave in a new deterministic
    order. All other launches retain their relative order, including TP.
    A same-group collective not owned by this scheduler remains a barrier.
    """
    launches = {match.launch: match for match in matches}
    groups = {_group(node) for node in launches}
    last_effect = -1
    last_unowned = dict.fromkeys(groups, -1)
    boundaries = {}
    releases = []
    for index, node in enumerate(nodes):
        if node in launches:
            related = [position for position, release in releases if _is_input_release(release, launches[node])]
            boundaries[node] = max(last_effect, last_unowned[_group(node)], *related)
        elif node.op not in ("placeholder", "get_attr", "output"):
            if node.meta.get("comm_type") in _STORAGE_RELEASES:
                releases.append((index, node))
            if is_barrier(node) or not node.meta.get("autograd_backward", False):
                last_effect = index
            if is_collective(node) and node.meta.get("comm_type") not in _PARAM_GATHERS:
                group = _group(node)
                if group in last_unowned:
                    last_unowned[group] = index
    return boundaries


def _advance_launches(graph: fx.Graph, matches: list[_Reduction]) -> int:
    """Launch each reduction after its inputs and communication boundaries are ready."""
    nodes = list(graph.nodes)
    order = {node: index for index, node in enumerate(nodes)}
    boundaries = _launch_boundaries(nodes, matches)
    launches = {match.launch for match in matches}
    moves = []
    for match in matches:
        launch = match.launch
        ready = max(boundaries[launch], *(order[arg] for arg in launch.all_input_nodes))
        # Targets remain stationary until all launches have moved. A target
        # that is itself a moving reduction could invalidate dependencies.
        position = ready + 1
        while position < len(nodes) and nodes[position] in launches:
            position += 1
        if position < order[launch]:
            moves.append((position, order[launch], launch))
    for position, _, launch in sorted(moves):
        target = nodes[position]
        target.prepend(launch)
        launch.meta["fsdp_grad_overlap_anchor"] = target.name
    return len(moves)


def _sink_waits(graph: fx.Graph, matches: list[_Reduction]) -> int:
    """Delay reduction wait/alias DAGs while protecting effects and input storage."""
    changed = 0
    order = {node: index for index, node in enumerate(graph.nodes)}
    for match in sorted(matches, key=lambda item: order[item.launch]):
        # Moving each branch separately preserves early consumers of a
        # branched view and the mandatory RS-wait-AR dependency in HSDP.
        for node in reversed(match.realization):
            if not node.users:
                continue
            nodes = list(graph.nodes)
            positions = {item: index for index, item in enumerate(nodes)}
            target = min(node.users, key=positions.get)
            for between in nodes[positions[node] + 1:positions[target]]:
                if is_barrier(between) or _is_input_release(between, match) or (
                    between.op != "output" and not between.meta.get("autograd_backward", False)
                ):
                    target = between
                    break
            if positions[target] > positions[node] + 1:
                target.prepend(node)
                changed += 1
    return changed


def overlap_grad_comm(graph_module: fx.GraphModule, pass_config: PassConfig) -> fx.GraphModule:
    """Advance tagged gradient launches and sink waits through immutable aliases.

    Args:
        graph_module: Joint graph after FSDPPass and parameter prefetch.
        pass_config: fsdp_overlap_grad_reduce controls this pass independently.

    Returns:
        The same graph module, with scheduling statistics in fsdp_grad_overlap.

    Note:
        FSDP collective order changes to static gradient readiness order. As
        with the original static FSDP graph, participating ranks must execute
        the same graph and configuration. No tensor-dependent decisions occur.
        TP launches, tensor arithmetic, gradient averaging and outputs are unchanged.
    """
    if not pass_config.fsdp_overlap_grad_reduce:
        return graph_module
    graph = graph_module.graph
    graph.lint()
    matches = _collect_reductions(graph)
    advanced = _advance_launches(graph, matches)
    sunk = _sink_waits(graph, matches)
    graph.lint()
    if advanced or sunk:
        graph_module.recompile()
    graph_module.meta["fsdp_grad_overlap"] = {
        "matched": len(matches), "launches_advanced": advanced, "wait_view_nodes_sunk": sunk,
    }
    _LOG.info("FSDP gradient overlap: matched=%s launches_advanced=%s wait_view_nodes_sunk=%s",
              len(matches), advanced, sunk)
    return graph_module
