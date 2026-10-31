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
from torch.ops import _c10d_functional, aten

from ...pass_config import PassConfig
from .utils import bucket_preparation, is_barrier, is_collective, is_op, is_view, rs_layout_preparation

_LOG = logging.getLogger(__name__)
_PARAM_GATHERS = ("fsdp_all_gather", "fsdp_all_gather_backward")
_STORAGE_RELEASES = ("fsdp_reshard_free", "fsdp_reshard_free_backward")


@dataclass
class _Reduction:
    """A tagged launch and the full alias DAG materializing its waited result."""

    launch: fx.Node
    realization: list[fx.Node]
    input_aliases: set[fx.Node]
    preparation: list[fx.Node]


def _is_contiguous_method(node: fx.Node) -> bool:
    return node.op == "call_method" and node.target == "contiguous"


def _alias_ancestors(inputs: list[fx.Node]) -> set[fx.Node]:
    """Track input storage through immutable aliases, without traversing compute."""
    aliases = set(inputs)
    stack = list(inputs)
    while stack:
        node = stack.pop()
        # Even an unmovable/shared Tensor.contiguous may return its input.
        if is_view(node) or _is_contiguous_method(node):
            for arg in node.all_input_nodes:
                if arg not in aliases:
                    aliases.add(arg)
                    stack.append(arg)
    return aliases


def _is_input_release(node: fx.Node, match: _Reduction) -> bool:
    return node.meta.get("comm_type") in _STORAGE_RELEASES and bool(
        _alias_ancestors(node.all_input_nodes) & match.input_aliases
    )


def _collect_reductions(graph: fx.Graph) -> tuple[list[_Reduction], int]:
    """Collect reductions and normalize their private method preparations."""
    order = {node: index for index, node in enumerate(graph.nodes)}
    matches = []
    normalized = 0
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
        preparation = bucket_preparation(launch, order) or rs_layout_preparation(launch)
        for source in preparation:
            if source.op == "call_method":
                # Only a matched backward RS owns this default Tensor method.
                # Its ATen schema exposes both purity and possible input aliasing.
                source.op = "call_function"
                source.target = aten.contiguous.default
                source.meta.update(autograd_backward=True, fsdp_reduce_scatter_prepare=launch.name)
                normalized += 1
        external = [arg for node in [*preparation, launch] for arg in node.all_input_nodes
                    if arg not in preparation]
        matches.append(_Reduction(launch, sorted(realization, key=order.get),
                                  _alias_ancestors([*external, *launch.all_input_nodes]), preparation))
    return matches, normalized


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
    moving = {node for match in matches for node in [*match.preparation, match.launch]}
    moves = []
    for match in matches:
        launch = match.launch
        chain = [*match.preparation, launch]
        inputs = [arg for node in chain for arg in node.all_input_nodes if arg not in chain]
        ready = max(boundaries[launch], *(order[arg] for arg in inputs))
        # Targets remain stationary until all launches have moved. A target
        # that is itself a moving reduction could invalidate dependencies.
        position = ready + 1
        while position < len(nodes) and nodes[position] in moving:
            position += 1
        if position < order[chain[0]]:
            moves.append((position, order[launch], chain))
    for position, _, chain in sorted(moves):
        target = nodes[position]
        for node in chain:
            target.prepend(node)
        chain[-1].meta["fsdp_grad_overlap_anchor"] = target.name
    return len(moves)


def _sink_waits(graph: fx.Graph, matches: list[_Reduction]) -> int:
    """Delay reduction wait/alias DAGs while protecting effects and input storage."""
    changed = 0
    order = {node: index for index, node in enumerate(graph.nodes)}
    for match in sorted(matches, key=lambda item: order[item.launch]):
        # Moving each branch separately preserves early consumers of a
        # branched view and the mandatory RS-wait-AR dependency in HSDP.
        next_realization = None
        for node in reversed(match.realization):
            if not node.users:
                continue
            nodes = list(graph.nodes)
            positions = {item: index for index, item in enumerate(nodes)}
            # Keep independent unpack branches stable across repeated scheduling.
            successors = list(node.users)
            if next_realization is not None:
                successors.append(next_realization)
            target = min(successors, key=positions.get)
            for between in nodes[positions[node] + 1:positions[target]]:
                if is_barrier(between) or _is_input_release(between, match) or (
                    between.op != "output" and not between.meta.get("autograd_backward", False)
                ):
                    target = between
                    break
            if positions[target] > positions[node] + 1:
                target.prepend(node)
                changed += 1
            next_realization = node
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
    before = {node: index for index, node in enumerate(graph.nodes)} if _LOG.isEnabledFor(logging.DEBUG) else {}
    matches, normalized = _collect_reductions(graph)
    advanced = _advance_launches(graph, matches)
    sunk = _sink_waits(graph, matches)
    graph.lint()
    if normalized or advanced or sunk:
        graph_module.recompile()
    if _LOG.isEnabledFor(logging.DEBUG):
        after = {node: index for index, node in enumerate(graph.nodes)}
        for match in matches:
            launch, wait = match.launch, match.realization[0]
            _LOG.debug("FSDP gradient overlap completed: bucket=%s launch=%s index_before=%s index_after=%s "
                       "wait=%s wait_before=%s wait_after=%s",
                       launch.meta.get("fsdp_bucket_id"), launch.name, before.get(launch), after[launch],
                       wait.name, before.get(wait), after[wait])
    graph_module.meta["fsdp_grad_overlap"] = {
        "matched": len(matches), "launches_advanced": advanced, "wait_view_nodes_sunk": sunk,
    }
    _LOG.info("FSDP gradient overlap: matched=%s launches_advanced=%s wait_view_nodes_sunk=%s",
              len(matches), advanced, sunk)
    return graph_module
