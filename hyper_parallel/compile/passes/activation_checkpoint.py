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
"""Activation Checkpoint Pass - static-graph activation checkpointing.

FX-level activation recomputation for the joint forward+backward graph,
following torchtitan graph_trainer's ``tag_with_memory_policy_pass`` +
``selective_activation_remat_pass`` design while reusing the dynamic-graph
configuration vocabulary (``ActivationCheckpointConfig``: ``mode`` =
off/full/selective, layer selection, model-adapter safe regions):

1. Tag: forward ``call_function`` nodes inside the checkpoint scope get a
   ``CheckpointPolicy`` in ``node.meta["recompute"]``. The scope is the
   discovered transformer blocks (the same HuggingFace-marker discovery the
   eager path requires), narrowed by the optional layer selectors and the
   model adapter's safe regions. Nodes outside the blocks (embedding, head,
   the ``train_fn`` loss chain) and compute/communication intensive
   operators under ``selective`` stay saved; everything else is a recompute
    candidate (``full`` forces every scoped node). Structural exceptions are
    always saved: RNG ops (the graph cannot replay random state), in-place
    mutations with their consumers and the readers of every tensor sharing
    the mutated storage (including view aliases), SymInt scalars.
2. Remat: every tagged node that the backward half consumes (transitively
   through other tagged nodes) is duplicated in front of its earliest
   backward consumer and the backward consumers are redirected to the
   duplicate. A forward original with no remaining readers is erased; one
   still read by forward code stays, and its backward readers now rebuild
   the value during backward. Either way the activation's forward lifetime
   ends at its last forward use instead of spanning the whole step. A
   cross-unit forward consumer forces the producing node to stay saved
   (recreating it would only waste compute), which keeps each recompute
   chain inside one block.

The backward half is identified by ``meta["autograd_backward"]``, which the
joint tracer guarantees on every supported torch (engine tag on patched
torch, structural reconstruction otherwise -- see
``tracer/graph_tracer.py``).

Ordering contract: the pass must run BEFORE ``FSDPPass``. Recomputed
duplicates are ordinary forward ops at tag time, so ``FSDPPass`` inserts
parameter all-gathers and the backward re-gathers around them correctly
(duplicates are tagged ``autograd_backward`` and count as backward
consumers). Running AC after FSDP would make recompute read sharded
placeholders. ``PpPass`` runs later and inherits the rewrite value-level.

Operator policy sets and transformer-block discovery are shared with the
eager path (``hyper_parallel.distributed.activation_checkpoint``), so a
model sees the same save/recompute vocabulary in both modes. RNG ops are
saved in every mode (a graph cannot replay random state). Communication
ops are saved too, except that ``ac_save_forward_all_gather`` may opt
forward (TP/SP/CP) all-gathers into backward replay.
"""

from __future__ import annotations

__all__ = ["ActivationCheckpointPass", "prepare_activation_checkpoint"]

import functools
import logging
import operator
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
from torch import fx
from torch._functorch import compile_utils
from torch._ops import OpOverload
from torch.utils.checkpoint import CheckpointPolicy

# Same-package reuse of the eager AC machinery: operator save/recompute sets,
# HF transformer-block discovery and model-adapter recompute specs. They are
# module-level privates because the eager module never exported them, but the
# coupling is the point: one policy vocabulary for both modes.
from hyper_parallel.distributed.activation_checkpoint import (
    _SELECTIVE_AC_COMM_OP_NAMES,
    _SELECTIVE_AC_FORCE_RECOMPUTE_OPS,
    _SELECTIVE_AC_MUST_SAVE_OPS,
    _adapter_recompute_spec,
    _existing_ops,
    _find_transformer_layer_container_infos,
    _resolve_op_attr,
    _resolve_torch_op,
    _select_excluded_targets,
    _select_layer_region_roots,
    _selected_checkpoint_layer_fqns,
)

from ..pass_config import PassConfig
from .base import GraphPass

_LOG = logging.getLogger(__name__)

# FSDP all-gathers are inserted after this pass, but forward (TP/SP/CP)
# all-gathers are already in the traced forward; saving them avoids a second
# collective during backward recompute. ``PassConfig.ac_save_forward_all_gather``
# selects that superset.
_GRAPH_AC_COMM_OPS = _existing_ops(
    *(_resolve_torch_op(name) for name in _SELECTIVE_AC_COMM_OP_NAMES),
    _resolve_op_attr(torch.ops, "deepep.dispatch.default"),
    _resolve_op_attr(torch.ops, "deepep.combine.default"),
    _resolve_op_attr(torch.ops, "hybridep.dispatch.default"),
    _resolve_op_attr(torch.ops, "hybridep.combine.default"),
)
_GRAPH_FORWARD_ALL_GATHER_OPS = _existing_ops(
    _resolve_torch_op("aten.all_gather_into_tensor"),
    _resolve_torch_op("_c10d_functional.all_gather_into_tensor"),
)


def _resolve_wait_tensor() -> Optional[Any]:
    """Return the c10d ``wait_tensor`` overload when this torch ships it."""
    functional = getattr(torch.ops, "_c10d_functional", None)
    wait = getattr(functional, "wait_tensor", None)
    if wait is None:
        return None
    return getattr(wait, "default", None)


# getitem / wait_tensor extract or await their producer's value: their
# save/recompute decision is inherited, and remat walks through them.
_TRANSPARENT_OPS: Tuple[Any, ...] = tuple(
    target
    for target in (operator.getitem, _resolve_wait_tensor())
    if target is not None
)

_RECOMPUTE_POLICIES = (
    CheckpointPolicy.PREFER_RECOMPUTE,
    CheckpointPolicy.MUST_RECOMPUTE,
)


def _must_recompute(node: fx.Node) -> bool:
    """Whether ``node`` carries a policy the partitioner would recompute."""
    return node.meta.get("recompute") in _RECOMPUTE_POLICIES


def _node_stack_fqns(node: fx.Node) -> List[str]:
    """Module FQNs on ``node``'s ``nn_module_stack``, innermost first."""
    stack = node.meta.get("nn_module_stack") or {}
    return [fqn for fqn, _ in reversed(list(stack.values()))]


def _at_or_under(fqn: str, root: str) -> bool:
    """Whether ``fqn`` equals ``root`` or lives below it."""
    return fqn == root or fqn.startswith(f"{root}.")


def _first_container(fqns: List[str], roots: frozenset) -> Optional[str]:
    """Return the first root that contains one of ``fqns`` (innermost wins).

    ``roots`` is sorted so a (currently impossible) nested-root match is still
    deterministic. ``fqns`` is innermost-first, so the first hit is the
    innermost containing root.
    """
    for fqn in fqns:
        for root in sorted(roots):
            if _at_or_under(fqn, root):
                return root
    return None


class _CheckpointScope:
    """Resolved recompute scope for one compilation.

    ``_resolve_scope`` always fills every field; the ``None`` defaults exist
    for direct construction in tests and mean "no region" / "no boundary
    units", which makes every node ineligible / disables the cross-unit rule.

    Attributes:
        region_fqns: FQNs whose subtree may recompute. Empty (or ``None``)
            makes every node ineligible (the pass becomes a no-op).
        exclude_fqns: FQNs whose subtree must stay saved (adapter exclusions).
        unit_fqns: Boundary units for the cross-unit save rule. Empty (or
            ``None``) disables the rule.
        on_applied: Adapter runtime hook that must run before the model's
            forward is traced, when the adapter's region shape is in effect.
    """

    def __init__(
        self,
        region_fqns: Optional[frozenset] = None,
        exclude_fqns: frozenset = frozenset(),
        unit_fqns: Optional[frozenset] = None,
        on_applied: Optional[Callable[[torch.nn.Module], None]] = None,
    ) -> None:
        """Store the resolved scope membership sets."""
        self.region_fqns = region_fqns
        self.exclude_fqns = exclude_fqns
        self.unit_fqns = unit_fqns
        self.on_applied = on_applied


def _adapter_scope(
    model: torch.nn.Module,
    selected_layers: frozenset,
    unit_fqns: frozenset,
    recompute_spec: Any,
) -> _CheckpointScope:
    """Build a scope from the adapter-declared recompute regions."""
    module_by_fqn = dict(model.named_modules())
    roots = _select_layer_region_roots(module_by_fqn, selected_layers, recompute_spec)
    if not roots:
        raise ValueError(
            "model adapter recompute patterns matched no region in the selected layers"
        )
    excluded = _select_excluded_targets(module_by_fqn, roots, recompute_spec)
    return _CheckpointScope(
        region_fqns=frozenset(fqn for fqn, _ in roots),
        exclude_fqns=frozenset(fqn for fqn, _ in excluded),
        unit_fqns=unit_fqns,
        on_applied=recompute_spec.on_applied,
    )


def _resolve_scope(
    model: Optional[torch.nn.Module], pass_config: PassConfig
) -> _CheckpointScope:
    """Resolve the checkpoint scope from the dynamic-config vocabulary.

    The recompute scope and the boundary units are the discovered
    transformer blocks — the same HuggingFace-marker discovery the eager
    path requires (``_resolve_activation_checkpoint_containers``), so a
    model is AC-eligible in both modes under the same contract. Nodes
    outside the blocks (embedding, head, the ``train_fn`` loss chain) are
    never tagged, which also keeps the head/epilogue backward from pulling
    the whole forward into one recompute chain. With
    ``ac_layer_count`` / ``ac_layer_indices`` only the selected blocks
    recompute; an adapter ``RecomputePolicy`` (``model_adapter_safe_regions``
    in the dynamic config) narrows the selected blocks to its region
    patterns and pulls ``exclude_patterns`` back out of the recompute.

    ``full`` with the default selection mirrors the eager auto-upgrade: when
    the model owns a ``RecomputePolicy``, all blocks are selected through its
    regions instead of recomputing every op in the blocks.

    Raises:
        ValueError: When the live model is missing, no transformer blocks
            are discovered, an explicitly requested safe-region selection has
            no adapter, or the adapter declares no recompute region.
    """
    if model is None:
        raise ValueError(
            "ActivationCheckpointPass requires the live model via kwargs "
            "(model=...) so it can discover the transformer blocks to "
            "checkpoint"
        )
    containers = _find_transformer_layer_container_infos(model)
    if not containers:
        raise ValueError(
            f"{type(model).__name__} has no module with a 'gradient_checkpointing' "
            "attribute and a non-empty repeated block container; activation "
            "checkpointing cannot resolve its layers"
        )
    block_fqns = frozenset(
        block.fqn for container in containers for block in container.blocks
    )
    explicit_regions = pass_config.ac_region_source == "model_adapter_safe_regions"
    has_count = pass_config.ac_layer_count is not None
    has_indices = pass_config.ac_layer_indices is not None
    if not has_count and not has_indices:
        if pass_config.ac_mode == "full":
            recompute_spec = _adapter_recompute_spec(model, required=explicit_regions)
            if recompute_spec is not None:
                if not recompute_spec.region_patterns:
                    raise ValueError(
                        "model adapter recompute requires "
                        "RecomputePolicy.region_patterns"
                    )
                return _adapter_scope(model, block_fqns, block_fqns, recompute_spec)
        return _CheckpointScope(region_fqns=block_fqns, unit_fqns=block_fqns)
    selection = SimpleNamespace(
        layer_count=pass_config.ac_layer_count,
        layer_indices=pass_config.ac_layer_indices,
    )
    selected_layers = frozenset(_selected_checkpoint_layer_fqns(containers, selection))
    if not selected_layers:
        # Zero selected layers is a no-op, not an error.
        _LOG.info("activation checkpoint layer selection matched zero layers")
        return _CheckpointScope(region_fqns=frozenset(), unit_fqns=block_fqns)
    recompute_spec = _adapter_recompute_spec(model, required=explicit_regions)
    if recompute_spec is None:
        if explicit_regions:
            raise ValueError(
                "activation_checkpoint.selection.source=model_adapter_safe_regions "
                "requires ModelAdapterSpec.recompute"
            )
        return _CheckpointScope(region_fqns=selected_layers, unit_fqns=block_fqns)
    if not recompute_spec.region_patterns:
        if explicit_regions:
            raise ValueError(
                "model adapter recompute requires RecomputePolicy.region_patterns"
            )
        return _CheckpointScope(region_fqns=selected_layers, unit_fqns=block_fqns)
    return _adapter_scope(model, selected_layers, block_fqns, recompute_spec)


def prepare_activation_checkpoint(
    model: Optional[torch.nn.Module], pass_config: PassConfig
) -> _CheckpointScope:
    """Resolve the AC scope and run the adapter hook before tracing.

    The adapter's ``on_applied`` hook configures runtime state the model's
    forward reads, so it must run before the joint graph is traced.
    ``GraphCompiler`` calls this once pre-trace and hands the scope to the
    pass, which avoids a second hook invocation.

    Args:
        model: The live model whose transformer blocks define the scope.
        pass_config: Parallel configuration driving ``ac_mode``.

    Returns:
        The resolved scope, including the adapter runtime hook when one is
        declared.
    """
    scope = _resolve_scope(model, pass_config)
    if scope.on_applied is not None and model is not None:
        scope.on_applied(model)
    return scope


def _is_eligible(node: fx.Node, scope: _CheckpointScope) -> bool:
    """Whether ``node`` may carry a recompute decision at all.

    Nodes outside the checkpoint regions stay saved: activation
    checkpointing targets transformer-block forwards, mirroring the eager
    path that wraps the discovered blocks only.
    """
    if not scope.region_fqns:
        return False
    fqns = _node_stack_fqns(node)
    if not fqns:
        return False
    if _first_container(fqns, scope.exclude_fqns) is not None:
        return False
    return _first_container(fqns, scope.region_fqns) is not None


def _make_policy_fn(pass_config: PassConfig) -> Callable[[fx.Node], CheckpointPolicy]:
    """Build the per-node policy for ``ac_mode``.

    ``full`` recomputes every scoped node. ``selective`` saves the shared
    compute/communication operator set and forces recompute only on the
    operators the eager policy must re-replay (in-place-unsafe outputs such
    as ``topk``); everything else prefers recompute.
    """
    if pass_config.ac_mode == "full":
        return lambda node: CheckpointPolicy.MUST_RECOMPUTE

    must_save = _SELECTIVE_AC_MUST_SAVE_OPS
    force_recompute = _SELECTIVE_AC_FORCE_RECOMPUTE_OPS

    def selective_policy(node: fx.Node) -> CheckpointPolicy:
        """Apply the shared selective-AC operator policy."""
        if node.target in force_recompute:
            return CheckpointPolicy.MUST_RECOMPUTE
        if node.target in must_save:
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    return selective_policy


def _unit_of(node: fx.Node, scope: _CheckpointScope) -> Optional[str]:
    """Return ``node``'s boundary block unit, if any."""
    if not scope.unit_fqns:
        return None
    return _first_container(_node_stack_fqns(node), scope.unit_fqns)


def _tag_policy_pass(
    graph_module: fx.GraphModule,
    pass_config: PassConfig,
    scope: _CheckpointScope,
) -> None:
    """Tag forward nodes per policy, applying the structural exceptions.

    RNG ops and SymInt scalars stay saved; getitem/wait_tensor inherit their
    parent's decision. Communication ops are saved in every mode (including
    ``full``): replaying a collective in backward changes communication
    semantics. All-gathers join the saved set when
    ``ac_save_forward_all_gather`` is enabled.
    """
    policy_fn = _make_policy_fn(pass_config)
    comm_ops = _GRAPH_AC_COMM_OPS
    if pass_config.ac_save_forward_all_gather:
        comm_ops = comm_ops | _GRAPH_FORWARD_ALL_GATHER_OPS
    for node in graph_module.graph.nodes:
        if node.op != "call_function" or node.meta.get("autograd_backward", False):
            continue
        if node.target in _TRANSPARENT_OPS:
            parent = node.args[0]
            if isinstance(parent, fx.Node) and "recompute" in parent.meta:
                node.meta["recompute"] = parent.meta["recompute"]
            continue
        if not _is_eligible(node, scope):
            continue
        if _is_rng_op(node) or node.target in comm_ops:
            node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
            continue
        value = node.meta.get("val")
        if isinstance(value, torch.SymInt):
            node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
            continue
        node.meta["recompute"] = policy_fn(node)


def _flatten_getitems(graph_module: fx.GraphModule) -> None:
    """Flatten tuple projections when this torch build provides the helper."""
    raise_getitems = getattr(compile_utils, "raise_getitems", None)
    if raise_getitems is not None:
        raise_getitems(graph_module)


def _is_mutable_op(target: Any) -> bool:
    """Return whether ``target`` is an in-place ``OpOverload``."""
    if not isinstance(target, OpOverload):
        return False
    return bool(getattr(getattr(target, "_schema", None), "is_mutable", False))


def _is_rng_op(node: fx.Node) -> bool:
    """Return True when a node's seeded nondeterminism cannot be ruled out."""
    return bool(
        hasattr(node.target, "tags")
        and torch.Tag.nondeterministic_seeded in node.target.tags
    )


def _mutated_arg_nodes(node: fx.Node) -> List[fx.Node]:
    """Return the input nodes whose tensors ``node`` overwrites in place."""
    schema = getattr(node.target, "_schema", None)
    if schema is None:
        return []
    mutated: List[fx.Node] = []
    for position, schema_arg in enumerate(schema.arguments):
        alias_info = getattr(schema_arg, "alias_info", None)
        if alias_info is None or not alias_info.is_write:
            continue
        arg = (
            node.args[position]
            if position < len(node.args)
            else node.kwargs.get(schema_arg.name)
        )
        if isinstance(arg, fx.Node):
            mutated.append(arg)
    return mutated


def _aliased_input_nodes(node: fx.Node) -> List[fx.Node]:
    """Return input nodes whose storage ``node``'s outputs may share.

    Alias relationships come from the operator schema. A return carrying
    ``alias_info`` aliases the arguments whose alias sets intersect; several
    multi-output views (``split`` / ``chunk`` / ``unbind``) instead mark the
    input with an ``after_set`` and leave the returns unannotated. In-place
    writes always share their argument's storage. ``getitem`` picks one output
    of a producer, so it inherits the producer's storage even though the
    builtin itself has no schema.
    """
    if node.op != "call_function":
        return []
    if node.target is operator.getitem:
        producer = node.args[0] if node.args else None
        return [producer] if isinstance(producer, fx.Node) else []
    schema = getattr(node.target, "_schema", None)
    if schema is None:
        return []
    result_names: Set[str] = set()
    for result in schema.returns:
        alias_info = getattr(result, "alias_info", None)
        if alias_info is not None:
            result_names.update(alias_info.before_set)
            result_names.update(alias_info.after_set)
    wildcard_result = "*" in result_names
    aliased: List[fx.Node] = []
    for position, schema_arg in enumerate(schema.arguments):
        alias_info = getattr(schema_arg, "alias_info", None)
        if alias_info is None:
            continue
        arg_names = set(alias_info.before_set) | set(alias_info.after_set)
        if not (
            getattr(alias_info, "is_write", False)
            or wildcard_result
            or "*" in arg_names
            or bool(arg_names & result_names)
        ):
            continue
        arg = (
            node.args[position]
            if position < len(node.args)
            else node.kwargs.get(schema_arg.name)
        )
        if isinstance(arg, fx.Node):
            aliased.append(arg)
    return aliased


def _forward_alias_neighbors(
    graph_module: fx.GraphModule,
) -> Dict[fx.Node, Set[fx.Node]]:
    """Build the undirected shared-storage graph over the forward graph."""
    neighbors: Dict[fx.Node, Set[fx.Node]] = {}
    for node in graph_module.graph.nodes:
        if node.meta.get("autograd_backward", False):
            continue
        for arg in _aliased_input_nodes(node):
            if arg.meta.get("autograd_backward", False):
                continue
            neighbors.setdefault(node, set()).add(arg)
            neighbors.setdefault(arg, set()).add(node)
    return neighbors


def _alias_closure(
    roots: Set[fx.Node], neighbors: Dict[fx.Node, Set[fx.Node]]
) -> Set[fx.Node]:
    """Return every forward node sharing storage with any of ``roots``.

    The walk is undirected so a mutation through one view reaches its base
    (backward along the alias edge) and then the sibling views of that base
    (forward along their edges).
    """
    closure = set(roots)
    stack = list(roots)
    while stack:
        current = stack.pop()
        for neighbor in neighbors.get(current, ()):
            if neighbor not in closure:
                closure.add(neighbor)
                stack.append(neighbor)
    return closure


def _tag_inplace_saves(graph_module: fx.GraphModule) -> None:
    """Force saves around in-place mutations and their aliased-storage readers.

    The mutation must run exactly once in forward, so it stays saved together
    with its forward consumers. Every forward reader of the mutated storage
    stays saved as well: a recompute duplicate would re-execute against either
    the post-mutation original or a pre-mutation duplicate of the producer,
    and neither matches what forward computed at the reader's own position.
    Eager autograd stays silent for such a read — its value reaches backward
    only through the recompute chain, never as a saved tensor. The storage
    closure follows operator-schema aliases, so readers of a view (or of a
    sibling view when the write goes through another view) are covered, and
    mutations outside the checkpoint scope still contribute their storage.
    """
    nodes = list(graph_module.graph.nodes)
    mutated_bases: Set[fx.Node] = set()
    for node in nodes:
        if (
            node.op != "call_function"
            or node.meta.get("autograd_backward", False)
            or not _is_mutable_op(node.target)
        ):
            continue
        mutated_bases.update(_mutated_arg_nodes(node))
        if "recompute" not in node.meta:
            continue
        node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
        for user in node.users:
            if (
                not user.meta.get("autograd_backward", False)
                and "recompute" in user.meta
            ):
                user.meta["recompute"] = CheckpointPolicy.MUST_SAVE
    if not mutated_bases:
        return
    aliased_storage = _alias_closure(
        mutated_bases, _forward_alias_neighbors(graph_module)
    )
    for node in nodes:
        if (
            node.op != "call_function"
            or node.meta.get("autograd_backward", False)
            or node.meta.get("recompute") not in _RECOMPUTE_POLICIES
        ):
            continue
        if not aliased_storage.isdisjoint(node.all_input_nodes):
            node.meta["recompute"] = CheckpointPolicy.MUST_SAVE


def _tag_cross_unit_saves(
    graph_module: fx.GraphModule, scope: _CheckpointScope
) -> None:
    """Save recomputable nodes read forward across a unit boundary.

    Their original activation stays alive for the forward consumer anyway, so
    recomputing them in backward would be pure waste.
    """
    for node in graph_module.graph.nodes:
        if (
            node.op != "call_function"
            or node.meta.get("autograd_backward", False)
            or node.meta.get("recompute") not in _RECOMPUTE_POLICIES
        ):
            continue
        if _has_cross_unit_forward_consumer(node, scope):
            node.meta["recompute"] = CheckpointPolicy.MUST_SAVE


def _tag_sac_policy(
    graph_module: fx.GraphModule,
    pass_config: PassConfig,
    scope: _CheckpointScope,
) -> None:
    """Annotate forward nodes with ``meta["recompute"]`` policies.

    Pass 1 tags per policy with the structural exceptions; pass 2 forces saves
    on in-place mutations, their consumers and the forward readers of the
    mutated storage (view aliases included); pass 3 forces saves on
    recomputable nodes read forward across a unit boundary (their original
    stays alive anyway, so the backward recompute would be pure waste).
    """
    if not scope.region_fqns:
        return
    _tag_policy_pass(graph_module, pass_config, scope)
    _tag_inplace_saves(graph_module)
    if scope.unit_fqns:
        _tag_cross_unit_saves(graph_module, scope)


def _has_cross_unit_forward_consumer(node: fx.Node, scope: _CheckpointScope) -> bool:
    """Whether ``node`` is read forward outside its own boundary unit.

    Transparent (inherited) getitem/wait_tensor chains are walked through;
    any other forward consumer in a different unit (or outside every unit,
    e.g. the epilogue) means the original must stay alive anyway.
    """
    node_unit = _unit_of(node, scope)
    policy = node.meta["recompute"]
    seen = {node}
    stack = [node]
    while stack:
        current = stack.pop()
        for user in current.users:
            if user in seen:
                continue
            seen.add(user)
            if user.op == "output" or user.meta.get("autograd_backward", False):
                continue
            if (
                user.op == "call_function"
                and user.target in _TRANSPARENT_OPS
                and user.meta.get("recompute") == policy
            ):
                stack.append(user)
                continue
            if _unit_of(user, scope) != node_unit:
                return True
    return False


def _duplicate_subgraph_get_attrs(
    graph_module: fx.GraphModule, fwd_node: fx.Node, bwd_target: fx.Node
) -> Dict[fx.Node, fx.Node]:
    """Give a recompute duplicate private copies of HOP subgraph get_attrs.

    Higher-order ops (flex_attention etc.) reference their subgraphs through
    ``get_attr`` nodes. Sharing the original would let a later regional
    Inductor pass place the shared subgraph in only one partition; fresh
    get_attr nodes keep each region self-contained.
    """
    duplicates: Dict[fx.Node, fx.Node] = {}
    for arg in fwd_node.all_input_nodes:
        if arg.op != "get_attr":
            continue
        if not isinstance(getattr(graph_module, arg.target, None), fx.GraphModule):
            continue
        with graph_module.graph.inserting_before(bwd_target):
            dup_attr = graph_module.graph.get_attr(arg.target)
            dup_attr.meta.update(arg.meta)
        duplicates[arg] = dup_attr
    return duplicates


def _remat(graph_module: fx.GraphModule) -> int:
    """Duplicate tagged recompute nodes in front of their backward consumers.

    Returns the number of duplicated forward nodes. The graph is mutated in
    place; originals whose consumers were all redirected are erased, and
    ``raise_getitems`` flattens output/tuple projections for better memory.
    """
    graph = graph_module.graph
    all_nodes = list(graph.nodes)
    order = {node: idx for idx, node in enumerate(all_nodes)}

    # Map each tagged forward node to its earliest backward consumer; the
    # transitive walk through tagged ancestors mirrors the partitioner's
    # recompute-closure gathering (saved ancestors stay live, so only tagged
    # ones need duplicating).
    remat_targets: Dict[fx.Node, fx.Node] = {}
    for bwd_node in all_nodes:
        if not bwd_node.meta.get("autograd_backward", False):
            continue
        seen: Set[fx.Node] = set()
        stack = list(bwd_node.all_input_nodes)
        while stack:
            current = stack.pop()
            if current in seen or not _must_recompute(current):
                continue
            seen.add(current)
            remat_targets.setdefault(current, bwd_node)
            stack.extend(current.all_input_nodes)

    if not remat_targets:
        return 0

    recomputed: Dict[fx.Node, fx.Node] = {}

    def remat_arg(arg: Any, get_attr_dups: Dict[fx.Node, fx.Node]) -> Any:
        """Redirect duplicate args to their recompute copies."""
        if isinstance(arg, fx.Node):
            if arg in get_attr_dups:
                return get_attr_dups[arg]
            if arg in recomputed:
                return recomputed[arg]
        return arg

    for fwd_node in sorted(remat_targets, key=order.__getitem__):
        bwd_target = remat_targets[fwd_node]
        get_attr_dups = _duplicate_subgraph_get_attrs(
            graph_module, fwd_node, bwd_target
        )
        with graph.inserting_before(bwd_target):
            duplicate = graph.node_copy(
                fwd_node,
                functools.partial(remat_arg, get_attr_dups=get_attr_dups),
            )
        # Leave the name to ``node_copy``/``create_node`` so the graph's name
        # namespace owns uniqueness; a direct ``.name =`` assignment bypasses
        # it and can collide with an existing node.
        duplicate.meta["autograd_backward"] = True
        duplicate.meta["ac_recomputed"] = True
        duplicate.meta["ac_recomputed_from"] = fwd_node.name
        # node_copy shares meta, including the policy tag. Drop it so a second
        # pass run cannot mistake the duplicate for a recompute candidate.
        duplicate.meta.pop("recompute", None)
        recomputed[fwd_node] = duplicate

    def redirect(arg: Any) -> Any:
        """Arg transform for backward consumers of recomputed originals."""
        if isinstance(arg, fx.Node) and arg in recomputed:
            return recomputed[arg]
        return arg

    direct_backward_consumers = {
        user
        for fwd_node in recomputed
        for user in fwd_node.users
        if user.meta.get("autograd_backward", False)
    }
    for bwd_node in direct_backward_consumers:
        bwd_node.args = fx.map_arg(bwd_node.args, redirect)
        bwd_node.kwargs = fx.map_arg(bwd_node.kwargs, redirect)

    for original in reversed(list(recomputed)):
        if not original.users:
            graph.erase_node(original)

    _flatten_getitems(graph_module)
    graph.lint()
    return len(recomputed)


class ActivationCheckpointPass(GraphPass):
    """Static-graph activation checkpointing (tag + recompute-materialize).

    Configure through ``PassConfig.ac_mode`` (``"off"`` / ``"full"`` /
    ``"selective"``) and the optional ``ac_layer_count`` /
    ``ac_layer_indices`` layer selectors. The trainer-level
    ``activation_checkpoint`` section is projected onto these fields by
    ``build_pass_config_from_trainer_config``.
    """

    name = "activation_checkpoint"

    def run(
        self,
        graph_module: fx.GraphModule,
        pass_config: PassConfig,
        **kwargs: Any,
    ) -> fx.GraphModule:
        """Tag the joint graph and materialize recompute duplicates.

        Args:
            graph_module: The joint forward+backward FX graph.
            pass_config: Parallel configuration driving ``ac_mode``.
            **kwargs: ``model`` (the live module) is required whenever
                ``ac_mode != "off"``: block discovery resolves the recompute
                scope from it. ``ac_scope`` is the pre-resolved scope from
                ``GraphCompiler`` (it also ran the adapter hook before
                tracing); direct callers may omit it.

        Returns:
            The transformed graph module.
        """
        if pass_config.ac_mode == "off":
            return graph_module
        scope = kwargs.get("ac_scope")
        if scope is None:
            scope = prepare_activation_checkpoint(kwargs.get("model"), pass_config)
        already_rematted = any(
            node.meta.get("ac_recomputed") for node in graph_module.graph.nodes
        )
        _tag_sac_policy(graph_module, pass_config, scope)
        duplicated = _remat(graph_module)
        _LOG.info(
            "Activation checkpointing (mode=%s) recomputed %d node(s)",
            pass_config.ac_mode,
            duplicated,
        )
        if duplicated == 0 and scope.region_fqns and not already_rematted:
            _LOG.warning(
                "activation checkpointing matched scope %s but recomputed no "
                "node; the model may already be wrapped by eager activation "
                "checkpointing, so graph-mode recompute is inactive",
                sorted(scope.region_fqns),
            )
        return graph_module
