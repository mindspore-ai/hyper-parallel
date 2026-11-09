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
"""Static-graph CPU offload - host-side activation swap.

Graph-mode counterpart of the dynamic ``activation_swap="attention"`` setting
(``hyper_parallel.distributed.attention_swap``). It reuses the same public
configuration surface - the Trainer's top-level ``activation_swap`` field -
and is driven by ``PassConfig.activation_swap``. The implementation follows
torchtitan graph_trainer's ``cpu_offload.py``:

1. Select forward activations produced inside ``self_attn`` modules that the
   backward half consumes (directly or through a chain of aliases), are large
   enough to amortize a transfer, are contiguous and independently owned, are
   never written in place, and fit the host-memory budget (largest-first).
   Aliases are grouped by real storage identity, so replay ops without schema
   ``alias_info`` (for example ``aten::_unsafe_view``) share the owner's single
   offload and lifetime. In-place writes are detected from the schema (including
   ops like ``aten::_foreach_add_`` that return no tensor), so a storage that
   changes after an earlier offload is never moved to host.
2. Insert the asynchronous transfer after each producer:

   ``cpu = ao.offload(gpu)`` is launched right after the producer, and
   ``cpu = ao.wait_tensor(cpu, keepalive=gpu, last_use_of_storage=...)`` is
   placed after the last forward read of the whole storage group. The D2H copy
   therefore overlaps the remaining forward compute, and the GPU storage is
   released as soon as the wait runs.
3. Insert ``gpu = ao.reload(cpu, device)`` plus a ``wait_tensor`` before the
   first backward consumer, redirecting every backward consumer (including
   consumers reached through replayed aliases) to the reloaded tensor. Reloads
   may be prefetched a few attention regions
   earlier to overlap H2D with preceding backward compute.

The pass runs before ``FSDPPass``: it only rewrites saved forward values, so
FSDP sees the resulting backward consumers when it inserts parameter
re-gathers. Attention swap is incompatible with activation checkpointing (the
dynamic rule is enforced in ``PassConfig``), so no recompute interacts with it.

Only ``activation_swap="attention"`` is enabled because that is the only mode
the dynamic attention-swap path exposes. Models without ``self_attn`` modules
raise the same error as the eager path. Symbolic (dynamic-shape) activations are
supported: selection heuristics (the 1 MiB threshold and the host budget) read
the trace-time size hints without specializing the graph, and the transfer ops
carry only the device so runtime shapes/strides come from the offloaded tensor.
Unbacked (data-dependent) sizes have no hint and are skipped.
"""

from __future__ import annotations

__all__ = ["CpuOffloadPass"]

import contextlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import fx

from .base import GraphPass
from .overlap.utils import is_view
from ..pass_config import PassConfig

try:  # Register the upstream ``ao::`` custom ops on import.
    import torch._functorch._activation_offloading.offload_ops  # noqa: F401

    _AO_OFFLOAD = torch.ops.ao.offload.default
    _AO_RELOAD = torch.ops.ao.reload.default
    _AO_WAIT = torch.ops.ao.wait_tensor.default
    _AO_OPS = frozenset((_AO_OFFLOAD, _AO_RELOAD, _AO_WAIT))
    _AO_AVAILABLE = True
except (ImportError, AttributeError):
    _AO_OFFLOAD = _AO_RELOAD = _AO_WAIT = None
    _AO_OPS = frozenset()
    _AO_AVAILABLE = False

_LOG = logging.getLogger(__name__)

# Match the dynamic attention-swap policy: only independently-owned tensors of
# at least 1 MiB amortize the transfer; smaller views/activations stay on device.
_MIN_OFFLOAD_BYTES = 1024 * 1024

_COLLECTIVE_NAMESPACES = frozenset(("_c10d_functional", "_c10d_functional_autograd"))


def _is_backward(node: fx.Node) -> bool:
    """Whether ``node`` belongs to the joint graph's backward half."""
    return bool(node.meta.get("autograd_backward", False))


def _stack_fqns(node: fx.Node) -> List[str]:
    """Module FQNs on ``node``'s ``nn_module_stack``, innermost first."""
    stack = node.meta.get("nn_module_stack") or {}
    return [fqn for fqn, _ in reversed(list(stack.values()))]


def _attention_roots(
    graph_module: fx.GraphModule, model: Optional[torch.nn.Module]
) -> frozenset:
    """Discover the ``self_attn`` module FQNs that define the offload scope.

    The eager attention-swap discovery only accepts modules literally named
    ``self_attn``; graph mode mirrors that contract and additionally accepts
    roots recorded in the traced module stack when no live model is supplied.
    """
    roots = set()
    if model is not None:
        roots.update(
            name
            for name, _ in model.named_modules()
            if name.rsplit(".", 1)[-1] == "self_attn"
        )
    for node in graph_module.graph.nodes:
        for fqn in _stack_fqns(node):
            if fqn.rsplit(".", 1)[-1] == "self_attn":
                roots.add(fqn)
    return frozenset(roots)


def _in_attention_scope(node: fx.Node, roots: frozenset) -> bool:
    """Whether a node was produced inside one of the attention roots."""
    if not roots:
        return False
    for fqn in _stack_fqns(node):
        if fqn.rsplit(".", 1)[-1] == "self_attn":
            return True
        if any(fqn == root or fqn.startswith(f"{root}.") for root in roots):
            return True
    return False


def _attention_root_of(node: fx.Node, roots: frozenset) -> Optional[str]:
    """Return the innermost attention root containing ``node`` (or ``None``)."""
    for fqn in _stack_fqns(node):
        for root in roots:
            if fqn == root or fqn.startswith(f"{root}."):
                return root
    return None


def _is_collective_or_wait(node: fx.Node) -> bool:
    """Whether the node is a functional collective launch or its wait.

    Both the launch namespace and the autograd-compat namespace are covered so
    a collective result is never moved to host.
    """
    namespace = getattr(node.target, "namespace", None)
    return namespace in _COLLECTIVE_NAMESPACES


def _is_dtensor(value: torch.Tensor) -> bool:
    """Recognize DTensor-like values that the ``ao::`` ops cannot transfer."""
    return hasattr(value, "device_mesh") or hasattr(value, "to_local")


def _hint_to_int(value: Any) -> Optional[int]:
    """Convert an ``int``/``SymInt`` to a plain int without installing a guard.

    Sizes only feed selection heuristics (the 1 MiB threshold and the host
    budget), so a symbolic value's trace-time hint is enough. ``int(SymInt)``
    would specialize the graph and add a shape guard; unbacked (data-dependent)
    symbols have no hint and yield ``None`` so the candidate is skipped.
    """
    try:
        if isinstance(value, torch.SymInt):
            value = value.node.hint
        return int(value)
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def _tensor_bytes(value: torch.Tensor) -> Optional[int]:
    """Return a hinted byte count, or ``None`` for unsupported values."""
    try:
        nbytes = value.nelement() * value.element_size()
    except Exception:  # pylint: disable=broad-exception-caught
        return None
    return _hint_to_int(nbytes)


def _is_contiguous(value: torch.Tensor) -> bool:
    """Check contiguity without tripping over symbolic fake tensors."""
    try:
        return bool(value.is_contiguous())
    except Exception:  # pylint: disable=broad-exception-caught
        return False


def _storage_key(value: torch.Tensor) -> Optional[int]:
    """Return a stable identity for the tensor's storage, or ``None`` if unknown.

    Aliasing is a storage property, not an operator property: replay-based view
    ops carry no schema ``alias_info`` (``aten::_unsafe_view``), and some
    schema-tagged views can still be full-size. FakeTensor preserves the
    underlying storage, so grouping by the real storage (``_cdata``) detects
    aliases that ``is_view`` misses without enumerating every op.

    ``None`` means the storage identity cannot be proven. The Python object
    ``id()`` is deliberately not used as a fallback: it is not stable across
    ``untyped_storage()`` wrappers, and a mismatched identity could let an
    alias or an in-place write slip through. Returning ``None`` instead makes
    callers skip the candidate.
    """
    try:
        storage = value.untyped_storage()
    except Exception:  # pylint: disable=broad-exception-caught
        return None
    cdata = getattr(storage, "_cdata", None)
    return int(cdata) if cdata else None


def _shares_storage(left: fx.Node, right: fx.Node) -> bool:
    """Whether two nodes' values refer to the same storage (unknown => False)."""
    left_value = left.meta.get("val")
    right_value = right.meta.get("val")
    if not isinstance(left_value, torch.Tensor) or not isinstance(
        right_value, torch.Tensor
    ):
        return False
    left_key = _storage_key(left_value)
    return left_key is not None and left_key == _storage_key(right_value)


def _is_alias(user: fx.Node, producer: fx.Node) -> bool:
    """Whether ``user`` aliases ``producer``'s storage (schema view or shared)."""
    return is_view(user) or _shares_storage(user, producer)


def _is_mutable(node: fx.Node) -> bool:
    """Whether the node targets an in-place/mutating op."""
    schema = getattr(node.target, "_schema", None)
    return schema is not None and schema.is_mutable


def _iter_nodes(value: Any) -> List[fx.Node]:
    """Collect the ``fx.Node`` leaves inside an argument (``Tensor[]`` etc.)."""
    nodes: List[fx.Node] = []
    fx.map_arg(value, nodes.append)
    return nodes


def _storage_keys(nodes: List[fx.Node]) -> List[int]:
    """Resolvable storage keys of the tensor-valued ``nodes``."""
    keys: List[int] = []
    for node in nodes:
        value = node.meta.get("val")
        if isinstance(value, torch.Tensor):
            key = _storage_key(value)
            if key is not None:
                keys.append(key)
    return keys


def _tensor_argument_keys(node: fx.Node) -> List[int]:
    """Storage keys of every tensor-valued argument to ``node``.

    Conservative fallback for nodes whose write target cannot be pinned down:
    skipping an offload only costs memory, while missing a write corrupts the
    backward reload.
    """
    return _storage_keys(_iter_nodes((node.args, node.kwargs)))


def _written_storage_keys(node: fx.Node) -> List[int]:
    """Storage keys that a mutating op writes through its arguments.

    Schema ``alias_info.is_write`` marks the arguments an op modifies in place.
    A node's output value is not enough to detect this: ``aten._foreach_add_``
    returns ``()`` but still writes its ``Tensor(a!)[]`` input, so grouping only
    returned tensors misses the write and could offload a value that the forward
    changes afterwards (making the backward reload stale). Written arguments are
    resolved by name first (keyword form) and then by positional index;
    ``fx.map_arg`` walks ``Tensor[]`` and nested containers to the nodes.

    Two conservative fallbacks keep this sound when the exact write target is
    unknowable:

    * a node without a schema (``call_method`` / ``call_module`` / plain
      callable) may mutate any argument, and
    * a mutable schema whose write argument cannot be resolved (vararg layout,
      an omitted positional default, or a missing ``meta['val']``).

    Both return every tensor argument's storage instead of an empty list, so an
    unprovable write never lets a candidate be offloaded.
    """
    schema = getattr(node.target, "_schema", None)
    if schema is None:
        return _tensor_argument_keys(node)
    if not schema.is_mutable:
        return []

    keys: List[int] = []
    resolved = False
    unresolved = False
    position = 0
    for argument in schema.arguments:
        if argument.kwarg_only:
            index = None
        else:
            index = position
            position += 1
        alias_info = argument.alias_info
        if alias_info is None or not alias_info.is_write:
            continue
        if argument.name in node.kwargs:
            value = node.kwargs[argument.name]
        elif index is not None and index < len(node.args):
            value = node.args[index]
        else:
            unresolved = True
            continue
        written = _iter_nodes(value)
        if not written:
            continue
        resolved = True
        written_keys = _storage_keys(written)
        if not written_keys:
            unresolved = True
        keys.extend(written_keys)
    if unresolved or not resolved:
        return _tensor_argument_keys(node)
    return keys


def _group_storage_members(graph: fx.Graph) -> Dict[int, List[fx.Node]]:
    """Group tensor-valued nodes by the real identity of their storage."""
    members: Dict[int, List[fx.Node]] = {}
    for node in graph.nodes:
        if node.op not in ("call_function", "placeholder", "get_attr"):
            continue
        value = node.meta.get("val")
        if not isinstance(value, torch.Tensor):
            continue
        key = _storage_key(value)
        if key is not None:
            members.setdefault(key, []).append(node)
    return members


def _write_unsafe_keys(graph: fx.Graph) -> set:
    """Storage keys written in place anywhere in the graph.

    Includes mutations whose return is not a tensor (``_foreach_add_`` returns
    ``()``): such a write still changes a storage that may already have been
    offloaded. ``call_method`` / ``call_module`` nodes carry no schema; their
    tensor arguments are marked conservatively because an unseen in-place write
    would make the backward reload stale.
    """
    unsafe: set = set()
    for node in graph.nodes:
        if node.op in ("call_function", "call_method", "call_module"):
            unsafe.update(_written_storage_keys(node))
    return unsafe


def _group_is_safe(nodes: List[fx.Node]) -> bool:
    """Whether a storage group has no external, mutating, or output sharer."""
    if any(node.op != "call_function" for node in nodes):
        return False
    if any(_is_mutable(node) for node in nodes):
        return False
    if any(user.op == "output" for node in nodes for user in node.users):
        return False
    return True


@dataclass(frozen=True)
class _StorageIndex:
    """Real-storage groups used to prove a candidate can be offloaded safely.

    A storage may be offloaded only when it is owned by a single forward
    ``call_function`` node (the ``owner``), no other node in the graph shares it
    through an external (placeholder/get_attr) or mutating op, and it is not a
    graph output. Writes are detected from the schema, not only from returned
    tensors, so an in-place op with no tensor return still poisons its input
    storage. Groups that cannot be proven safe are marked ``unsafe``.
    """

    owner: Dict[int, fx.Node]
    unsafe: frozenset

    @classmethod
    def build(cls, graph: fx.Graph, order: Dict[fx.Node, int]) -> "_StorageIndex":
        """Group tensor-valued nodes by storage and validate each group."""
        members = _group_storage_members(graph)
        unsafe = _write_unsafe_keys(graph)
        owner: Dict[int, fx.Node] = {}
        for key, nodes in members.items():
            if key in unsafe:
                continue
            if not _group_is_safe(nodes):
                unsafe.add(key)
                continue
            owner[key] = min(nodes, key=lambda node: order[node])
        return cls(owner=owner, unsafe=frozenset(unsafe))


def _can_offload(node: fx.Node, storage: _StorageIndex) -> bool:
    """Whether a forward node's output is a profitable, safely-owned candidate.

    Views are skipped because they alias their base, so offloading them frees
    nothing and, for ``getitem`` of a multi-output op, could release storage a
    sibling still uses. Shared storage is detected by real identity: only the
    group owner is eligible, and ``unsafe`` groups (external or mutating
    aliases) are left on device.
    """
    if node.op != "call_function":
        return False
    if node.target in _AO_OPS:
        return False
    if is_view(node):
        return False
    if _is_collective_or_wait(node):
        return False
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor) or _is_dtensor(value):
        return False
    if value.dim() < 2:
        return False
    if not _is_contiguous(value):
        return False
    nbytes = _tensor_bytes(value)
    if nbytes is None or nbytes < _MIN_OFFLOAD_BYTES:
        return False
    # Offload one storage at a time. Only the group owner (the node that
    # allocated the storage) is eligible; aliases are handled by the owner's
    # view chain, and external/mutating sharers mark the group unsafe.
    key = _storage_key(value)
    if key is None or key in storage.unsafe:
        return False
    if storage.owner.get(key) is not node:
        return False
    return True


def _view_chain(
    node: fx.Node,
) -> Tuple[List[fx.Node], List[fx.Node], List[Tuple[fx.Node, fx.Node]]]:
    """Collect forward aliases of ``node`` and their backward consumers.

    Aliases are found by real storage identity as well as schema views, so
    replay ops without ``alias_info`` (``aten::_unsafe_view``) join the chain.

    Returns:
        ``(lifetime_views, replay_views, view_bwd_redirects)``. ``lifetime_views``
        are every forward alias view (needed to find the producer's true last
        forward use, so the storage is not released while a forward-only view is
        still live). ``replay_views`` is the subset that must be rebuilt on top
        of the reloaded tensor because it (or a descendant view) is read in
        backward. Each redirect pair ``(view_node, backward_user)`` must be
        repointed after replay.
    """
    children: Dict[fx.Node, List[fx.Node]] = {}
    backward_users: Dict[fx.Node, List[fx.Node]] = {}
    lifetime_views: List[fx.Node] = []
    visited: set = set()

    def collect(current: fx.Node) -> None:
        """Walk alias users and record backward readers."""
        for user in current.users:
            if user.op != "call_function":
                continue
            if _is_backward(user):
                backward_users.setdefault(current, []).append(user)
            elif _is_alias(user, current) and user not in visited:
                visited.add(user)
                children.setdefault(current, []).append(user)
                lifetime_views.append(user)
                collect(user)

    collect(node)

    needed: set = set()
    memo: Dict[fx.Node, bool] = {}

    def is_read_in_backward(current: fx.Node) -> bool:
        """Whether a view subtree has any backward reader (memoized)."""
        cached = memo.get(current)
        if cached is not None:
            return cached
        needed_here = bool(backward_users.get(current))
        for child in children.get(current, ()):
            if is_read_in_backward(child):
                needed_here = True
        memo[current] = needed_here
        if needed_here and current is not node:
            needed.add(current)
        return needed_here

    is_read_in_backward(node)
    replay_views = [view for view in lifetime_views if view in needed]
    redirects = [
        (view, user) for view in replay_views for user in backward_users.get(view, ())
    ]
    return lifetime_views, replay_views, redirects


@dataclass
class _OffloadInfo:
    """Resolved backward consumers and forward lifetime of one candidate."""

    node: fx.Node
    direct_bwd_users: List[fx.Node]
    lifetime_views: List[fx.Node]
    replay_views: List[fx.Node]
    view_bwd_redirects: List[Tuple[fx.Node, fx.Node]]
    first_bwd_consumer: fx.Node
    last_forward_use: fx.Node
    last_use_arg: fx.Node

    @property
    def all_bwd_users(self) -> List[fx.Node]:
        """Every backward consumer, direct or reached through a view."""
        return self.direct_bwd_users + [user for _, user in self.view_bwd_redirects]


def _collect_offload_info(
    node: fx.Node, order: Dict[fx.Node, int]
) -> Optional[_OffloadInfo]:
    """Resolve one candidate's backward users and its last forward use."""
    direct_bwd_users = [user for user in node.users if _is_backward(user)]
    lifetime_views, replay_views, view_bwd_redirects = _view_chain(node)
    all_bwd_users = direct_bwd_users + [user for _, user in view_bwd_redirects]
    if not all_bwd_users:
        return None

    last_forward_use = node
    last_index = order[node]
    for chain_node in (node, *lifetime_views):
        for user in chain_node.users:
            if user.op != "call_function" or _is_backward(user):
                continue
            user_index = order.get(user)
            if user_index is not None and user_index > last_index:
                last_forward_use = user
                last_index = user_index

    if last_forward_use is node:
        last_use_arg: fx.Node = node
    elif isinstance(last_forward_use.meta.get("val"), torch.Tensor):
        last_use_arg = last_forward_use
    else:
        # The forward ``ao.wait_tensor`` frees the producer's storage, so it must
        # carry a data-dependency edge on the last forward consumer
        # (``last_use_of_storage``). A tuple/list-returning consumer (split,
        # chunk, max, ...) has no single tensor value to pin that edge; skip the
        # candidate rather than risk a reordering pass hoisting the storage free
        # before its last forward read.
        return None

    return _OffloadInfo(
        node=node,
        direct_bwd_users=direct_bwd_users,
        lifetime_views=lifetime_views,
        replay_views=replay_views,
        view_bwd_redirects=view_bwd_redirects,
        first_bwd_consumer=min(all_bwd_users, key=lambda item: order[item]),
        last_forward_use=last_forward_use,
        last_use_arg=last_use_arg,
    )


def _collect_candidates(
    graph_module: fx.GraphModule,
    roots: frozenset,
    order: Dict[fx.Node, int],
) -> List[Tuple[fx.Node, int, _OffloadInfo]]:
    """Collect in-scope forward activations that can be offloaded."""
    storage = _StorageIndex.build(graph_module.graph, order)
    candidates: List[Tuple[fx.Node, int, _OffloadInfo]] = []
    for node in list(graph_module.graph.nodes):
        if _is_backward(node):
            continue
        if not _in_attention_scope(node, roots) or not _can_offload(node, storage):
            continue
        info = _collect_offload_info(node, order)
        if info is None:
            continue
        nbytes = _tensor_bytes(node.meta["val"])
        if nbytes is None:
            continue
        candidates.append((node, nbytes, info))
    return candidates


def _select_within_budget(
    candidates: List[Tuple[fx.Node, int, _OffloadInfo]],
    budget_bytes: float,
) -> Tuple[List[_OffloadInfo], int]:
    """Pick largest-first until the host budget is exhausted (0 means all)."""
    candidates.sort(key=lambda item: item[1], reverse=True)
    selected: List[_OffloadInfo] = []
    total_bytes = 0
    for _, nbytes, info in candidates:
        if 0 < budget_bytes < total_bytes + nbytes:
            continue
        selected.append(info)
        total_bytes += nbytes
    return selected, total_bytes


def _copy_meta(source: fx.Node) -> Dict[str, Any]:
    """Copy node metadata suitable for an inserted offload-chain node."""
    return {
        key: value
        for key, value in source.meta.items()
        if key not in ("val", "recompute")
    }


def _insert_forward_offload(
    graph_module: fx.GraphModule,
    info: _OffloadInfo,
) -> fx.Node:
    """Insert ``ao.offload`` and the deferred forward ``ao.wait_tensor``.

    Returns:
        The wait node whose (CPU) result is the reload source.
    """
    graph = graph_module.graph
    node = info.node
    value = node.meta["val"]
    src_meta = _copy_meta(node)

    with graph.inserting_after(node):
        offload = graph.call_function(_AO_OFFLOAD, args=(node,))
    offload.meta.update(src_meta)
    offload.meta["val"] = value.to(torch.device("cpu"))

    if info.last_forward_use is node:
        anchor = offload
    else:
        anchor = info.last_forward_use

    with graph.inserting_after(anchor):
        wait = graph.call_function(
            _AO_WAIT,
            args=(offload, node, info.last_use_arg),
        )
    wait.meta.update(src_meta)
    wait.meta["val"] = offload.meta["val"]
    wait.meta["autograd_backward"] = False
    return wait


def _insert_backward_reload(
    graph_module: fx.GraphModule,
    info: _OffloadInfo,
    wait_offload: fx.Node,
    order: Dict[fx.Node, int],
) -> fx.Node:
    """Insert ``ao.reload`` + ``ao.wait_tensor`` and redirect backward readers."""
    graph = graph_module.graph
    node = info.node
    value = node.meta["val"]
    src_meta = _copy_meta(node)
    first_bwd = info.first_bwd_consumer

    with graph.inserting_before(first_bwd):
        # Candidates are contiguous, so the offloaded host tensor already carries
        # the runtime shape and contiguous stride. Omitting the explicit
        # size/stride keeps symbolic values out of the graph arguments: baking a
        # ``SymInt`` here would render an undefined symbol in the generated
        # forward (dynamic shapes).
        reload = graph.call_function(
            _AO_RELOAD,
            args=(wait_offload, value.device, None, None),
        )
    reload.meta.update(src_meta)
    reload.meta["val"] = value
    reload.meta["autograd_backward"] = True

    with graph.inserting_before(first_bwd):
        wait = graph.call_function(_AO_WAIT, args=(reload, wait_offload))
    wait.meta.update(src_meta)
    wait.meta["val"] = value
    wait.meta["autograd_backward"] = True

    for user in info.direct_bwd_users:
        user.replace_input_with(node, wait)

    if info.replay_views:
        replay_map: Dict[fx.Node, fx.Node] = {node: wait}
        for view_node in sorted(info.replay_views, key=lambda item: order[item]):
            new_args = fx.map_arg(
                view_node.args,
                lambda arg: (
                    replay_map.get(arg, arg) if isinstance(arg, fx.Node) else arg
                ),
            )
            new_kwargs = fx.map_arg(
                view_node.kwargs,
                lambda arg: (
                    replay_map.get(arg, arg) if isinstance(arg, fx.Node) else arg
                ),
            )
            with graph.inserting_before(first_bwd):
                replayed = graph.call_function(
                    view_node.target, args=new_args, kwargs=new_kwargs
                )
            replayed.meta.update(
                {k: v for k, v in view_node.meta.items() if k != "recompute"}
            )
            replayed.meta["autograd_backward"] = True
            replay_map[view_node] = replayed
        for consumed, user in info.view_bwd_redirects:
            replayed = replay_map.get(consumed)
            if replayed is not None:
                user.replace_input_with(consumed, replayed)
    return reload


def _region_anchors(graph_module: fx.GraphModule, roots: frozenset) -> List[fx.Node]:
    """Return the first backward node of each contiguous attention region."""
    anchors: List[fx.Node] = []
    previous_root: Optional[str] = None
    for node in graph_module.graph.nodes:
        if (
            node.op != "call_function"
            or not _is_backward(node)
            or node.target in _AO_OPS
        ):
            continue
        root = _attention_root_of(node, roots)
        if root is not None and root != previous_root:
            anchors.append(node)
        previous_root = root
    return anchors


def _anchor_positions(anchors: List[fx.Node], roots: frozenset) -> Dict[str, List[int]]:
    """Map each attention root to its ordered region-anchor indices."""
    positions: Dict[str, List[int]] = {}
    for index, anchor in enumerate(anchors):
        root = _attention_root_of(anchor, roots)
        if root is not None:
            positions.setdefault(root, []).append(index)
    return positions


def _reload_nodes(graph: fx.Graph) -> List[fx.Node]:
    """Return every backward ``ao.reload`` node in graph order."""
    return [
        node
        for node in graph.nodes
        if node.op == "call_function"
        and node.target is _AO_RELOAD
        and _is_backward(node)
    ]


def _move_reload(
    reload_node: fx.Node,
    roots: frozenset,
    anchors: List[fx.Node],
    anchor_positions: Dict[str, List[int]],
    order: Dict[fx.Node, int],
    n_regions: int,
) -> bool:
    """Move one reload earlier by ``n_regions`` regions when possible."""
    root = _attention_root_of(reload_node, roots)
    positions = anchor_positions.get(root) if root is not None else None
    if not positions:
        return False
    reload_index = order[reload_node]
    # Latest occurrence of this root at or before the reload is its own
    # region when a module is reused across blocks; otherwise fall back to
    # the first occurrence (reload inserted before the region's first node).
    earlier = [pos for pos in positions if order[anchors[pos]] <= reload_index]
    region_index = earlier[-1] if earlier else positions[0]
    anchor = anchors[max(0, region_index - n_regions)]
    if order[anchor] >= reload_index:
        return False
    anchor.prepend(reload_node)
    return True


def _prefetch_reloads(
    graph_module: fx.GraphModule,
    n_regions: int,
    roots: frozenset,
) -> int:
    """Move backward reloads ``n_regions`` attention regions earlier.

    Regions are located by graph position rather than by root name so a
    ``self_attn`` module that appears in several blocks (same FQN, or tied
    weights) still maps each reload to the region it actually belongs to.
    """
    if n_regions <= 0 or not roots:
        return 0

    order = {node: index for index, node in enumerate(graph_module.graph.nodes)}

    # First backward node of each contiguous attention region. Tracking the
    # previous node's root (including ``None`` gaps) re-opens a region when the
    # same root resumes after out-of-scope backward work.
    anchors = _region_anchors(graph_module, roots)
    if not anchors:
        return 0

    anchor_positions = _anchor_positions(anchors, roots)
    moved = 0
    for reload_node in _reload_nodes(graph_module.graph):
        if _move_reload(
            reload_node, roots, anchors, anchor_positions, order, n_regions
        ):
            moved += 1
    return moved


def _graph_shape_env(graph_module: fx.GraphModule):
    """Return the ``ShapeEnv`` backing the graph's fake tensors, if any."""
    for node in graph_module.graph.nodes:
        fake_mode = getattr(node.meta.get("val"), "fake_mode", None)
        shape_env = getattr(fake_mode, "shape_env", None)
        if shape_env is not None:
            return shape_env
    return None


class CpuOffloadPass(GraphPass):
    """Offload saved attention activations to host memory.

    Configure through ``PassConfig.activation_swap`` (reusing the dynamic
    Trainer field) and the graph-only ``offload_budget_gb`` /
    ``offload_prefetch_n_layers`` knobs. The pass is a no-op unless
    ``activation_swap == "attention"``.
    """

    name = "cpu_offload"

    def run(
        self,
        graph_module: fx.GraphModule,
        pass_config: PassConfig,
        **kwargs: Any,
    ) -> fx.GraphModule:
        """Insert async host offload/reload around saved attention activations.

        Args:
            graph_module: The joint forward+backward FX graph.
            pass_config: Parallel configuration. ``activation_swap`` enables
                the pass and ``offload_budget_gb`` /
                ``offload_prefetch_n_layers`` tune it.
            **kwargs: ``model`` is used to discover the ``self_attn`` modules
                (the graph's module stack is the fallback).

        Returns:
            The transformed graph module.

        Raises:
            ValueError: When ``activation_swap="attention"`` but the model has
                no ``self_attn`` module, matching the eager attention-swap
                contract.
        """
        if pass_config.activation_swap == "none":
            return graph_module
        if not _AO_AVAILABLE:
            raise RuntimeError(
                "activation_swap='attention' requires torch's "
                "torch._functorch._activation_offloading.offload_ops (ao:: "
                "offload/reload/wait ops), which is unavailable in this torch "
                "build; graph-mode CPU offload cannot run"
            )

        shape_env = _graph_shape_env(graph_module)
        suppress_guards = getattr(shape_env, "suppress_guards", None)
        context = (
            suppress_guards()
            if suppress_guards is not None
            else contextlib.nullcontext()
        )
        # Selection and rewrite only read trace-time size hints. Suppress guard
        # recording so inspecting symbolic sizes/strides (contiguity, byte
        # counts, the CPU meta value) can never specialize the dynamic graph.
        with context:
            roots = _attention_roots(graph_module, kwargs.get("model"))
            if not roots:
                raise ValueError(
                    "activation_swap='attention' found no self_attn modules; "
                    "expected attention modules registered as self_attn"
                )

            graph = graph_module.graph
            order = {node: index for index, node in enumerate(graph.nodes)}

            candidates = _collect_candidates(graph_module, roots, order)
            budget_bytes = pass_config.offload_budget_gb * (1024**3)
            selected, total_bytes = _select_within_budget(candidates, budget_bytes)

            if not selected:
                _LOG.info("CPU offload: no attention activations selected")
                return graph_module

            wait_offloads: List[Tuple[_OffloadInfo, fx.Node]] = []
            for info in selected:
                wait_offload = _insert_forward_offload(graph_module, info)
                wait_offloads.append((info, wait_offload))

            for info, wait_offload in wait_offloads:
                _insert_backward_reload(graph_module, info, wait_offload, order)

            moved = _prefetch_reloads(
                graph_module,
                pass_config.offload_prefetch_n_layers,
                roots,
            )

        graph.lint()
        graph_module.recompile()
        _LOG.info(
            "CPU offload: offloaded %d attention activation(s) "
            "(%.2f MiB), prefetched %d reload(s)",
            len(selected),
            total_bytes / (1024 * 1024),
            moved,
        )
        return graph_module
