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
"""Deterministic, read-only planning for explicitly tagged FSDP collectives."""

import copy
import hashlib
import json
import logging
import math
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Mapping, Optional, Sequence, Union

import torch
from torch import fx, nn
from torch.ops import _c10d_functional as c10d

from ..overlap.utils import is_barrier, is_collective, is_op, is_view, rs_layout_preparation
from ..parallel.fsdp_pass import _free_tensor_storage

_LOG = logging.getLogger(__name__)

_KINDS = {"fsdp_all_gather": c10d.all_gather_into_tensor,
          "fsdp_all_gather_backward": c10d.all_gather_into_tensor,
          "fsdp_reduce_scatter": c10d.reduce_scatter_tensor}
_FREES = {"fsdp_reshard_free", "fsdp_reshard_free_backward"}


ModuleBucketPlans = Sequence[Union[str, Sequence[str]]]


def normalize_module_bucket_plans(plans: ModuleBucketPlans) -> tuple[tuple[str, ...], ...]:
    """Freeze explicit module groups and reject ambiguous or wildcard ownership.

    Args:
        plans: Each entry is a module FQN or a nonempty sequence of module FQNs.
            Prefixes include descendants on a dot boundary. Root scope is unsupported.

    Returns:
        Disjoint groups in caller order. An empty plan selects no modules.

    Raises:
        ValueError: A path is invalid, repeated, or overlaps another path.
    """
    if not isinstance(plans, (list, tuple)):
        raise ValueError("module_bucket_plans must be a list or tuple of module groups")
    groups, prefixes = [], []
    for entry in plans:
        group = (entry,) if isinstance(entry, str) else entry
        if not isinstance(group, (list, tuple)) or not group:
            raise ValueError("Each module bucket group must be a FQN or a nonempty list/tuple")
        for fqn in group:
            if (not isinstance(fqn, str) or not fqn or any(char in fqn for char in "*?[]")
                    or any(not part or part.strip() != part for part in fqn.split("."))):
                raise ValueError(f"Invalid explicit module FQN: {fqn!r}")
            if any(fqn == old or fqn.startswith(old + ".") or old.startswith(fqn + ".") for old in prefixes):
                raise ValueError(f"Overlapping module bucket scopes: {fqn!r}")
            prefixes.append(fqn)
        groups.append(tuple(group))
    return tuple(groups)


def match_module_bucket(param_name: str, plans: tuple[tuple[str, ...], ...]) -> Optional[tuple[str, ...]]:
    """Return the logical module group owning a parameter, or None if unselected.

    Args:
        param_name: Fully qualified parameter name.
        plans: Normalized explicit module groups.
    """
    for group in plans:
        if any(param_name.startswith(fqn + ".") for fqn in group):
            return group
    return None


def build_module_bucket_plans(
    model: nn.Module, *, block_module_types: tuple[type[nn.Module], ...] = (),
    module_bucket_plans: ModuleBucketPlans = (),
) -> tuple[tuple[str, ...], ...]:
    """Build groups from model-owned block types and explicit additional modules.

    Args:
        model: Model whose named_modules paths match FSDPPass parameter FQNs.
        block_module_types: Concrete decoder/block classes supplied by the model
            adapter. Each outermost matching non-root module becomes one group.
        module_bucket_plans: Additional groups, e.g. embedding and norm/head.
            No naming heuristic or model-wide wildcard is used.

    Returns:
        Validated immutable groups. Their order describes ownership, not the
        collective execution order, which is derived from the graph.

    Raises:
        ValueError: Types, module paths or group ownership are invalid.
    """
    if not isinstance(model, nn.Module):
        raise ValueError("model must be a torch.nn.Module")
    if (not isinstance(block_module_types, tuple)
            or any(not isinstance(kind, type) or not issubclass(kind, nn.Module) for kind in block_module_types)):
        raise ValueError("block_module_types must be a tuple of torch.nn.Module subclasses")
    explicit = normalize_module_bucket_plans(module_bucket_plans)
    modules = dict(model.named_modules())
    for group in explicit:
        for fqn in group:
            if fqn not in modules:
                raise ValueError(f"Module bucket scope does not exist in model: {fqn!r}")
    blocks = []
    for fqn, module in modules.items():
        if (fqn and isinstance(module, block_module_types)
                and not any(fqn.startswith(parent + ".") for parent in blocks)):
            blocks.append(fqn)
    return normalize_module_bucket_plans([*blocks, *explicit])


def _static_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class BucketPolicy:
    """Optional physical bucket limits inside a logical module group.

    Args:
        cap_bytes: Maximum packed input/output bytes; None leaves capacity uncapped.
        max_input_distance: Optional FX-node distance between input producers.
        max_consumer_distance: Optional FX-node distance between first consumers.
            All defaults are None, matching module-driven rather than size-driven
            grouping. Safety barriers and communication compatibility always apply.
    """

    cap_bytes: Optional[int] = None
    max_input_distance: Optional[int] = None
    max_consumer_distance: Optional[int] = None

    def __post_init__(self):
        for name in ("cap_bytes", "max_input_distance", "max_consumer_distance"):
            value = getattr(self, name)
            if value is not None and (not _static_int(value) or value < (1 if name == "cap_bytes" else 0)):
                raise ValueError(f"{name} must be None or a non-negative integer (cap_bytes must be positive)")


@dataclass(frozen=True)
class Candidate:
    """One communication instance, with static layout and scheduling bounds."""

    launch: fx.Node = field(repr=False)
    wait: fx.Node = field(repr=False)
    source: fx.Node = field(repr=False)
    param_name: str
    state_idx: int
    kind: str
    backward: bool
    scope: str
    module_fqns: tuple[str, ...]
    group: str
    ranks: tuple[int, ...]
    dtype: torch.dtype
    device: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    ready: int
    first_use: int
    reduce_op: Optional[str]

    @property
    def size_bytes(self) -> int:
        """Return the larger of the input and output buffer sizes."""
        return max(math.prod(self.input_shape), math.prod(self.output_shape)) * self.dtype.itemsize

    @property
    def key(self) -> tuple:
        """Return the complete compatibility key for this local graph."""
        return (self.kind, self.backward, self.scope, self.group, self.ranks,
                self.dtype, self.device, self.reduce_op)

    def describe(self) -> dict:
        """Return rank-independent wire layout plus useful local diagnostics."""
        return {"param_name": self.param_name, "state_idx": self.state_idx,
                "kind": self.kind, "backward": self.backward, "scope": self.scope, "module_fqns": self.module_fqns,
                "group": self.group, "ranks": self.ranks, "dtype": str(self.dtype),
                "device_type": torch.device(self.device).type,
                "input_shape": self.input_shape, "output_shape": self.output_shape,
                "reduce_op": self.reduce_op}


@dataclass(frozen=True)
class Bucket:
    """Ordered members of a proposed bucket; singleton means no fusion."""

    members: tuple[Candidate, ...]
    bucket_id: str = ""

    @property
    def supports_rs_rewrite(self) -> bool:
        """Whether this bucket has an executable multi-member RS rewrite."""
        return len(self.members) > 1 and self.members[0].kind == "fsdp_reduce_scatter"

    @property
    def supports_ag_rewrite(self) -> bool:
        """Whether independent copy-out can materialize this multi-member AG bucket."""
        return len(self.members) > 1 and self.members[0].kind in ("fsdp_all_gather", "fsdp_all_gather_backward")

    def describe(self) -> dict:
        """Describe packed offsets, bytes and scheduling bounds without mutation."""
        offset = 0
        members = []
        for member in self.members:
            local_numel = math.prod(member.output_shape if member.kind == "fsdp_reduce_scatter"
                                    else member.input_shape)
            members.append({**member.describe(), "local_offset": offset, "local_numel": local_numel})
            offset += local_numel
        return {"bucket_id": self.bucket_id, "rewrite_supported": self.supports_rs_rewrite or self.supports_ag_rewrite,
                "members": members, "size_bytes": sum(m.size_bytes for m in self.members),
                "latest_input": max(m.ready for m in self.members),
                "earliest_consumer": min(m.first_use for m in self.members),
                "fusable": len(self.members) > 1}


@dataclass(frozen=True)
class BucketPlan:
    """Dry-run result tied to one graph snapshot; never changes that graph."""

    buckets: tuple[Bucket, ...]
    decisions: tuple[dict, ...]
    signature: str
    group_ranks: dict[str, tuple[int, ...]]
    wire_collectives: dict[str, list]
    module_bucket_plans: Optional[tuple[tuple[str, ...], ...]]
    summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Export planning facts on demand; never reads erased nodes' arguments.

        Returns an independent JSON-serializable dict, not an application receipt.
        Original readiness bounds and layout remain available after rewriting.
        """
        return copy.deepcopy({"buckets": [bucket.describe() for bucket in self.buckets],
                              "decisions": self.decisions, "summary": self.summary,
                              "signature": self.signature, "module_bucket_plans": self.module_bucket_plans,
                              "group_ranks": self.group_ranks,
                              "group_digests": {group: self.group_digest(group) for group in self.group_ranks}})

    def group_digest(self, group: str) -> str:
        """Hash layout/order for one PG, excluding rank-local device indices.

        Args:
            group: Process group name in this plan.
        """
        layouts = [[member.describe() for member in bucket.members]
                   for bucket in self.buckets if bucket.members[0].group == group]
        return _digest((self.group_ranks[group], layouts, self.wire_collectives.get(group, [])))


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def graph_signature(graph: fx.Graph) -> str:
    """Fingerprint graph structure and planning facts to reject stale plans.

    Args:
        graph: Graph whose structure and FSDP metadata define the snapshot.
    """
    records = []
    for node in graph.nodes:
        metadata = {key: value for key, value in node.meta.items()
                    if key.startswith("fsdp_") or key in
                    ("comm_type", "comm_group", "param_name", "state_idx", "autograd_backward")}
        records.append((node.name, str(node.target), str(node.args), str(node.kwargs), metadata))
    return _digest(records)


def validate_group_plans(plans_by_rank: Mapping[int, Union[BucketPlan, dict]], group: str) -> None:
    """Validate reports exchanged by the caller, without creating collectives.

    Args:
        plans_by_rank: Plans or JSON-serializable to_dict() reports for every
            global rank in this process group. Exchange reports, not FX nodes.
        group: Exact process-group name used by the communication nodes.

    Raises:
        ValueError: Reports are missing ranks or disagree on membership/order/layout.
    """
    if not plans_by_rank:
        raise ValueError("No rank reports supplied")
    reports = {rank: plan.to_dict() if isinstance(plan, BucketPlan) else plan
               for rank, plan in plans_by_rank.items()}
    first = next(iter(reports.values()))
    expected = first["group_ranks"].get(group)
    if expected is None or set(plans_by_rank) != set(expected):
        raise ValueError(f"Expected reports for group {group} ranks {expected}")
    for report in reports.values():
        if (tuple(report["group_ranks"].get(group, ())) != tuple(expected)
                or report["group_digests"].get(group) != first["group_digests"].get(group)):
            raise ValueError(f"Bucket plans disagree for group {group}")


def _consumers(wait: fx.Node) -> set[fx.Node]:
    """Return real consumers of a wait, traversing only immutable view chains."""
    consumers, seen, stack = set(), {wait}, [wait]
    while stack:
        for user in stack.pop().users:
            if is_view(user):
                if user not in seen:
                    seen.add(user)
                    stack.append(user)
            else:
                consumers.add(user)
    return consumers


def _signature_reason(node: fx.Node, kind: str, groups: dict) -> Optional[str]:
    """Return why the node signature, group or parameter identity blocks bucketing."""
    backward = bool(node.meta.get("autograd_backward", False))
    if backward != (kind != "fsdp_all_gather"):
        return "phase_mismatch"
    if node.kwargs or len(node.args) != (4 if kind == "fsdp_reduce_scatter" else 3):
        return "unsupported_signature"
    source, group = node.args[0], node.args[-1]
    if not isinstance(source, fx.Node) or not isinstance(group, str) or group not in groups:
        return "unknown_process_group"
    degree = node.args[-2]
    if not _static_int(degree) or degree <= 1 or degree != len(groups[group]):
        return "group_size_mismatch"
    if node.meta.get("comm_group", group) != group:
        return "group_metadata_mismatch"
    name, index = node.meta.get("param_name"), node.meta.get("state_idx")
    if not isinstance(name, str) or not name or not _static_int(index) or index < 0:
        return "missing_parameter_identity"
    return None


def _layout_reason(kind: str, degree: int, spec: dict) -> Optional[str]:
    """Return why the recorded bucket spec cannot be packed statically."""
    input_shape, output_shape = spec.get("input_shape"), spec.get("output_shape")
    for shape in (input_shape, output_shape):
        if (not isinstance(shape, (list, tuple)) or not shape
                or not all(_static_int(dim) and dim > 0 for dim in shape)):
            return "unknown_or_dynamic_shape"
    dtype, device = spec.get("dtype"), spec.get("device")
    if not isinstance(dtype, torch.dtype) or dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        return "unsupported_dtype"
    if not isinstance(device, str):
        return "unknown_device"
    try:
        torch.device(device)
    except (RuntimeError, ValueError):
        return "unknown_device"
    smaller, larger = (output_shape, input_shape) if kind == "fsdp_reduce_scatter" else (input_shape, output_shape)
    if tuple(larger) != (smaller[0] * degree, *smaller[1:]):
        return "invalid_dim0_layout"
    return None


def _candidate(node: fx.Node, module_fqns: Optional[tuple], groups: dict, order: dict) -> tuple:
    """Return (candidate, None) for a bucketable collective, else (None, skip reason)."""
    kind = node.meta.get("comm_type")
    if not is_op(node, _KINDS[kind]) or node.meta.get("fsdp_bucket_members"):
        return None, "unsupported_or_already_bucketed"
    reason = _signature_reason(node, kind, groups)
    if reason:
        return None, reason
    spec = node.meta.get("fsdp_bucket_spec", {})
    reason = _layout_reason(kind, node.args[-2], spec)
    if reason:
        return None, reason
    waits = list(node.users)
    if len(waits) != 1 or not is_op(waits[0], c10d.wait_tensor):
        return None, "unsupported_wait_users"
    consumers = _consumers(waits[0])
    if not consumers:
        return None, "no_consumer"
    if kind == "fsdp_reduce_scatter":
        if node.args[1] not in ("sum", "avg"):
            return None, "unsupported_reduce_op"
        if any(user.op != "output" for user in consumers):
            return None, "non_output_rs_consumer"
    if module_fqns is None:
        return None, "outside_module_plan"
    name = node.meta.get("param_name")
    index = node.meta.get("state_idx")
    backward = bool(node.meta.get("autograd_backward", False))
    candidate = Candidate(node, waits[0], node.args[0], name, index, kind, backward, module_fqns[0], module_fqns,
                          node.args[-1], groups[node.args[-1]], spec.get("dtype"), spec.get("device"),
                          tuple(spec.get("input_shape")), tuple(spec.get("output_shape")), order[node.args[0]],
                          min(order[user] for user in consumers),
                          node.args[1] if kind == "fsdp_reduce_scatter" else None)
    return candidate, None


def _membership_reason(members: list[Candidate], candidate: Candidate, policy: BucketPolicy) -> Optional[str]:
    """Return why compatibility, uniqueness or capacity rejects the candidate."""
    if members[-1].key != candidate.key:
        return "incompatible_key"
    combined = [*members, candidate]
    param_names = {m.param_name for m in combined}
    state_indices = {m.state_idx for m in combined}
    if len(param_names) != len(combined) or len(state_indices) != len(combined):
        return "repeated_parameter_instance"
    if policy.cap_bytes is not None and sum(m.size_bytes for m in combined) > policy.cap_bytes:
        return "capacity"
    return None


def _window_reason(members: list[Candidate], candidate: Candidate, policy: BucketPolicy) -> Optional[str]:
    """Return why scheduling windows or dependencies reject the fusion."""
    combined = [*members, candidate]
    if (policy.max_input_distance is not None
            and max(m.ready for m in combined) - min(m.ready for m in combined) > policy.max_input_distance):
        return "input_distance"
    if (policy.max_consumer_distance is not None
            and max(m.first_use for m in combined) - min(m.first_use for m in combined) > policy.max_consumer_distance):
        return "consumer_distance"
    # Only immutable result views may move after the latest input. Actual AG
    # consumers (including frees) must remain after that insertion point.
    if max(m.ready for m in combined) >= min(m.first_use for m in combined):
        return "dependency_conflict"
    return None


def _owned_free(node: fx.Node, candidate: Candidate, combined: list) -> bool:
    """Whether node frees exactly one member's waited result via copy-out."""
    return (candidate.kind != "fsdp_reduce_scatter"
            and node.op == "call_function" and node.target is _free_tensor_storage
            and not node.kwargs and any(node.args == (member.wait,) for member in combined))


def _intervening_reason(members: list[Candidate], candidate: Candidate, nodes: list, order: dict) -> Optional[str]:
    """Return why nodes between the first member and the candidate block fusion."""
    combined = [*members, candidate]
    owned = {m.launch for m in combined}
    preparations = set()
    for member in combined:
        if member.kind == "fsdp_reduce_scatter":
            preparations.update(rs_layout_preparation(member.launch))
    for node in nodes[order[members[0].launch] + 1:order[candidate.launch]]:
        if node in preparations:
            # Only these default, private RS layout nodes are known pure here.
            continue
        if node.meta.get("comm_type") in _FREES:
            # Copy-out preserves each member's independent storage and original
            # free point. Only that exact materialization may be crossed.
            if not _owned_free(node, candidate, combined):
                return "effect_or_storage_release"
        elif is_barrier(node):
            return "effect_or_storage_release"
        if is_collective(node) and node not in owned:
            return "intervening_collective"
        if node.op not in ("placeholder", "get_attr", "output") and (
                bool(node.meta.get("autograd_backward", False)) != candidate.backward):
            return "phase_boundary"
    return None


def _split_reason(members: list[Candidate], candidate: Candidate, nodes: list, order: dict,
                  policy: BucketPolicy) -> Optional[str]:
    """Return why candidate cannot join members, or None when fusion is safe."""
    reason = _membership_reason(members, candidate, policy)
    if reason:
        return reason
    reason = _window_reason(members, candidate, policy)
    if reason:
        return reason
    return _intervening_reason(members, candidate, nodes, order)


def _finalize_bucket(members: list[Candidate]) -> Bucket:
    """Freeze bucket members in deterministic (state_idx, param_name) order."""
    ordered = sorted(members, key=lambda member: (member.state_idx, member.param_name))
    return Bucket(tuple(ordered))


def _physical_buckets(candidates: list[Candidate], nodes: list, order: dict,
                      policy: BucketPolicy, decisions: list) -> list[Bucket]:
    """Group by module/phase/compatibility, then split where legal motion stops."""
    compatible = {}
    for candidate in candidates:
        compatible.setdefault(candidate.key, []).append(candidate)
    buckets = []
    for group in compatible.values():
        current = []
        for candidate in group:
            reason = _split_reason(current, candidate, nodes, order, policy) if current else None
            if reason:
                buckets.append(_finalize_bucket(current))
                decisions.append({"node": candidate.launch.name, "previous_node": current[-1].launch.name,
                                  "action": "split", "reason": reason})
                current = []
            current.append(candidate)
            if policy.cap_bytes is not None and candidate.size_bytes > policy.cap_bytes:
                decisions.append({"node": candidate.launch.name, "action": "singleton", "reason": "oversized"})
        if current:
            buckets.append(_finalize_bucket(current))
    # Model declaration order never silently reorders the collective stream.
    buckets.sort(key=lambda bucket: min(order[member.launch] for member in bucket.members))
    return buckets


def _parameter_scope(name: object, module_plans: Optional[tuple]) -> Optional[tuple[str, ...]]:
    if not isinstance(name, str) or not name:
        return None
    if module_plans is not None:
        return match_module_bucket(name, module_plans)
    return (name.rpartition(".")[0] or f"<parameter:{name}>",)


def _log_plan(plan: BucketPlan, selected: dict, counts: Counter, order: dict) -> None:
    """Log diagnostics where the plan is produced, without a second representation."""
    _LOG.info("FSDP bucket plan: graph=%s summary=%s", plan.signature[:12], plan.summary)
    if not _LOG.isEnabledFor(logging.DEBUG):
        return
    for scope, names in selected.items():
        _LOG.debug("FSDP bucket scope: graph=%s modules=%s parameters=%s collectives=%s",
                   plan.signature[:12], scope, sorted(names), counts[scope])
    for decision in plan.decisions:
        _LOG.debug("FSDP bucket decision: graph=%s %s", plan.signature[:12], decision)
    for bucket in plan.buckets:
        members = bucket.members
        first = members[0]
        _LOG.debug("FSDP bucket planned: bucket=%s modules=%s kind=%s group=%s ranks=%s dtype=%s device=%s "
                   "reduce_op=%s members=%s input_span=%s original_launches=%s action=%s",
                   bucket.bucket_id, first.module_fqns, first.kind, first.group, first.ranks, first.dtype,
                   first.device, first.reduce_op, [member.param_name for member in members],
                   (min(member.ready for member in members), max(member.ready for member in members)),
                   [(member.launch.name, order[member.launch]) for member in members],
                   "rewrite_rs" if bucket.supports_rs_rewrite else
                   ("rewrite_ag_copy_out" if bucket.supports_ag_rewrite else "singleton"))


def _validate_group_ranks(groups: dict) -> None:
    """Reject process-group names or rank lists that cannot define a consistent plan."""
    for group, ranks in groups.items():
        if (not isinstance(group, str) or not ranks or len(set(ranks)) != len(ranks)
                or any(not _static_int(rank) or rank < 0 for rank in ranks)):
            raise ValueError("group_ranks must map PG names to distinct ordered non-negative global ranks")


def _record_wire_collective(wire: dict, node: fx.Node, group: str) -> None:
    """Append one rank-comparable wire record for a tagged collective."""
    spec = dict(node.meta.get("fsdp_bucket_spec", {}))
    if "device" in spec:
        spec["device"] = str(spec["device"]).split(":", 1)[0]
    reduce_op = (node.args[1] if node.meta.get("comm_type") == "fsdp_reduce_scatter"
                 and len(node.args) > 1 else None)
    wire[group].append((node.meta.get("comm_type"), node.meta.get("param_name"),
                        node.meta.get("state_idx"), spec, reduce_op))


def _scan_collectives(nodes: list, groups: dict, module_plans: Optional[tuple],
                      pp_degree: int, dp_mode: str) -> tuple:
    """Scan tagged collectives into wire records, skip decisions and candidates.

    Returns:
        (candidates, wire, selected, counts, decisions, tagged_count); physical
        bucketing appends further split/singleton decisions to ``decisions``.
    """
    order = {node: index for index, node in enumerate(nodes)}
    decisions, candidates = [], []
    wire = {group: [] for group in groups}
    selected = {scope: set() for scope in module_plans or ()}
    counts = Counter()
    tagged_count = 0
    for node in nodes:
        if node.meta.get("comm_type") not in _KINDS:
            continue
        tagged_count += 1
        name = node.meta.get("param_name")
        scope = _parameter_scope(name, module_plans)
        if scope is not None:
            selected.setdefault(scope, set()).add(name)
            counts[scope] += 1
        group = node.args[-1] if node.args else None
        if isinstance(group, str) and group in wire:
            _record_wire_collective(wire, node, group)
        if pp_degree != 1 or dp_mode != "fsdp":
            candidate, reason = None, "unsupported_parallel_mode"
        else:
            candidate, reason = _candidate(node, scope, groups, order)
        if reason:
            decisions.append({"node": node.name, "action": "skip", "reason": reason})
        else:
            candidates.append(candidate)
    return candidates, wire, selected, counts, decisions, tagged_count


def _unmatched_module_paths(selected: dict) -> list:
    """List scope paths that ended up selecting no parameter at all."""
    unmatched = []
    for scope, names in selected.items():
        for path in scope:
            if not any(name.startswith(path + ".") for name in names):
                unmatched.append(path)
    return unmatched


def plan_fsdp_buckets(graph_module: fx.GraphModule, *, group_ranks: Mapping[str, tuple[int, ...]],
                      module_bucket_plans: Optional[ModuleBucketPlans] = None,
                      policy: Optional[BucketPolicy] = None, pp_degree: int = 1,
                      dp_mode: str = "fsdp") -> BucketPlan:
    """Collect by model module and compatibility, then plan safe physical buckets.

    Args:
        graph_module: Joint graph after FSDPPass and before overlap scheduling.
        group_ranks: Exact PG names mapped to ordered global ranks, supplied by caller.
        module_bucket_plans: Explicit FQN groups, as in TorchTitan. None falls back
            to immediate parameter parents; an empty list selects nothing. Unmatched
            parameters remain untouched. Prefix ownership must be unambiguous.
        policy: Optional capacity/distance limits; no size/distance cap by default.
        pp_degree: Pipeline degree; this first implementation supports only one stage.
        dp_mode: Only pure FSDP is supported; HSDP/DDP receive explicit skip decisions.

    Returns:
        Read-only plan with stable parameter layouts and explicit skip/split reasons.
        RS and AG may launch after the latest input, before any real consumer.
        AG rewriting uses independent copies, allowing exact member frees inside a bucket.
    """
    policy = policy or BucketPolicy()
    module_plans = None if module_bucket_plans is None else normalize_module_bucket_plans(module_bucket_plans)
    groups = {group: tuple(ranks) for group, ranks in group_ranks.items()}
    _validate_group_ranks(groups)
    graph_module.graph.lint()
    nodes = list(graph_module.graph.nodes)
    order = {node: index for index, node in enumerate(nodes)}
    candidates, wire, selected, counts, decisions, tagged_count = _scan_collectives(
        nodes, groups, module_plans, pp_degree, dp_mode)
    buckets = _physical_buckets(candidates, nodes, order, policy, decisions)
    signature = graph_signature(graph_module.graph)
    buckets = tuple(replace(bucket, bucket_id=f"{signature[:12]}:bucket_{index}")
                    for index, bucket in enumerate(buckets))
    unmatched_module_paths = (_unmatched_module_paths(selected)
                              if module_plans is not None else [])
    summary = {"tagged_collectives": tagged_count, "selected_parameters": sum(map(len, selected.values())),
               "selected_collectives": sum(counts.values()), "candidate_collectives": len(candidates),
               "buckets": len(buckets), "singletons": sum(len(bucket.members) == 1 for bucket in buckets),
               "rs_rewrite_candidates": sum(bucket.supports_rs_rewrite for bucket in buckets),
               "ag_rewrite_candidates": sum(bucket.supports_ag_rewrite for bucket in buckets),
               "reason_counts": {action: dict(Counter(item["reason"] for item in decisions if item["action"] == action))
                                 for action in ("skip", "split", "singleton")},
               "unmatched_module_paths": unmatched_module_paths}
    plan = BucketPlan(buckets, tuple(decisions), signature, groups, wire, module_plans, summary)
    _log_plan(plan, selected, counts, order)
    return plan
