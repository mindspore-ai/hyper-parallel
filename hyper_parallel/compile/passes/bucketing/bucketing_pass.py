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
"""Shared FSDP bucket graph rewriting with kind-specific packing and ownership."""

import logging
import math
import operator
from functools import partial
from typing import Any

import torch
import torch.distributed as dist
from torch import fx
from torch.ops import _c10d_functional as c10d
from torch.distributed.distributed_c10d import _resolve_process_group

from ...pass_config import PassConfig
from ..base import GraphPass
from ..overlap.utils import is_view, rs_layout_preparation
from .planner import (
    Bucket, BucketPlan, Candidate, build_module_bucket_plans, graph_signature, plan_fsdp_buckets,
    validate_group_plans,
)

_LOG = logging.getLogger(__name__)
_GATHERS = ("fsdp_all_gather", "fsdp_all_gather_backward")


def _create_node(graph: fx.Graph, bucket: Bucket, first: Candidate, target: object, args: tuple,
                 *, preparation: bool = False, **kwargs: object) -> fx.Node:
    """Create a node carrying phase, scope and optional private preparation ownership."""
    node = graph.call_function(target, args, kwargs)
    node.meta.update(autograd_backward=first.backward, fsdp_bucket_id=bucket.bucket_id,
                     fsdp_bucket_module_fqns=first.module_fqns)
    if preparation:
        node.meta["fsdp_bucket_prepare"] = bucket.bucket_id
    return node


def _prepare_rs_layouts(members: tuple, bucket: Bucket) -> None:
    """Expose private RS preparation nodes as pure contiguous calls."""
    for member in members:
        for source in rs_layout_preparation(member.launch):
            # Preserve upstream contiguous and expose its purity to scheduling.
            source.op = "call_function"
            source.target = torch.ops.aten.contiguous.default
            source.meta.update(autograd_backward=True, fsdp_bucket_prepare=bucket.bucket_id,
                               fsdp_bucket_id=bucket.bucket_id, fsdp_reduce_scatter_prepare=member.launch.name)


def _bucket_anchor(order: dict, members: tuple) -> tuple:
    """Return (insertion anchor, first launch) placing the bucket after all inputs."""
    first_launch = min((member.launch for member in members), key=order.get)
    latest_input = max((member.launch.args[0] for member in members), key=order.get)
    anchor = first_launch if order[latest_input] < order[first_launch] else latest_input.next
    return anchor, first_launch


def _delayed_result_views(members: tuple, order: dict, anchor: fx.Node) -> list:
    """Collect immutable member views scheduled before the insertion anchor."""
    pending = [member.wait for member in members]
    views = set()
    while pending:
        for user in pending.pop().users:
            if is_view(user) and user not in views:
                views.add(user)
                pending.append(user)
    return sorted((node for node in views if order[node] < order[anchor]), key=order.get)


def _launch_packed_collective(create, packed, first: Candidate, members: tuple, degree: int,
                              gather: bool, numels: list) -> tuple:
    """Create the fused collective launch and its wait node at the current insertion point."""
    if gather:
        launch = create(c10d.all_gather_into_tensor.default, (packed, degree, first.group))
        replica_bytes = sum(member.size_bytes for member in members)
        launch.meta.update(fsdp_unsharded_nbytes=replica_bytes, fsdp_bucket_workspace_nbytes=replica_bytes)
    else:
        packed = create(torch.ops.aten.reshape.default, (packed, (-1,)), preparation=True)
        launch = create(c10d.reduce_scatter_tensor.default, (packed, first.reduce_op, degree, first.group))
    launch.meta.update(comm_type=first.kind, comm_group=first.group, fsdp_degree=degree,
                       fsdp_bucket_members=tuple(member.param_name for member in members),
                       fsdp_bucket_spec={"input_shape": (sum(numels) * (1 if gather else degree),),
                                         "output_shape": (sum(numels) * (degree if gather else 1),),
                                         "dtype": first.dtype, "device": first.device})
    wait = create(c10d.wait_tensor.default, (launch,))
    wait.meta["wait_for"] = launch.name
    return launch, wait


def _unpack_member_outputs(create, wait: fx.Node, numels: list, members: tuple,
                           gather: bool, degree: int, bucket: Bucket) -> list:
    """Split the waited packed result back into per-member reshape outputs."""
    unpack = create(torch.ops.aten.reshape.default, (wait, (degree, -1))) if gather else wait
    parts = create(torch.ops.aten.split_with_sizes.default, (unpack, numels, 1 if gather else 0))
    outputs = []
    for index, member in enumerate(members):
        part = create(operator.getitem, (parts, index))
        if gather:
            # reshape/contiguous may alias; each original storage free needs
            # a copy independent of both the workspace and other members.
            part = create(torch.ops.aten.clone.default, (part,), memory_format=torch.contiguous_format)
            part.meta["fsdp_bucket_copy_out"] = bucket.bucket_id
        output = create(torch.ops.aten.reshape.default, (part, member.output_shape))
        output.meta.update(param_name=member.param_name, state_idx=member.state_idx)
        outputs.append(output)
    return outputs


def _erase_members(graph: fx.Graph, members: tuple, launch: fx.Node, gather: bool) -> None:
    """Remove original wait/launch pairs and retarget gather back-pointers."""
    for member in members:
        if gather:
            for node in graph.nodes:
                if node.meta.get("fsdp_ag_node") == member.launch.name:
                    node.meta["fsdp_ag_node"] = launch.name
        graph.erase_node(member.wait)
        graph.erase_node(member.launch)


def _merge_bucket(graph: fx.Graph, bucket: Bucket) -> fx.Node:
    """Place one packed collective after its inputs and repair immutable result views."""
    members = bucket.members
    first = members[0]
    gather = bucket.supports_ag_rewrite
    degree = len(first.ranks)
    if not gather:
        _prepare_rs_layouts(members, bucket)
    order = {node: index for index, node in enumerate(graph.nodes)}
    anchor, first_launch = _bucket_anchor(order, members)
    delayed_views = _delayed_result_views(members, order, anchor)
    numels = [math.prod(member.input_shape if gather else member.output_shape) for member in members]
    _LOG.debug("FSDP bucket preparing: bucket=%s placement=%s anchor=%s members=%s",
               bucket.bucket_id, "before_first_launch" if anchor is first_launch else "after_latest_input",
               anchor.name, [member.param_name for member in members])
    create = partial(_create_node, graph, bucket, first)
    with graph.inserting_before(anchor):
        rows = [create(torch.ops.aten.reshape.default,
                       (member.launch.args[0], (-1,) if gather else (degree, -1)), preparation=True)
                for member in members]
        packed = create(torch.ops.aten.cat.default, (rows, 0 if gather else 1), preparation=True)
        launch, wait = _launch_packed_collective(create, packed, first, members, degree, gather, numels)
        outputs = _unpack_member_outputs(create, wait, numels, members, gather, degree, bucket)
    for member, output in zip(members, outputs):
        member.wait.replace_all_uses_with(output)
    cursor = outputs[-1]
    for view in delayed_views:
        cursor.append(view)
        cursor = view
    _erase_members(graph, members, launch, gather)
    return launch


def _validate_plan_snapshot(graph_module: fx.GraphModule, plan: BucketPlan) -> None:
    """Reject plans built against a different or outdated graph snapshot."""
    if any(member.launch.graph is not graph_module.graph for bucket in plan.buckets for member in bucket.members):
        raise ValueError("Bucket plan belongs to a different graph")
    if graph_signature(graph_module.graph) != plan.signature:
        raise ValueError("Stale bucket plan: re-plan after changing the graph or its metadata")


def _rewrite_stats(graph_module: fx.GraphModule, buckets: list, kinds: tuple, before: int,
                   gather: bool) -> dict:
    """Summarize one completed kind-specific rewrite over the updated graph."""
    kind = "ag" if gather else "rs"
    member_bytes = sum(member.size_bytes for bucket in buckets for member in bucket.members)
    collectives_after = sum(node.meta.get("comm_type") in kinds for node in graph_module.graph.nodes)
    bytes_key = "copy_out_bytes" if gather else "packed_input_bytes"
    return {"buckets": len(buckets),
            "original_collectives": sum(len(bucket.members) for bucket in buckets),
            bytes_key: member_bytes,
            f"{kind}_collectives_before": before,
            f"{kind}_collectives_after": collectives_after}


def _apply_buckets(graph_module: fx.GraphModule, plan: BucketPlan, *, gather: bool) -> fx.GraphModule:
    """Validate the snapshot and publish kind-specific statistics only after success."""
    _validate_plan_snapshot(graph_module, plan)
    if gather:
        buckets = [bucket for bucket in plan.buckets if bucket.supports_ag_rewrite]
    else:
        buckets = [bucket for bucket in plan.buckets if bucket.supports_rs_rewrite]
    kinds = _GATHERS if gather else ("fsdp_reduce_scatter",)
    before = sum(node.meta.get("comm_type") in kinds for node in graph_module.graph.nodes)
    launches = [_merge_bucket(graph_module.graph, bucket) for bucket in buckets]
    if buckets:
        graph_module.graph.lint()
        graph_module.recompile()
    kind = "ag" if gather else "rs"
    stats = _rewrite_stats(graph_module, buckets, kinds, before, gather)
    graph_module.meta[f"fsdp_{kind}_bucketing"] = stats
    _LOG.info("FSDP bucket rewrite completed: kind=%s graph=%s stats=%s", kind, plan.signature[:12], stats)
    if _LOG.isEnabledFor(logging.DEBUG):
        order = {node: index for index, node in enumerate(graph_module.graph.nodes)}
        for launch in launches:
            _LOG.debug("FSDP bucket applied: kind=%s bucket=%s launch=%s index=%s coordinate_space=rewritten_graph",
                       kind, launch.meta["fsdp_bucket_id"], launch.name, order[launch])
    return graph_module


def apply_ag_buckets(graph_module: fx.GraphModule, plan: BucketPlan) -> fx.GraphModule:
    """Fuse non-out AG with independent copies, preserving original consumers and frees.

    Args:
        graph_module: Exact graph inspected by the planner, before overlap.
        plan: Fresh rank-consistent plan; re-plan after RS or any other graph change.

    Returns:
        Updated graph with each padded parameter owning independent storage.
        Singleton/RS buckets stay unchanged. The workspace uses normal backend
        stream/reference lifetime; prefetch bytes do not bound peak memory.

    Raises:
        ValueError: The plan belongs to another graph or is stale.

    Note:
        Caller owns cross-rank validation. Only immutable result views may move.
        Stats describe this invocation; failed rewrites do not automatically roll back.
    """
    return _apply_buckets(graph_module, plan, gather=True)


def apply_rs_buckets(graph_module: fx.GraphModule, plan: BucketPlan) -> fx.GraphModule:
    """Fuse RS with rank-major packing and shared output views, preserving output slots.

    Args:
        graph_module: Exact graph inspected by the planner, before overlap.
        plan: Fresh rank-consistent plan; re-plan after AG or any other graph change.

    Returns:
        Updated graph. Singleton/AG buckets stay unchanged; stats describe this call.
        Run overlap afterwards to schedule private preparation and launch together.

    Raises:
        ValueError: The plan belongs to another graph or is stale.

    Note:
        Caller owns cross-rank validation. Failed rewrites do not automatically roll back.
    """
    return _apply_buckets(graph_module, plan, gather=False)


class FsdpBucketingPass(GraphPass):
    """Plan and fuse FSDP communication after partitioning and before overlap."""

    name = "fsdp_bucketing"

    def run(self, graph_module: fx.GraphModule, pass_config: PassConfig, **kwargs: Any) -> fx.GraphModule:
        """Apply Python-configured module buckets with compile-time rank validation.

        Args:
            graph_module: Joint graph annotated by FSDPPass.
            pass_config: Bucketing switch and optional module scopes; identical across ranks.
            **kwargs: Live model and registered dense/expert FSDP group names from the compiler.

        Returns:
            Graph with fused AG/RS, or unchanged graph when disabled/unsupported.
        """
        if not pass_config.fsdp_bucketing or not pass_config.fsdp_enabled or not dist.is_initialized():
            return graph_module
        if dist.get_world_size() == 1:
            return graph_module
        if pass_config.dp_mode != "fsdp" or pass_config.pp_enabled or (pass_config.pp_degree or 1) != 1:
            _LOG.info("Skipped FSDP bucketing: requires dp_mode=fsdp and PP disabled")
            return graph_module
        modules = pass_config.fsdp_bucket_modules
        if modules is not None:
            modules = build_module_bucket_plans(kwargs["model"], module_bucket_plans=modules)
        names = [kwargs.get("fsdp_group_name", "fsdp")]
        if kwargs.get("expert_param_names"):
            names.append(kwargs.get("expert_fsdp_group_name", "fsdp_expert"))
        groups = {name: _resolve_process_group(name) for name in names}
        ranks = {name: tuple(dist.get_process_group_ranks(group)) for name, group in groups.items()}
        for apply in (apply_ag_buckets, apply_rs_buckets):
            plan = plan_fsdp_buckets(graph_module, group_ranks=ranks, module_bucket_plans=modules)
            for name, group in groups.items():
                report = {"group_ranks": {name: ranks[name]}, "group_digests": {name: plan.group_digest(name)}}
                reports = [None] * len(ranks[name])
                dist.all_gather_object(reports, report, group=group)
                validate_group_plans(dict(zip(ranks[name], reports)), name)
            apply(graph_module, plan)
        return graph_module
