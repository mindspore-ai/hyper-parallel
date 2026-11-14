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
"""Pool quantile-balancing statistics over declared unique-token groups."""

from collections.abc import Sequence
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.modules.quantile_balancing import QuantileBalancer


def _quantile_group_size(process_groups: Sequence[Any], bucket_size: int) -> int:
    """Validate orthogonal group inputs and compute the declared row-bound multiplier."""
    if bucket_size < 1 or len({id(group) for group in process_groups}) != len(process_groups):
        raise ValueError("QB requires positive bucket_size and unique process_groups")
    if process_groups and not dist.is_initialized():
        raise ValueError("QB process_groups require initialized torch.distributed")
    group_size = 1
    for group in process_groups:
        group_size *= dist.get_world_size(group)
    return group_size


@torch.no_grad()
def finalize_quantile_balancers(
    entries: Sequence[tuple[QuantileBalancer, torch.Tensor]], *,
    process_groups: Sequence[Any] = (), bucket_size: int = 8,
) -> list[torch.Tensor]:
    """Commit compatible routers in bounded buckets without reading device counts.

    Args:
        entries: Balancer/bias pairs in the same order on every participating rank.
        process_groups: Orthogonal groups containing unique tokens, excluding EP replicas.
        bucket_size: Maximum histograms retained in one reduction buffer.

    Returns:
        Device boolean scalars indicating nonempty global batches. The callback
        need not consume them. Empty batches preserve bias and update counters.

    Note:
        Int32 storage requires a declared local row bound. Reduction uses int32
        only when that bound times the process-group sizes fits int32; otherwise
        it promotes to int64. Each histogram row already carries the token count.
    """
    group_size = _quantile_group_size(process_groups, bucket_size)
    results = []
    # Compatible dtype/device/shape groups are separated by the model-owned caller.
    for start in range(0, len(entries), bucket_size):
        bucket = entries[start:start + bucket_size]
        biases = []
        for balancer, bias in bucket:
            biases.append(balancer.prepare_statistics(bias))
        use_int32 = all(qb.histogram.dtype == torch.int32 and
                        qb.max_tokens_per_rank * group_size <= torch.iinfo(torch.int32).max for qb, _ in bucket)
        pooled = torch.stack([qb.histogram for qb, _ in bucket]).to(torch.int32 if use_int32 else torch.int64)
        for group in process_groups:
            dist.all_reduce(pooled, op=dist.ReduceOp.SUM, group=group)
        for (balancer, _), bias, histogram in zip(bucket, biases, pooled):
            results.append(balancer.commit_statistics(bias, histogram))
    return results


def configure_router_groups(gate: torch.nn.Module, mesh: Any, cp_mesh: Any) -> tuple[Any, ...]:
    """Pool each distinct DP and CP token shard once, excluding overlapping EP.

    Args:
        gate: Router receiving expert_bias_update_groups for its QB callback.
        mesh: Active plan submesh; its root supplies DP token-ownership axes.
        cp_mesh: Chronological context mesh, or None without context parallelism.

    Returns:
        Nontrivial orthogonal DP/CP process groups assigned to the router.
    """
    # Local compute receives the active plan submesh, which can omit DP.
    # Resolve its public parent mesh before assigning a callback override.
    token_mesh = mesh
    while getattr(token_mesh, "root_mesh", None) is not None:
        token_mesh = token_mesh.root_mesh
    names = tuple(getattr(token_mesh, "mesh_dim_names", ()) or ())
    dp_names = ("dp",) if "dp" in names else tuple(name for name in ("dp_replicate", "dp_shard") if name in names)
    axes = [token_mesh[name] for name in dp_names]
    if cp_mesh is not None:
        axes.append(cp_mesh)
    groups = tuple(axis.get_group() for axis in axes if axis.size() > 1)
    if len({id(group) for group in groups}) != len(groups):
        raise ValueError("Router statistics require distinct orthogonal DP and CP groups")
    gate.expert_bias_update_groups = groups
    return groups
