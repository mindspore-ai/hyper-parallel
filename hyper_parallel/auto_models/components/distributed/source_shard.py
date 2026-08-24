# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""source_shard: build_source_shard_info (05 §6.7.1).

source_shard_info is read from the ShardingPlan (rather than from DTensors — under
production the parameters have already been unwrapped by
_local_params_context, and only the plan retains the complete placement
information).

Each entry records the parameter's **complete source layout**: the placements
on every source (non-FSDP) mesh axis, together with the source sub-mesh.
``fully_shard`` later prefixes its own FSDP shard/replicate dimensions, so the
final sharded parameter carries the full distributed layout (FSDP dims +
source dims) for downstream consumers such as distributed checkpointing.
"""

from typing import Any, Dict, List, Optional, Tuple

from hyper_parallel.core.dtensor.placement_types import Placement, Shard
from hyper_parallel.auto_models.components.distributed.sharding_config import (
    ShardingPlan,
    resolve_placements,
)

# Mesh dimensions whose sharding semantics belong to FSDP (or to other
# parallelism concerns), never to the source layout recorded here:
# - dp/dp_shard/dp_replicate/cp: FSDP shards weights over the whole DP+CP
#   domain (cp is folded into the fsdp shard axis at mesh build time), and
#   FSDP's reduce-scatter already covers gradient sync on these axes;
# - pp: different stages hold different parameters (distinguished by FQN),
#   a weight is never distributed along pp;
# - fsdp_*/edp_*: the FSDP child-mesh axes themselves.
# Declaring a Shard placement for a weight on one of these axes would
# double-count the sharding (Phase A shard + FSDP shard on the same ranks),
# so build_source_shard_info rejects it fail-fast.
FSDP_OWNED_DIMS = frozenset({
    "dp", "dp_replicate", "dp_shard", "cp", "pp",
    "fsdp_replicate", "fsdp_shard",
    "edp", "edp_replicate", "edp_shard",
})


def _source_dim_names(mesh, fallback_dim_names):
    """Return the non-FSDP axis names of a source mesh.

    Falls back to the plan's axis names when the mesh carries no dimension
    names (e.g. a mock or a plain 1-D TP mesh).
    """
    dim_names = getattr(mesh, "mesh_dim_names", None) if mesh is not None else None
    if dim_names is None:
        dim_names = tuple(fallback_dim_names or ())
    return tuple(name for name in dim_names if name not in FSDP_OWNED_DIMS)


def _source_sub_mesh(mesh, source_dim_names):
    """Slice the source sub-mesh covering exactly ``source_dim_names``."""
    if mesh is None:
        return None
    dim_names = getattr(mesh, "mesh_dim_names", None)
    if dim_names is None or tuple(dim_names) == tuple(source_dim_names):
        return mesh
    if not source_dim_names:
        return None
    if len(source_dim_names) == 1:
        return mesh[source_dim_names[0]]
    return mesh[source_dim_names]


def _check_fsdp_owned_axes(full_fqn, named_placement):
    """Reject weight sharding declared on axes owned by FSDP/PP."""
    for axis, placement in named_placement.items():
        if axis in FSDP_OWNED_DIMS and not placement.is_replicate():
            raise ValueError(
                f"{full_fqn}: placement {placement} on axis {axis!r} conflicts with "
                "FSDP ownership of that axis (weights are sharded over the DP+CP "
                "domain by fully_shard itself); declare Replicate or remove the axis"
            )


def build_source_shard_info(
    plan: ShardingPlan,
    dense_source_mesh: Any,
    *,
    expert_source_mesh: Any = None,
    tied_pairs: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, Tuple[Tuple[Placement, ...], Any]]:
    """Build complete source-layout metadata for dense and routed-expert parameters.

    Args:
        plan: The finalized ShardingPlan.
        dense_source_mesh: Dense-region source mesh. May be the full dense
            FSDP mesh (e.g. ``(fsdp_replicate, fsdp_shard, tp)``) — FSDP-owned
            axes are stripped; or a plain TP child mesh (used as-is).
        expert_source_mesh: Full expert mesh (e.g. ``(edp_shard, ep)``) or the
            EP child mesh; required when any spec marks routed experts.
        tied_pairs: parameter pairs with shared storage (defaults to
            plan.tied_pairs). Both ends of a tied pair must map to the same
            placements — axis by axis, the finer sharding wins (Shard takes
            precedence over Replicate), guaranteeing consistent TP all-reduce /
            reduce-scatter semantics on both ends.

    Returns:
        ``{param_fqn: (placements_tuple, source_sub_mesh)}`` where
        ``placements_tuple`` holds one placement per source (non-FSDP) mesh
        axis, missing axes filled with ``Replicate()``.
    """
    dense_source_dims = _source_dim_names(dense_source_mesh, plan.mesh_dim_names)
    dense_mesh = _source_sub_mesh(dense_source_mesh, dense_source_dims)
    expert_source_dims = _source_dim_names(expert_source_mesh, ("ep",))
    expert_mesh = _source_sub_mesh(expert_source_mesh, expert_source_dims)

    info = {}
    for fqn, spec in plan.modules.items():
        for param_name, named_placement in spec.params.items():
            full_fqn = f"{fqn}.{param_name}"
            _check_fsdp_owned_axes(full_fqn, named_placement)
            if spec._ep_size > 0 and param_name.startswith("experts."):  # pylint: disable=protected-access
                if expert_source_mesh is None:
                    raise ValueError(
                        "Routed expert metadata requires an expert EP source mesh"
                    )
                placements = tuple(resolve_placements(named_placement, expert_source_dims))
                info[full_fqn] = (placements, expert_mesh)
            else:
                placements = tuple(resolve_placements(named_placement, dense_source_dims))
                info[full_fqn] = (placements, dense_mesh)

    pairs = tied_pairs if tied_pairs is not None else plan.tied_pairs
    if pairs:
        for a, b in pairs:
            if a in info and b in info:
                pa, _ = info[a]
                pb, _ = info[b]
                if pa != pb and len(pa) == len(pb):
                    norm = tuple(
                        x if isinstance(x, Shard) else y
                        for x, y in zip(pa, pb)
                    )
                    info[a] = (norm, info[a][1])
                    info[b] = (norm, info[b][1])
    return info
