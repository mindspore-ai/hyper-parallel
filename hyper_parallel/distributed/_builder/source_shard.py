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

from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple, TypeAlias

import hyper_parallel.core.fully_shard.utils as fully_shard_utils
from hyper_parallel import DeviceMesh, DTensor, Replicate
from hyper_parallel.core.dtensor.placement_types import Partial, Placement, Shard
from hyper_parallel.platform import get_platform
from hyper_parallel.distributed.plan import ShardingPlan
from hyper_parallel.distributed.recipe_spec import resolve_placements

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


# ────────────────────────────────────────────────────────────────────────────
# FSDP2 source-layout discovery (moved verbatim from components/distributed/
# fsdp2.py in stage 4f, 05 §15.2.4 row 406; ``self`` became an explicit
# ``manager`` parameter — FSDP2Manager calls these free functions).
# ────────────────────────────────────────────────────────────────────────────

platform = get_platform()
ModuleClass = platform.Module
ParameterClass = platform.Parameter

SourceShardInfoByFQN: TypeAlias = Mapping[  # pylint: disable=invalid-name
    str, tuple[tuple[Placement, ...], DeviceMesh]
]


SourceShardInfoByParam: TypeAlias = dict[  # pylint: disable=invalid-name
    ParameterClass,
    "fully_shard_utils.SourceShardMetaInfo",
]


def _build_dtensor_source_shard_info(
    manager,
    model: ModuleClass,
) -> SourceShardInfoByParam | None:
    """Build source metadata from DTensor parameters when TP is active."""
    if manager.mesh_context.tp_size <= 1:
        return None

    parameters = list(model.parameters())
    if not parameters or not all(
        isinstance(parameter, DTensor) for parameter in parameters
    ):
        raise ValueError(
            "TP is enabled but source_shard_info is missing and "
            "not all model parameters are DTensors"
        )
    return {
        parameter: fully_shard_utils.SourceShardMetaInfo(  # pylint: disable=no-member
            mesh=parameter.device_mesh,
            placements=tuple(parameter.placements),
            origin_is_dtensor=True,
        )
        for parameter in parameters
    }


def _build_parameter_source_shard_info(
    parameter_fqn: str,
    placements: tuple[Placement, ...],
    source_mesh: DeviceMesh,
) -> fully_shard_utils.SourceShardMetaInfo:
    """Validate and build source metadata for one model parameter."""
    if any(isinstance(placement, Partial) for placement in placements):
        raise ValueError(
            "source_shard_info does not support Partial placement: "
            f"{parameter_fqn}"
        )
    return fully_shard_utils.SourceShardMetaInfo(  # pylint: disable=no-member
        mesh=source_mesh,
        placements=tuple(placements),
        origin_is_dtensor=False,
    )


def _record_parameter_source_shard_info(
    metadata_by_parameter: SourceShardInfoByParam,
    parameter: ParameterClass,
    parameter_fqn: str,
    parameter_source_shard_info: fully_shard_utils.SourceShardMetaInfo,
) -> None:
    """Record one layout while rejecting conflicting tied aliases."""
    previous_source_shard_info = metadata_by_parameter.get(parameter)
    if previous_source_shard_info is not None and (
        previous_source_shard_info.mesh is not parameter_source_shard_info.mesh
        or previous_source_shard_info.placements
        != parameter_source_shard_info.placements
    ):
        raise ValueError(
            "Tied parameter aliases have conflicting source layouts; "
            f"conflict found at {parameter_fqn}"
        )
    metadata_by_parameter[parameter] = parameter_source_shard_info


def _build_source_shard_info_by_param(
    manager,
    model: ModuleClass,
    source_shard_info: SourceShardInfoByFQN | None,
) -> SourceShardInfoByParam | None:
    """Resolve stable FQNs to parameter identities after model rewrites.

    Activation-checkpoint wrappers may add an internal child-module prefix,
    but the supported wrapper contract removes that prefix from
    ``named_modules``/``named_parameters``. Therefore planner metadata
    produced before checkpointing remains resolvable here, while FSDP still
    receives the final parameter objects that it will manage.

    Args:
        model: Model after sharding and checkpoint rewrites, before layer
            compile is installed.
        source_shard_info: Source-layout metadata produced by the TP planner.

    Returns:
        Parameter-keyed metadata for later ``fully_shard`` calls, or
        ``None`` when tensor parallelism is disabled.

    Raises:
        ValueError: If TP metadata is missing, references an unknown FQN,
            contains ``Partial``, or gives tied aliases conflicting layouts.
    """
    if source_shard_info is None:
        return _build_dtensor_source_shard_info(manager, model)
    if manager.mesh_context.tp_size > 1 and not source_shard_info:
        raise ValueError("TP is enabled but source_shard_info is empty")

    parameters_by_fqn = dict(model.named_parameters(remove_duplicate=False))
    metadata_by_parameter: SourceShardInfoByParam = {}
    for parameter_fqn, (placements, source_mesh) in source_shard_info.items():
        parameter = parameters_by_fqn.get(parameter_fqn)
        if parameter is None:
            raise ValueError(
                f"source_shard_info contains unknown parameter FQN: {parameter_fqn}"
            )
        parameter_source_shard_info = _build_parameter_source_shard_info(
            parameter_fqn,
            placements,
            source_mesh,
        )
        _record_parameter_source_shard_info(
            metadata_by_parameter,
            parameter,
            parameter_fqn,
            parameter_source_shard_info,
        )
    return metadata_by_parameter


def _get_default_source_shard_info(manager) -> fully_shard_utils.SourceShardMetaInfo:
    """Build all-Replicate source metadata covering every non-FSDP mesh axis.

    Used to complete source_shard_info for parameters the plan does not mention;
    the axis set must match the per-parameter entries so that all parameters
    in one ``fully_shard`` unit share the same source-mesh dimensionality.
    """
    world_mesh = manager.mesh_context.fsdp_non_moe_mesh or manager.mesh_context.device_mesh
    if world_mesh is None or world_mesh.mesh_dim_names is None:
        raise ValueError("TP metadata completion requires a named world mesh")
    source_dim_names = tuple(
        dim_name
        for dim_name in world_mesh.mesh_dim_names
        if dim_name not in FSDP_OWNED_DIMS
    )
    if not source_dim_names:
        raise ValueError(
            "TP is enabled but the world mesh has no non-FSDP source dimension"
        )
    source_mesh = (
        world_mesh
        if tuple(world_mesh.mesh_dim_names) == source_dim_names
        else world_mesh[source_dim_names[0]]
        if len(source_dim_names) == 1
        else world_mesh[source_dim_names]
    )
    return fully_shard_utils.SourceShardMetaInfo(  # pylint: disable=no-member
        mesh=source_mesh,
        placements=tuple(Replicate() for _ in source_dim_names),
        origin_is_dtensor=False,
    )


def _build_managed_source_shard_info(
    manager,
    owner: ModuleClass,
    owner_by_parameter: Mapping[ParameterClass, ModuleClass],
    metadata_by_parameter: SourceShardInfoByParam | None,
) -> SourceShardInfoByParam | None:
    """Select and complete metadata for parameters managed by one wrap call."""
    if metadata_by_parameter is None:
        return None

    default_source_shard_info = None
    if (
        manager.mesh_context.fsdp_non_moe_mesh is not None
        or manager.mesh_context.device_mesh is not None
    ):
        default_source_shard_info = _get_default_source_shard_info(manager, )
    managed_source_shard_info = {}
    for parameter, parameter_owner in owner_by_parameter.items():
        if parameter_owner is not owner:
            continue
        source_shard_info = metadata_by_parameter.get(parameter)
        if source_shard_info is None:
            if default_source_shard_info is None:
                raise ValueError(
                    "source_shard_info is present but does not cover an FSDP-managed parameter"
                )
            source_shard_info = default_source_shard_info
        managed_source_shard_info[parameter] = source_shard_info
    return managed_source_shard_info


def _source_infos_for_fully_shard(
    managed_source_shard_info: SourceShardInfoByParam | None,
) -> SourceShardInfoByParam | None:
    """Return explicit metadata only for plain source-layout parameters.

    Validate-mode parameters remain native DTensors. The platform FSDP
    state derives their source layouts from each parameter directly and
    rejects duplicate explicit metadata.
    """
    if managed_source_shard_info and all(
        source_shard_info.origin_is_dtensor
        for source_shard_info in managed_source_shard_info.values()
    ):
        return None
    return managed_source_shard_info
