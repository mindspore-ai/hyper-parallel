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
"""Private mixed-pipeline assembly used by Dry-run examples and profiling."""
# This experimental assembly is intentionally Torch-only.
# pylint: disable=forbidden-backend-import,protected-access

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, ContextManager, Optional

import torch
import torch.distributed as dist
from torch import nn

from hyper_parallel.distributed.apply import apply_sharding_plan
from hyper_parallel.distributed._builder.source_shard import (
    _build_source_shard_info_by_param,
    _get_default_source_shard_info,
    _source_infos_for_fully_shard,
)
from hyper_parallel.distributed.mesh import DistributedSetup, MeshContext
from hyper_parallel.distributed.plan import ShardingPlan
from hyper_parallel.distributed.recipe_spec import resolve_placements
from hyper_parallel.trainer.config import TrainerConfig
from hyper_parallel.trainer.dry_run_pipeline import DryRunBoundaryLeaf, DryRunPipelineChunk
from hyper_parallel import fully_shard, init_device_mesh
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.fully_shard.api import HSDPModule
from hyper_parallel.core.utils import compute_local_shape_and_global_offset
from hyper_parallel.platform.torch.dry_run import DryRunRuntime


@dataclass(frozen=True)
class _DryRunParallelContext:
    """Keep experimental root and PP meshes beside normal stage state."""

    setup: DistributedSetup
    root_mesh: Optional[DeviceMesh]
    pp_mesh: Optional[DeviceMesh]


@dataclass(frozen=True)
class _DryRunStageShardingResult:
    """Keep the planner result needed to resolve one stage's boundaries."""

    model: nn.Module
    source_shard_info: Optional[dict[str, Any]]
    plan: Optional[ShardingPlan]


@dataclass(frozen=True)
class _PreparedPipelineChunk:
    """Bundle one parallelized chunk with its static PP wire metadata."""

    chunk: DryRunPipelineChunk
    input_metadata: list[list[Any]]
    output_metadata: list[list[Any]]

    @property
    def module(self) -> nn.Module:
        """Return the stage module owned by the original chunk."""
        return self.chunk.module


def _build_pipeline_meshes(
        device_type: str,
        pp_size: int,
        dp_size: int,
        cp_size: int,
        tp_size: int,
) -> tuple[DeviceMesh, DeviceMesh, DeviceMesh]:
    """Build private root, PP, and stage-compute meshes."""
    root_mesh = init_device_mesh(
        device_type=device_type,
        mesh_shape=(pp_size, dp_size, cp_size, tp_size),
        mesh_dim_names=("pp", "dp", "cp", "tp"),
        init_backend=dist.is_initialized(),
    )
    return root_mesh, root_mesh["pp"], root_mesh[("dp", "cp", "tp")]


def _prewarm_pipeline_meshes(root_mesh: DeviceMesh, mesh_context: MeshContext, pp_mesh: DeviceMesh) -> None:
    """Create all process groups before FakeTensorMode or model execution."""
    meshes = [root_mesh, pp_mesh, mesh_context.device_mesh, mesh_context.fsdp_non_moe_mesh]
    for axis in ("dp", "cp", "tp"):
        meshes.append(mesh_context.device_mesh[axis])
    active_names = tuple(
        axis
        for axis, size in (("cp", mesh_context.cp_size), ("tp", mesh_context.tp_size))
        if size > 1
    )
    if active_names:
        active_mesh = mesh_context.device_mesh[active_names]
        meshes.append(active_mesh)
        meshes.extend(active_mesh[axis] for axis in active_names)
    if mesh_context.dp_cp_mesh is not None:
        meshes.append(mesh_context.dp_cp_mesh)
    dense_selector: str | tuple[str, str] = "fsdp_shard"
    if mesh_context.dp_replicate_size > 1:
        dense_selector = ("fsdp_replicate", "fsdp_shard")
    meshes.append(mesh_context.fsdp_non_moe_mesh[dense_selector])
    for mesh in meshes:
        for mesh_dim in range(mesh.ndim):
            mesh.get_group(mesh_dim)


def _prewarm_standard_meshes(mesh_context: MeshContext) -> None:
    """Materialize child meshes used by the existing non-pipeline path."""
    meshes = [mesh_context.device_mesh, mesh_context.fsdp_non_moe_mesh]
    for axis in ("dp", "cp", "tp"):
        meshes.append(mesh_context.device_mesh[axis])
    active_names = tuple(
        axis
        for axis, size in (("cp", mesh_context.cp_size), ("tp", mesh_context.tp_size))
        if size > 1
    )
    if active_names:
        active_mesh = mesh_context.device_mesh[active_names]
        meshes.append(active_mesh)
        meshes.extend(active_mesh[axis] for axis in active_names)
    if mesh_context.dp_cp_mesh is not None:
        meshes.append(mesh_context.dp_cp_mesh)
    dense_selector: str | tuple[str, str] = "fsdp_shard"
    if mesh_context.dp_replicate_size > 1:
        dense_selector = ("fsdp_replicate", "fsdp_shard")
    meshes.append(mesh_context.fsdp_non_moe_mesh[dense_selector])
    if mesh_context.fsdp_moe_mesh is not None:
        meshes.extend((mesh_context.fsdp_moe_mesh, mesh_context.fsdp_moe_mesh["ep"]))
        expert_selector: str | tuple[str, str] = "edp_shard"
        if "edp_replicate" in mesh_context.fsdp_moe_mesh.mesh_dim_names:
            expert_selector = ("edp_replicate", "edp_shard")
        meshes.append(mesh_context.fsdp_moe_mesh[expert_selector])
    for mesh in meshes:
        for mesh_dim in range(mesh.ndim):
            mesh.get_group(mesh_dim)


def _build_standard_distributed_setup(
        config: TrainerConfig,
        runtime: DryRunRuntime,
        device_type: str,
) -> _DryRunParallelContext:
    """Build the established non-pipeline MeshContext domains."""
    accelerator = config.accelerator
    tp_size = max(1, accelerator.tp_size)
    cp_size = max(1, accelerator.cp_size)
    ep_size = max(1, accelerator.ep_size)
    dp_size = runtime.world_size // (tp_size * cp_size)
    dp_shard_size = config.fsdp_config.dp_shard_size
    dp_replicate_size = dp_size * cp_size // dp_shard_size
    mesh_context = MeshContext(
        dp_size=dp_size,
        dp_replicate_size=dp_replicate_size,
        dp_shard_size=dp_shard_size,
        edp_shard_size=config.fsdp_config.edp_shard_size,
        tp_size=tp_size,
        cp_size=cp_size,
        pp_size=1,
        ep_size=ep_size,
        sequence_parallel=bool(accelerator.sequence_parallel),
        loss_parallel=bool(accelerator.loss_parallel),
    )
    mesh_context.build_meshs(device_type, runtime.world_size)
    mesh_context.dp_rank = mesh_context.device_mesh.get_local_rank("dp")
    mesh_context.tp_rank = mesh_context.device_mesh.get_local_rank("tp")
    mesh_context.cp_rank = mesh_context.device_mesh.get_local_rank("cp")
    mesh_context.ep_rank = (
        mesh_context.fsdp_moe_mesh.get_local_rank("ep")
        if mesh_context.fsdp_moe_mesh is not None
        else 0
    )
    mesh_context.pp_rank = 0
    _prewarm_standard_meshes(mesh_context)
    fsdp_enabled = (
        dp_shard_size > 1
        or dp_replicate_size > 1
        or config.fsdp_config.edp_shard_size > 1
    )
    setup = DistributedSetup(
        mesh_context=mesh_context,
        strategy_config=config.fsdp_config if fsdp_enabled else None,
        plan_overrides=config.plan_overrides,
        low_precision_config=getattr(config.training, "low_precision", None),
        fp32_main_params=config.optimizer.fp32_main_params,
    )
    return _DryRunParallelContext(setup, None, None)


def build_distributed_setup(
        config: TrainerConfig,
        runtime: DryRunRuntime,
        device_type: str,
) -> _DryRunParallelContext:
    """Build root-derived stage meshes without changing public MeshContext semantics."""
    accelerator = config.accelerator
    if max(1, accelerator.pp_size) == 1:
        return _build_standard_distributed_setup(config, runtime, device_type)
    tp_size = max(1, accelerator.tp_size)
    cp_size = max(1, accelerator.cp_size)
    pp_size = max(1, accelerator.pp_size)
    ep_size = max(1, accelerator.ep_size)
    dp_size = runtime.world_size // (tp_size * cp_size * pp_size)
    dp_shard_size = config.fsdp_config.dp_shard_size
    dp_replicate_size = dp_size * cp_size // dp_shard_size
    mesh_context = MeshContext(
        dp_size=dp_size,
        dp_replicate_size=dp_replicate_size,
        dp_shard_size=dp_shard_size,
        edp_shard_size=config.fsdp_config.edp_shard_size,
        tp_size=tp_size,
        cp_size=cp_size,
        pp_size=pp_size,
        ep_size=ep_size,
        sequence_parallel=bool(accelerator.sequence_parallel),
        loss_parallel=bool(accelerator.loss_parallel),
    )
    root_mesh, pp_mesh, stage_mesh = _build_pipeline_meshes(
        device_type, pp_size, dp_size, cp_size, tp_size
    )
    mesh_context.build_meshs(device_type, runtime.world_size)
    if tuple(mesh_context.device_mesh.rank_list) != tuple(stage_mesh.rank_list):
        raise ValueError("Pipeline root mesh and MeshContext stage ranks do not match")
    mesh_context.dp_rank = stage_mesh.get_local_rank("dp")
    mesh_context.tp_rank = stage_mesh.get_local_rank("tp")
    mesh_context.cp_rank = stage_mesh.get_local_rank("cp")
    mesh_context.pp_rank = pp_mesh.get_local_rank()
    mesh_context.ep_rank = 0
    _prewarm_pipeline_meshes(root_mesh, mesh_context, pp_mesh)
    fsdp_enabled = dp_shard_size > 1 or dp_replicate_size > 1
    setup = DistributedSetup(
        mesh_context=mesh_context,
        strategy_config=config.fsdp_config if fsdp_enabled else None,
        plan_overrides=config.plan_overrides,
        low_precision_config=getattr(config.training, "low_precision", None),
        fp32_main_params=config.optimizer.fp32_main_params,
    )
    return _DryRunParallelContext(setup, root_mesh, pp_mesh)


def build_pipeline_chunks(
        config: TrainerConfig,
        model: nn.Module,
        loss_fn: nn.Module,
        num_micro_batches: int,
        parallel_context: _DryRunParallelContext,
        boundary_dtype: torch.dtype,
        micro_batch_shape: tuple[int, int],
) -> tuple[DryRunPipelineChunk, ...]:
    """Invoke and validate the configured model-specific PP builder."""
    if parallel_context.pp_mesh is None:
        raise ValueError("Pipeline chunk construction requires pp_size > 1")
    pp_rank = parallel_context.pp_mesh.get_local_rank()
    # ParallelBatch has already sharded sequence tensors across CP ranks. The
    # boundary contract is global, so restore its global sequence dimension
    # before the planner resolves the stage-local boundary exactly once.
    global_sequence_length = (
        micro_batch_shape[1] * parallel_context.setup.mesh_context.cp_size
    )
    chunks = config.dry_run.pipeline_stage_builder.build(
        full_model=model,
        loss_fn=loss_fn,
        pp_rank=pp_rank,
        pp_size=config.accelerator.pp_size,
        num_micro_batches=num_micro_batches,
        pp_vpp=config.accelerator.pp_vpp,
        pp_layer_split=config.accelerator.pp_layer_split,
        boundary_batch_size=micro_batch_shape[0],
        sequence_length=global_sequence_length,
        boundary_dtype=boundary_dtype,
    )
    if not isinstance(chunks, tuple) or not all(isinstance(chunk, DryRunPipelineChunk) for chunk in chunks):
        raise ValueError(
            "dry_run.pipeline_stage_builder must return a tuple of "
            f"DryRunPipelineChunk objects, got {type(chunks).__name__}"
        )
    if len(chunks) != config.accelerator.pp_vpp:
        raise ValueError(
            "Dry-run pipeline builder must return one chunk per virtual stage, "
            f"got {len(chunks)} for pp_vpp={config.accelerator.pp_vpp}"
        )
    if len({chunk.hidden_size for chunk in chunks}) != 1:
        raise ValueError("All dry-run pipeline chunks must use the same hidden_size")
    expected_indices = tuple(
        pp_rank + virtual_index * config.accelerator.pp_size
        for virtual_index in range(config.accelerator.pp_vpp)
    )
    actual_indices = tuple(chunk.stage_index for chunk in chunks)
    if actual_indices != expected_indices:
        raise ValueError(
            "Dry-run pipeline builder returned unexpected stage indices: "
            f"expected {expected_indices}, got {actual_indices}"
        )
    return chunks


def _plan_and_apply_pipeline_chunk(
        chunk: DryRunPipelineChunk,
        setup: DistributedSetup,
        planner: Any,
        validate_placement: bool,
) -> _DryRunStageShardingResult:
    """Apply the existing TP/CP planner to one already-split stage."""
    mesh = setup.mesh_context
    if mesh.tp_size <= 1 and mesh.cp_size <= 1:
        return _DryRunStageShardingResult(chunk.module, None, None)
    plan = planner.plan(
        chunk.module,
        mesh.device_mesh,
        tp_size=mesh.tp_size,
        cp_size=mesh.cp_size,
        ep_size=1,
        sequence_parallel=False,
        loss_parallel=False,
    )
    model, source_shard_info = apply_sharding_plan(
        chunk.module, plan, mesh, validate_mode=validate_placement
    )
    return _DryRunStageShardingResult(model, source_shard_info, plan)


def _active_plan_mesh(mesh: DeviceMesh, plan: ShardingPlan) -> DeviceMesh:
    """Return the stage submesh matching the planner's active dimensions."""
    active_names = tuple(plan.mesh_dim_names)
    if not active_names or tuple(mesh.mesh_dim_names or ()) == active_names:
        return mesh
    selector: str | tuple[str, ...] = active_names[0]
    if len(active_names) > 1:
        selector = active_names
    return mesh[selector]


def _resolve_boundary_metadata(
        leaves: tuple[DryRunBoundaryLeaf, ...],
        sharding: _DryRunStageShardingResult,
        mesh: MeshContext,
        contract_field: str,
) -> list[list[Any]]:
    """Resolve ordered boundary leaves into PipelineStage metadata."""
    if contract_field not in ("in_dst", "out_dst"):
        raise ValueError(f"Unknown boundary contract field: {contract_field}")
    if mesh.tp_size <= 1 and mesh.cp_size <= 1:
        return [[leaf.global_shape, leaf.dtype, leaf.requires_grad] for leaf in leaves]
    if sharding.plan is None:
        raise ValueError("TP/CP pipeline boundary resolution requires a ShardingPlan")
    metadata = []
    for leaf_index, leaf in enumerate(leaves):
        if not leaf.anchor_fqn:
            raise ValueError(f"TP/CP pipeline boundary leaf {leaf_index} is missing an anchor FQN")
        active_mesh = _active_plan_mesh(mesh.device_mesh, sharding.plan)
        module_spec = sharding.plan.modules.get(leaf.anchor_fqn)
        if module_spec is None:
            raise ValueError(
                f"TP/CP pipeline boundary anchor {leaf.anchor_fqn!r} is absent from the ShardingPlan"
            )
        declared = getattr(module_spec, contract_field)
        if not declared or leaf.tensor_name not in declared:
            raise ValueError(
                f"TP/CP pipeline boundary anchor {leaf.anchor_fqn!r} has no "
                f"{contract_field}[{leaf.tensor_name!r}] declaration"
            )
        placements = tuple(resolve_placements(declared[leaf.tensor_name], sharding.plan.mesh_dim_names))
        local_shape = tuple(
            compute_local_shape_and_global_offset(leaf.global_shape, active_mesh, placements)
        )
        if leaf.wire_kind == "dtensor":
            layout = _build_layout(active_mesh, placements, len(leaf.global_shape))
            metadata.append([local_shape, leaf.dtype, layout, leaf.requires_grad])
        else:
            metadata.append([local_shape, leaf.dtype, leaf.requires_grad])
    return metadata


def _unit_parameters(unit: tuple[nn.Module, ...]) -> set[nn.Parameter]:
    """Return deduplicated trainable parameters owned by one unit."""
    return {
        parameter
        for module in unit
        for parameter in module.parameters()
        if parameter.requires_grad
    }


def _apply_pipeline_fsdp(
        chunk: DryRunPipelineChunk,
        sharding: _DryRunStageShardingResult,
        fsdp_manager: Any,
        fsdp_device_context: Callable[[], ContextManager[Any]],
) -> None:
    """Manually fully-shard declared units while keeping the stage root plain."""
    if fsdp_manager is None:
        return
    trainable_parameters = {parameter for parameter in chunk.module.parameters() if parameter.requires_grad}
    parameters_by_unit = []
    managed_parameters: set[nn.Parameter] = set()
    for unit_index, unit in enumerate(chunk.fsdp_units):
        unit_parameters = _unit_parameters(unit)
        overlap = managed_parameters.intersection(unit_parameters)
        if overlap:
            raise ValueError(
                f"Dry-run pipeline FSDP unit {unit_index} manages {len(overlap)} duplicate parameters"
            )
        managed_parameters.update(unit_parameters)
        parameters_by_unit.append(unit_parameters)
    missing_parameters = trainable_parameters.difference(managed_parameters)
    unexpected_parameters = managed_parameters.difference(trainable_parameters)
    if missing_parameters or unexpected_parameters:
        raise ValueError(
            "Dry-run pipeline FSDP units must manage every trainable parameter exactly once: "
            f"missing={len(missing_parameters)}, unexpected={len(unexpected_parameters)}"
        )

    metadata_by_parameter = _build_source_shard_info_by_param(
        fsdp_manager,
        chunk.module,
        sharding.source_shard_info,
    )
    replicate_parameters = fsdp_manager._resolve_replicate_params(chunk.module)
    fsdp_mesh = fsdp_manager._build_fsdp_actual_mesh()
    fsdp_kwargs, _ = fsdp_manager._build_fully_shard_kwargs(fsdp_mesh)
    default_source_info = (
        _get_default_source_shard_info(fsdp_manager)
        if metadata_by_parameter is not None
        else None
    )
    wrapped_modules = []
    with fsdp_device_context():
        for unit, unit_parameters in zip(chunk.fsdp_units, parameters_by_unit):
            source_infos = None
            if metadata_by_parameter is not None:
                source_infos = {
                    parameter: metadata_by_parameter.get(parameter, default_source_info)
                    for parameter in unit_parameters
                }
            source_infos_for_shard = _source_infos_for_fully_shard(source_infos)
            unit_replicate_parameters = None
            if replicate_parameters is not None:
                unit_replicate_parameters = unit_parameters.intersection(replicate_parameters)
            fully_shard(
                unit[0] if len(unit) == 1 else list(unit),
                source_shard_infos=source_infos_for_shard,
                replicate_params=unit_replicate_parameters,
                **fsdp_kwargs,
            )
            control_module = unit[0]
            fsdp_manager._configure_source_layout_gradient_scaling(control_module, source_infos)
            wrapped_modules.append(control_module)
    fsdp_manager._configure_prefetch(wrapped_modules)
    if isinstance(chunk.module, HSDPModule):
        raise RuntimeError("Dry-run pipeline stage execution root must remain unwrapped")


def prepare_pipeline_chunks(
        chunks: tuple[DryRunPipelineChunk, ...],
        setup: DistributedSetup,
        planner: Any,
        fsdp_manager: Any,
        validate_placement: bool,
        fsdp_device_context: Callable[[], ContextManager[Any]] = nullcontext,
) -> tuple[_PreparedPipelineChunk, ...]:
    """Apply normal-builder stage-local TP/FSDP and resolve wire metadata."""
    prepared = []
    for chunk in chunks:
        sharding = _plan_and_apply_pipeline_chunk(
            chunk, setup, planner, validate_placement
        )
        if sharding.model is not chunk.module:
            raise RuntimeError("Stage-local sharding must preserve the pipeline module identity")
        _apply_pipeline_fsdp(chunk, sharding, fsdp_manager, fsdp_device_context)
        prepared.append(
            _PreparedPipelineChunk(
                chunk,
                _resolve_boundary_metadata(chunk.input_boundary, sharding, setup.mesh_context, "in_dst"),
                _resolve_boundary_metadata(chunk.output_boundary, sharding, setup.mesh_context, "out_dst"),
            )
        )
    return tuple(prepared)


__all__ = [
    "_DryRunParallelContext",
    "_DryRunStageShardingResult",
    "_PreparedPipelineChunk",
    "_build_pipeline_meshes",
    "_resolve_boundary_metadata",
    "build_distributed_setup",
    "build_pipeline_chunks",
    "prepare_pipeline_chunks",
]
