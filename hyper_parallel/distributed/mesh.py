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
"""Mesh domains and the distributed-setup container for AutoModels.

Split out of the former ``components/distributed/infrastructure.py`` in
stage 7 (05 §15.11 step 3): the mesh topology objects are AutoModels-side
model-construction inputs, while process-group lifecycle helpers moved to
``hyper_parallel.trainer.runtime.distributed`` /
``hyper_parallel.trainer.runtime.logging``.
"""

import inspect
import logging
import math
from dataclasses import dataclass, field
from typing import Any

import torch.distributed as dist

from hyper_parallel import init_device_mesh
from hyper_parallel.models.build_options import (  # pylint: disable=syntax-error
    get_device_type,
)

logger = logging.getLogger(__name__)


def _init_topology_mesh(mesh_kwargs: dict[str, Any]) -> Any:
    """Create a mesh while preserving compatibility with lightweight test fakes."""
    supported_arguments = inspect.signature(init_device_mesh).parameters
    filtered_kwargs = {
        name: value
        for name, value in mesh_kwargs.items()
        if name in supported_arguments
    }
    return init_device_mesh(**filtered_kwargs)


@dataclass
class MeshContext:
    """Runtime topology and the mesh domains consumed by dual-mode Trainer."""
    device_mesh: Any = None
    fsdp_non_moe_mesh: Any = None
    fsdp_moe_mesh: Any = None
    dp_size: int = 1
    dp_replicate_size: int = 1
    dp_shard_size: int = 1
    edp_shard_size: int = 1
    tp_size: int = 1
    cp_size: int = 1
    pp_size: int = 1
    ep_size: int = 1
    dp_rank: int = 0
    tp_rank: int = 0
    cp_rank: int = 0
    pp_rank: int = 0
    ep_rank: int = 0
    sequence_parallel: bool = False
    loss_parallel: bool = False

    @property
    def pp_enabled(self) -> bool:
        """Whether pipeline parallelism is enabled (pp_size > 1)."""
        return self.pp_size > 1

    @property
    def dp_cp_mesh(self) -> Any:
        """DP+CP joint mesh for all-reduce.

        Mirrors the derivation in ``FinetuneRecipe.setup()``: take the DP
        dimension together with CP (if present) and flatten to 1-D.
        """
        if self.device_mesh is None:
            return None
        dim_names = self.device_mesh.mesh_dim_names or ()
        selected_dims = tuple(dim_name for dim_name in ("dp", "cp") if dim_name in dim_names)
        if not selected_dims:
            selected_dims = tuple(
                dim_name
                for dim_name in ("dp_replicate", "dp_shard", "cp")
                if dim_name in dim_names
            )
        if not selected_dims:
            return None
        sub = (
            self.device_mesh[selected_dims[0]]
            if len(selected_dims) == 1
            else self.device_mesh[selected_dims]
        )
        if sub.ndim > 1:
            sub = sub.flatten("dp_cp")
        return sub

    @property
    def cp_mesh(self) -> Any:
        """CP submesh for CP utilities (e.g. shard_batch_for_cp).

        Returns ``None`` when context parallelism is disabled (cp_size == 1)
        or no DeviceMesh is available.
        """
        if self.device_mesh is None:
            return None
        dim_names = self.device_mesh.mesh_dim_names or ()
        if "cp" in dim_names:
            return self.device_mesh["cp"]
        return None

    def build_meshs(self, device_type: str, world_size: int) -> None:
        """Build the device, dense-FSDP and expert-FSDP mesh domains.

        The device mesh preserves the existing (dp, cp, tp) rank order. Dense
        FSDP flattens the DP+CP domain and reshapes it according to the
        configured shard size, while expert FSDP derives an (edp, ep)
        topology from the flattened device domain.

        Args:
            device_type: Device type accepted by init_device_mesh.
            world_size: Number of ranks in the process group.

        Raises:
            ValueError: If the configured topology cannot cover world_size or
                cannot form the requested expert mesh.
        """
        dense_world_size = self.dp_size * self.cp_size * self.tp_size
        expected_world_size = dense_world_size * self.pp_size
        if expected_world_size != world_size:
            raise ValueError(
                "dp_size * cp_size * tp_size * pp_size "
                f"must equal world_size ({expected_world_size} != {world_size})"
            )
        fsdp_data_parallel_size = self.dp_size * self.cp_size
        configured_fsdp_size = self.dp_replicate_size * self.dp_shard_size
        if configured_fsdp_size != fsdp_data_parallel_size:
            raise ValueError(
                "dp_replicate_size * dp_shard_size must equal dp_size * cp_size "
                f"({configured_fsdp_size} != {fsdp_data_parallel_size})"
            )

        init_backend = dist.is_initialized()
        stage_rank_start = 0
        if init_backend and self.pp_size > 1:
            stage_rank_start = (dist.get_rank() // dense_world_size) * dense_world_size
        stage_rank_list = tuple(
            range(stage_rank_start, stage_rank_start + dense_world_size)
        )
        device_mesh_kwargs = {
            "device_type": device_type,
            "mesh_shape": (self.dp_size, self.cp_size, self.tp_size),
            "mesh_dim_names": ("dp", "cp", "tp"),
            "init_backend": init_backend,
        }
        if init_backend:
            device_mesh_kwargs["rank_list"] = stage_rank_list
        self.device_mesh = _init_topology_mesh(device_mesh_kwargs)

        dense_rank_list = (
            tuple(self.device_mesh.rank_list)
            if hasattr(self.device_mesh, "rank_list")
            else None
        )
        dense_mesh_kwargs = {
            "device_type": device_type,
            "mesh_shape": (
                self.dp_replicate_size,
                self.dp_shard_size,
                self.tp_size,
            ),
            "mesh_dim_names": ("fsdp_replicate", "fsdp_shard", "tp"),
            "init_backend": init_backend,
        }
        if init_backend and dense_rank_list is not None:
            dense_mesh_kwargs["rank_list"] = dense_rank_list
        self.fsdp_non_moe_mesh = _init_topology_mesh(dense_mesh_kwargs)
        _validate_dense_tp_rank_layout(self.device_mesh, self.fsdp_non_moe_mesh)

        self.fsdp_moe_mesh = None
        if self.ep_size > 1:
            expert_domain_size = math.prod(self.device_mesh.mesh_shape)
            if expert_domain_size % self.ep_size != 0:
                raise ValueError(
                    f"expert domain size ({expert_domain_size}) must be divisible by "
                    f"ep_size ({self.ep_size})"
                )
            edp_size = expert_domain_size // self.ep_size
            if edp_size % self.edp_shard_size != 0:
                raise ValueError(
                    f"expert data-parallel size ({edp_size}) must be divisible by "
                    f"edp_shard_size ({self.edp_shard_size})"
                )
            edp_replicate_size = edp_size // self.edp_shard_size
            if edp_replicate_size > 1:
                expert_mesh_shape = (edp_replicate_size, self.edp_shard_size, self.ep_size)
                expert_mesh_names = ("edp_replicate", "edp_shard", "ep")
            else:
                expert_mesh_shape = (self.edp_shard_size, self.ep_size)
                expert_mesh_names = ("edp_shard", "ep")
            expert_mesh_kwargs = {
                "device_type": device_type,
                "mesh_shape": expert_mesh_shape,
                "mesh_dim_names": expert_mesh_names,
                "init_backend": init_backend,
            }
            if init_backend and dense_rank_list is not None:
                expert_mesh_kwargs["rank_list"] = dense_rank_list
            self.fsdp_moe_mesh = _init_topology_mesh(expert_mesh_kwargs)


def _validate_dense_tp_rank_layout(device_mesh: Any, fsdp_non_moe_mesh: Any) -> None:
    """Ensure both TP child meshes describe the same groups and local rank."""
    if not hasattr(device_mesh, "rank_list") or not hasattr(fsdp_non_moe_mesh, "rank_list"):
        return
    device_tp_mesh = device_mesh["tp"]
    fsdp_tp_mesh = fsdp_non_moe_mesh["tp"]
    if tuple(device_tp_mesh.rank_list) != tuple(fsdp_tp_mesh.rank_list):
        raise ValueError("device_mesh['tp'] and fsdp_non_moe_mesh['tp'] rank groups differ")
    if device_tp_mesh.get_local_rank() != fsdp_tp_mesh.get_local_rank():
        raise ValueError("device_mesh['tp'] and fsdp_non_moe_mesh['tp'] local ranks differ")


@dataclass
class DistributedSetup:
    """Unified distributed configuration container.

    Following design doc 06 §3.
    """
    mesh_context: MeshContext = field(default_factory=MeshContext)
    strategy_config: Any = None
    pipeline_config: Any = None
    moe_parallel_config: Any = None
    activation_checkpointing: Any = None
    # Normalized ``{match: ModuleShardingSpec}`` mapping (or None) — the
    # Trainer desugars YAML PlanOverride entries via
    # entries_to_plan_overrides (when-filtering against the built mesh
    # included) BEFORE the AutoModels build pipeline runs; AutoModels never
    # sees the raw Trainer YAML DTO list (05 §15.2.6).
    plan_overrides: Any = None
    # Normalized module replacement rules (tuple[ModuleReplacementSpec])
    # desugared Trainer-side via entries_to_module_replacements.
    module_replacements: Any = None
    # TrainingConfig.training.low_precision. Kept on the setup so model
    # construction and sharding consume one resolved policy object.
    low_precision_config: Any = None
    fp32_main_params: bool = False


def _build_device_mesh_from_accelerator(
    accel: Any,
    dp_shard_size: int,
    dp_replicate_size: int,
    world_size: int,
    edp_shard_size: int = 1,
) -> tuple[MeshContext, tuple[str, ...]]:
    """Build the explicit MeshContext domains from accelerator topology."""
    tp_size = max(1, accel.tp_size)
    cp_size = max(1, accel.cp_size)
    pp_size = max(1, accel.pp_size)
    ep_size = max(1, accel.ep_size)

    dp_size = world_size // (tp_size * cp_size * pp_size)
    mesh_context = MeshContext(
        dp_size=dp_size,
        dp_replicate_size=dp_replicate_size,
        dp_shard_size=dp_shard_size,
        edp_shard_size=edp_shard_size,
        tp_size=tp_size,
        cp_size=cp_size,
        pp_size=pp_size,
        ep_size=ep_size,
        sequence_parallel=bool(accel.sequence_parallel),
        loss_parallel=bool(accel.loss_parallel),
    )
    mesh_context.build_meshs(get_device_type(), world_size)
    device_mesh_shape = (
        mesh_context.device_mesh.mesh_shape
        if hasattr(mesh_context.device_mesh, "mesh_shape")
        else None
    )
    dense_shape = (
        mesh_context.fsdp_non_moe_mesh.mesh_shape
        if hasattr(mesh_context.fsdp_non_moe_mesh, "mesh_shape")
        else None
    )
    expert_shape = (
        mesh_context.fsdp_moe_mesh.mesh_shape
        if mesh_context.fsdp_moe_mesh is not None and hasattr(mesh_context.fsdp_moe_mesh, "mesh_shape")
        else None
    )
    logger.info(
        "Built device mesh %s, dense FSDP mesh %s and expert FSDP mesh %s",
        device_mesh_shape,
        dense_shape,
        expert_shape,
    )
    return mesh_context, mesh_context.device_mesh.mesh_dim_names


__all__ = [
    "DistributedSetup",
    "MeshContext",
]
