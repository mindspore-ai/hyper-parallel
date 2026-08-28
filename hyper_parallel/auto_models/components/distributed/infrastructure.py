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
"""Distributed infrastructure — stubs for distributed setup, mesh, init.

Following design doc 06_distributed_infrastructure.md.
"""

from datetime import timedelta
import inspect
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any

import torch.distributed as dist

from hyper_parallel import init_device_mesh
from hyper_parallel.auto_models.components.utils.device import (  # pylint: disable=syntax-error
    get_device_type,
    get_torch_device,
)

logger = logging.getLogger(__name__)

_DATASET_BARRIER_TIMEOUT = timedelta(hours=10)


class OnlineDatasetBarrier:
    """Synchronize long Online mapping builds through a diagnostic Gloo group."""

    def __init__(self, timeout: timedelta = _DATASET_BARRIER_TIMEOUT) -> None:
        """Store the timeout and defer auxiliary group creation."""
        if timeout.total_seconds() <= 0:
            raise ValueError("Online Dataset barrier timeout must be positive")
        self.timeout = timeout
        self._gloo_group: Any = None
        self._gloo_unavailable = False

    def __call__(self) -> None:
        """Wait up to ten hours and identify missing ranks when supported."""
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return

        if self._gloo_unavailable:
            dist.barrier()
            return

        if self._gloo_group is None:
            try:
                self._gloo_group = dist.new_group(
                    backend="gloo",
                    timeout=self.timeout,
                )
            except (RuntimeError, ValueError) as error:
                self._gloo_unavailable = True
                logger.warning(
                    "Online Dataset Gloo group is unavailable; falling back "
                    "to the default process-group barrier: %s",
                    error,
                )
                dist.barrier()
                return

        dist.monitored_barrier(
            group=self._gloo_group,
            timeout=self.timeout,
            wait_all_ranks=True,
        )


def _init_topology_mesh(mesh_kwargs: dict[str, Any]) -> Any:
    """Create a mesh while preserving compatibility with lightweight test fakes."""
    supported_arguments = inspect.signature(init_device_mesh).parameters
    filtered_kwargs = {
        name: value
        for name, value in mesh_kwargs.items()
        if name in supported_arguments
    }
    return init_device_mesh(**filtered_kwargs)


# ── MeshContext ──

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


# ── DistributedSetup (stub) ──

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
    # TrainerConfig.plan_overrides (List[PlanOverride]) — the YAML transport
    # form of plan_overrides (injected fields + contract DSL + when
    # conditions); instantiate_infrastructure desugars it via
    # entries_to_plan_overrides (including when filtering) before building
    # the planner
    plan_overrides: Any = None
    # TrainingConfig.training.low_precision. Kept on the setup so model
    # construction and sharding consume one resolved policy object.
    low_precision_config: Any = None


# ── initialize_distributed (stub) ──

def initialize_distributed(backend: str = "nccl") -> Any:
    """Initialize torch.distributed process group.

    Stub — calls dist.init_process_group if not already initialized.
    Falls back to gloo when the requested backend is unavailable (e.g. CPU).
    Sets the current accelerator device from ``LOCAL_RANK`` before process
    group initialization.
    """
    device_type = get_device_type()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if device_type in ("cuda", "npu"):
        get_torch_device().set_device(local_rank)

    if not dist.is_initialized():
        effective_backend = backend
        if backend == "nccl" and device_type == "npu":
            effective_backend = "hccl"
            logger.info("Using HCCL instead of NCCL on the NPU backend.")
        elif backend == "nccl" and device_type == "cpu":
            effective_backend = "gloo"
            logger.warning(
                "CUDA not available; falling back to '%s' backend for distributed.",
                effective_backend,
            )
        dist.init_process_group(backend=effective_backend)

    return dist


# ── create_distributed_setup_from_config (stub) ──

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


def create_distributed_setup_from_config(cfg: Any) -> DistributedSetup:
    """Create DistributedSetup and build the configured mesh domains."""
    accel = cfg.accelerator if cfg is not None and hasattr(cfg, "accelerator") else None
    if accel is None:
        return DistributedSetup(mesh_context=MeshContext())

    fsdp_config = cfg.fsdp_config
    dp_shard_size = max(1, fsdp_config.dp_shard_size)
    edp_shard_size = max(1, fsdp_config.edp_shard_size)
    tp_size = max(1, accel.tp_size)
    cp_size = max(1, accel.cp_size)
    pp_size = max(1, accel.pp_size)
    ep_size = max(1, accel.ep_size)

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    non_dp_size = tp_size * cp_size * pp_size
    if world_size % non_dp_size != 0:
        raise ValueError(
            f"world_size {world_size} is not divisible by non-DP size {non_dp_size}"
        )
    dp_size = world_size // non_dp_size
    fsdp_data_parallel_size = dp_size * cp_size
    if fsdp_data_parallel_size % dp_shard_size != 0:
        raise ValueError(
            "DP+CP size "
            f"{fsdp_data_parallel_size} is not divisible by FSDP shard size "
            f"{dp_shard_size}"
        )
    dp_replicate_size = fsdp_data_parallel_size // dp_shard_size
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
    if dist.is_initialized():
        mesh_context, _ = _build_device_mesh_from_accelerator(
            accel,
            dp_shard_size,
            dp_replicate_size,
            world_size,
            edp_shard_size,
        )

    def _local_rank(dim: str) -> int:
        if (
            mesh_context.device_mesh is None
            or dim not in (mesh_context.device_mesh.mesh_dim_names or ())
        ):
            return 0
        return mesh_context.device_mesh.get_local_rank(dim)

    device_mesh_dim_names = (
        mesh_context.device_mesh.mesh_dim_names
        if mesh_context.device_mesh is not None
        else ()
    )
    if "dp" in device_mesh_dim_names:
        mesh_context.dp_rank = _local_rank("dp")
    else:
        dp_shard_rank = _local_rank("dp_shard")
        dp_replicate_rank = _local_rank("dp_replicate")
        mesh_context.dp_rank = dp_replicate_rank * dp_shard_size + dp_shard_rank
    mesh_context.tp_rank = _local_rank("tp")
    mesh_context.cp_rank = _local_rank("cp")
    if mesh_context.pp_size > 1 and dist.is_initialized():
        stage_world_size = mesh_context.dp_size * mesh_context.cp_size * mesh_context.tp_size
        mesh_context.pp_rank = dist.get_rank() // stage_world_size
    else:
        mesh_context.pp_rank = 0
    mesh_context.ep_rank = (
        mesh_context.fsdp_moe_mesh.get_local_rank("ep")
        if mesh_context.fsdp_moe_mesh is not None
        else 0
    )

    fsdp_enabled = dist.is_initialized() and (
        dp_shard_size > 1 or dp_replicate_size > 1 or edp_shard_size > 1
    )
    strategy_config = fsdp_config if fsdp_enabled else None
    return DistributedSetup(
        mesh_context=mesh_context,
        strategy_config=strategy_config,
        plan_overrides=getattr(cfg, "plan_overrides", None),
        low_precision_config=getattr(
            getattr(cfg, "training", None),
            "low_precision",
            None,
        ),
    )


# ── Helper functions (from design doc §7.1) ──

def _is_rank_0() -> bool:
    """True if global rank 0 (or distributed not initialized)."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def setup_logging() -> None:
    """Setup logging with rank filter (stub)."""
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.INFO,
    )


def apply_cache_compatibility_patches() -> None:
    """Apply cache compatibility patches for transformers (stub)."""


def destroy_process_group() -> None:
    """Destroy the process group and clear caches tied to the distributed runtime."""
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    finally:
        # Import here to avoid adding the cache modules to the auto-model import graph during package initialization.
        from hyper_parallel.auto_models.components.distributed.cp_utils import (  # pylint: disable=C0415
            _HYBRID_MESH_CACHE,
        )
        from hyper_parallel.core.dtensor.device_mesh import _DEVICE_MESH_MAP  # pylint: disable=C0415
        from hyper_parallel.core.dtensor.dtensor import _LAYOUT_CACHE  # pylint: disable=C0415
        from hyper_parallel.core.dtensor.tensor_redistribution import _tensor_redistribution  # pylint: disable=C0415
        from hyper_parallel.core.fully_shard.hsdp_param import _GROUP_INFO_CACHE  # pylint: disable=C0415
        from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS  # pylint: disable=C0415

        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()
        _GROUP_INFO_CACHE.clear()
        _HYBRID_MESH_CACHE.clear()
        _tensor_redistribution._transform_cache.clear()  # pylint: disable=protected-access
        _tensor_redistribution.is_init = False
        _tensor_redistribution.rank_id = None
