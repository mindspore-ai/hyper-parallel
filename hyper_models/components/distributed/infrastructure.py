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

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.distributed as dist

from hyper_models.components.distributed.config import FSDP2Config

logger = logging.getLogger(__name__)


# ── MeshContext (stub) ──

@dataclass
class MeshContext:
    """Runtime topology — read from DeviceMesh.

    Following design doc 06 §2.
    Stub — returns defaults when no mesh is available.
    """
    device_mesh: Any = None
    dp_size: int = 1
    dp_replicate_size: int = 1
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
        return self.pp_size > 1

    @property
    def dp_cp_mesh(self):
        """DP+CP joint mesh for all-reduce.

        Mirrors the derivation in ``FinetuneRecipe.setup()``: take the DP
        dimension together with CP (if present) and flatten to 1-D.
        """
        mesh = self.device_mesh
        if mesh is None:
            return None
        dim_names = mesh.mesh_dim_names
        if "cp" in dim_names and "dp_shard_cp" in dim_names:
            sub = mesh[("dp_shard_cp", "cp")]
        elif "dp_shard_cp" in dim_names:
            sub = mesh["dp_shard_cp"]
        elif "dp" in dim_names:
            sub = mesh[("dp", "cp")] if "cp" in dim_names else mesh["dp"]
        elif "dp_replicate" in dim_names:
            sub = mesh[("dp_replicate", "cp")] if "cp" in dim_names else mesh["dp_replicate"]
        else:
            sub = mesh
        if sub.ndim > 1:
            sub = sub._flatten("dp_cp")
        return sub

    @property
    def cp_mesh(self):
        """CP submesh for CP utilities (e.g. shard_batch_for_cp).

        Returns ``None`` when context parallelism is disabled (cp_size == 1)
        or no DeviceMesh is available.
        """
        if self.device_mesh is None:
            return None
        dim_names = self.device_mesh.mesh_dim_names
        if "cp" in dim_names:
            return self.device_mesh["cp"]
        return None


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


# ── initialize_distributed (stub) ──

def initialize_distributed(backend: str = "nccl") -> Any:
    """Initialize torch.distributed process group.

    Stub — calls dist.init_process_group if not already initialized.
    Falls back to gloo when the requested backend is unavailable (e.g. CPU).
    Sets the current CUDA device from LOCAL_RANK when CUDA is available.
    """
    if not dist.is_initialized():
        effective_backend = backend
        if backend == "nccl" and not torch.cuda.is_available():
            effective_backend = "gloo"
            logger.warning(
                "CUDA not available; falling back to '%s' backend for distributed.",
                effective_backend,
            )
        dist.init_process_group(backend=effective_backend)

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

    return dist


# ── create_distributed_setup_from_config (stub) ──

def _build_device_mesh_from_accelerator(
    accel: Any,
    world_size: int,
) -> tuple[Any, tuple[str, ...]] | tuple[None, tuple[()]]:
    """Build a hyper_parallel DeviceMesh from accelerator topology.

    Returns (device_mesh, dim_names) when the topology matches world_size,
    otherwise (None, ()).
    """
    dp_shard_size = max(1, getattr(accel, "dp_shard_size", 1))
    dp_replicate_size = max(1, getattr(accel, "dp_replicate_size", 1))
    tp_size = max(1, getattr(accel, "tp_size", 1))
    cp_size = max(1, getattr(accel, "cp_size", 1))
    pp_size = max(1, getattr(accel, "pp_size", 1))
    ep_size = max(1, getattr(accel, "ep_size", 1))

    requested_size = dp_shard_size * dp_replicate_size * tp_size * cp_size * pp_size * ep_size
    if requested_size != world_size:
        logger.warning(
            "Accelerator topology (%d ranks requested) does not match world_size (%d). "
            "Using stub MeshContext without a DeviceMesh.",
            requested_size, world_size,
        )
        return None, ()

    mesh_dims = []
    # Order matters for FSDP2/HSDP conventions: replicate -> shard -> cp -> tp -> pp.
    if dp_replicate_size > 1:
        mesh_dims.append(("dp_replicate", dp_replicate_size))
    if dp_shard_size > 1:
        mesh_dims.append(("dp_shard_cp", dp_shard_size))
    if cp_size > 1:
        mesh_dims.append(("cp", cp_size))
    if tp_size > 1:
        mesh_dims.append(("tp", tp_size))
    if pp_size > 1:
        mesh_dims.append(("pp", pp_size))
    if ep_size > 1:
        mesh_dims.append(("ep", ep_size))

    if not mesh_dims:
        return None, ()

    dim_names, mesh_shape = zip(*mesh_dims)
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    from hyper_parallel import init_device_mesh
    device_mesh = init_device_mesh(
        device_type=device_type,
        mesh_shape=mesh_shape,
        mesh_dim_names=dim_names,
    )
    logger.info(
        "Built DeviceMesh %s on device %s for world_size=%d",
        mesh_shape, device_type, world_size,
    )
    return device_mesh, dim_names


def create_distributed_setup_from_config(cfg: Any) -> DistributedSetup:
    """Create DistributedSetup from config.

    Stub — returns a default DistributedSetup with MeshContext.
    When running under torchrun (dist initialized and world_size > 1) and the
    accelerator topology multiplies to world_size, builds a real DeviceMesh so
    that TP/CP/DP paths are exercised.
    """
    accel = getattr(cfg, "accelerator", None)
    if accel is None:
        return DistributedSetup(mesh_context=MeshContext())

    dp_shard_size = max(1, getattr(accel, "dp_shard_size", 1))
    dp_replicate_size = max(1, getattr(accel, "dp_replicate_size", 1))
    tp_size = max(1, getattr(accel, "tp_size", 1))
    cp_size = max(1, getattr(accel, "cp_size", 1))
    pp_size = max(1, getattr(accel, "pp_size", 1))
    ep_size = max(1, getattr(accel, "ep_size", 1))

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device_mesh = None
    dim_names = ()
    if dist.is_initialized() and world_size > 1:
        try:
            device_mesh, dim_names = _build_device_mesh_from_accelerator(accel, world_size)
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "Failed to build DeviceMesh from accelerator config: %s. "
                "Falling back to stub MeshContext.",
                exc,
            )
            device_mesh = None
            dim_names = ()

    def _local_rank(dim: str) -> int:
        return device_mesh.get_local_rank(dim) if dim in dim_names else 0

    mesh_ctx = MeshContext(
        device_mesh=device_mesh,
        dp_size=dp_shard_size,
        dp_replicate_size=dp_replicate_size,
        tp_size=tp_size,
        cp_size=cp_size,
        pp_size=pp_size,
        ep_size=ep_size,
        dp_rank=_local_rank("dp_shard_cp"),
        tp_rank=_local_rank("tp"),
        cp_rank=_local_rank("cp"),
        pp_rank=_local_rank("pp"),
        ep_rank=_local_rank("ep"),
        sequence_parallel=bool(getattr(accel, "sequence_parallel", False)),
        loss_parallel=bool(getattr(accel, "loss_parallel", False)),
    )

    # Default to FSDP2 only when running multi-card distributed training.
    # Single-card (world_size == 1) should not instantiate FSDP2Manager,
    # otherwise every single-card run would see the FSDP2 stub warning.
    strategy_config = FSDP2Config() if (dist.is_initialized() and world_size > 1) else None
    return DistributedSetup(
        mesh_context=mesh_ctx,
        strategy_config=strategy_config,
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
    pass


def destroy_process_group() -> None:
    """Destroy the process group."""
    if dist.is_initialized():
        dist.destroy_process_group()
