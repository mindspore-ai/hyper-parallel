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
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.distributed as dist

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
        """DP+CP joint mesh for all-reduce."""
        return self.device_mesh


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
    """
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return dist


# ── create_distributed_setup_from_config (stub) ──

def create_distributed_setup_from_config(cfg: Any) -> DistributedSetup:
    """Create DistributedSetup from config.

    Stub — returns a default DistributedSetup with MeshContext.
    Full implementation will build DeviceMesh from cfg.accelerator fields.
    """
    accel = getattr(cfg, "accelerator", None)
    if accel is None:
        return DistributedSetup(mesh_context=MeshContext())

    mesh_ctx = MeshContext(
        dp_size=max(1, getattr(accel, "dp_shard_size", 1)),
        dp_replicate_size=max(1, getattr(accel, "dp_replicate_size", 1)),
        tp_size=max(1, getattr(accel, "tp_size", 1)),
        cp_size=max(1, getattr(accel, "cp_size", 1)),
        pp_size=max(1, getattr(accel, "pp_size", 1)),
        ep_size=max(1, getattr(accel, "ep_size", 1)),
        sequence_parallel=bool(getattr(accel, "sequence_parallel", False)),
        loss_parallel=bool(getattr(accel, "loss_parallel", False)),
    )
    return DistributedSetup(mesh_context=mesh_ctx)


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