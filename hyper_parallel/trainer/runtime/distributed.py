# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Collective and process-group lifecycle helpers for the trainer runtime.

``get_world_size_safe`` / ``get_global_rank_safe`` / ``get_local_rank_safe``
merged in from the former ``components/distributed/init_utils.py`` in stage 7
(05 §15.11 step 3); ``initialize_distributed`` /
``create_distributed_setup_from_config`` / ``destroy_process_group`` come from
the former ``components/distributed/infrastructure.py`` (05 §15.11 step 3,
plan §1195-1196). The mesh objects themselves are AutoModels-side in
``hyper_parallel.distributed.mesh``.
"""

import logging
import os
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Union

import torch
from torch import distributed as dist

from hyper_parallel.distributed.mesh import (
    DistributedSetup,
    MeshContext,
    _build_device_mesh_from_accelerator,
)
from hyper_parallel.trainer.runtime.device import get_device_type, get_torch_device

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from torch.distributed import ProcessGroup


def all_gather(tensor: "torch.Tensor", world_size: int) -> "torch.Tensor":
    """
    Gathers the tensor from all ranks and concats them along the first dim.
    """
    output_tensor = torch.empty(world_size * tensor.numel(), dtype=tensor.dtype, device=get_device_type())
    dist.all_gather_into_tensor(output_tensor, tensor)
    return output_tensor.view(-1, *tensor.size()[1:])


def all_reduce(
    data: Union[int, float, List[Union[int, float]], "torch.Tensor"],
    op: Literal["mean", "sum", "max", "min"] = "mean",
    group: Optional["ProcessGroup"] = None,
) -> Union[int, float, List[Union[int, float]]]:
    """
    Performs all reduce in the given process group.
    """
    if not dist.is_initialized():
        raise RuntimeError("Distributed environment is not initialized.")

    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float, device=get_device_type())

    reduce_ops = {
        "mean": dist.ReduceOp.SUM,
        "sum": dist.ReduceOp.SUM,
        "max": dist.ReduceOp.MAX,
        "min": dist.ReduceOp.MIN,
    }
    dist.all_reduce(data, op=reduce_ops[op], group=group)
    if op == "mean":  # ReduceOp.AVG is not supported by the NPU backend
        data /= dist.get_world_size(group=group)

    if data.numel() == 1:
        return data.item()
    return data.tolist()


__all__ = ["all_gather", "all_reduce"]
def get_world_size_safe() -> int:
    """Return dist.get_world_size() if initialized, else 1."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_global_rank_safe() -> int:
    """Return dist.get_rank() if initialized, else 0."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_local_rank_safe() -> int:
    """Return dist.get_node_local_rank() if initialized, else 0."""
    if dist.is_initialized():
        return dist.get_node_local_rank()
    return 0
def initialize_distributed(backend: str = "nccl") -> Any:
    """Initialize torch.distributed process group.

    Calls dist.init_process_group if not already initialized.
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
        # plan_overrides/module_replacements stay unset here: the Trainer
        # desugars the raw YAML entries onto the setup (05 §15.2.6) before
        # model construction.
        low_precision_config=getattr(
            getattr(cfg, "training", None),
            "low_precision",
            None,
        ),
        fp32_main_params=cfg.optimizer.fp32_main_params,
    )


def destroy_process_group() -> None:
    """Destroy the process group and clear caches tied to the distributed runtime."""
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    finally:
        # Import here to avoid adding the cache modules to the auto-model import graph during package initialization.
        from hyper_parallel.distributed.context_parallel.collectives import (  # pylint: disable=C0415
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
