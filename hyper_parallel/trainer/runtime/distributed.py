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

__all__ = ["all_gather", "all_reduce"]

import logging
import os
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Union

import torch
from torch import distributed as dist

from hyper_parallel.distributed.mesh import (
    DistributedSetup,
    MeshContext,
    ParallelTopologySpec,
    resolve_parallel_layout,
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
    reduce_op = reduce_ops.get(op)
    if reduce_op is None:
        raise ValueError("op must be one of: mean, sum, max, min")
    dist.all_reduce(data, op=reduce_op, group=group)
    if op == "mean":  # ReduceOp.AVG is not supported by the NPU backend
        data /= dist.get_world_size(group=group)

    if data.numel() == 1:
        return data.item()
    return data.tolist()


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


def _populate_mesh_ranks(mesh_context: MeshContext, dp_shard_size: int) -> None:
    """Populate local mesh ranks after device-mesh construction."""
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


def create_distributed_setup_from_config(cfg: Any) -> DistributedSetup:
    """Create DistributedSetup and build the configured mesh domains."""
    accel = cfg.accelerator if cfg is not None and hasattr(cfg, "accelerator") else None
    if accel is None:
        return DistributedSetup(mesh_context=MeshContext())

    topology_spec = ParallelTopologySpec.from_config(cfg)
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    layout = resolve_parallel_layout(topology_spec, world_size)
    mesh_context = MeshContext.from_layout(layout)
    if dist.is_initialized():
        mesh_context.build_meshs(get_device_type(), world_size)

    _populate_mesh_ranks(mesh_context, topology_spec.dp_shard_size)

    return DistributedSetup.from_config(
        cfg,
        mesh_context=mesh_context,
        strategy_enabled=dist.is_initialized(),
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
        from hyper_parallel.core.utils.communication import EXISTING_COMM_GROUPS  # pylint: disable=C0415
        from hyper_parallel.core.pipeline_parallel._p2p import (  # pylint: disable=C0415
            _P2P_MULTI_STREAM_GROUPS,
        )

        EXISTING_COMM_GROUPS.clear()
        _P2P_MULTI_STREAM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()
        _HYBRID_MESH_CACHE.clear()
        _tensor_redistribution._transform_cache.clear()  # pylint: disable=protected-access
        _tensor_redistribution.is_init = False
        _tensor_redistribution.rank_id = None
