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
"""Build shared RuntimeConfigs for HyperMegaGate Route and RouteGrad."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist

from hyper_parallel.core.multicore.profiler.profiling import (
    _PreparedMegaKernelRuntime,
    _prepare_mega_kernel_runtime_config,
)
from hyper_parallel.core.multicore.profiler.profiler import _enable_runtime_config_tensor
from hyper_parallel.core.multicore.scheduler.config import (
    EVENT_INVALID_ID,
    INVALID_PROFILE_OWNER_ID,
    TaskAiCoreType,
    TaskDescC,
    TaskType,
    RuntimeConfigC,
    NUM_WORKERS_VECTOR,
)
from hyper_parallel.core.multicore.scheduler.runtime import allocate_runtime_config

from .profiling import (
    MEGA_GATE_ROUTE_GRAD_K1_STAGE_NAMES,
    MEGA_GATE_ROUTE_GRAD_STAGE_NAMES,
    MEGA_GATE_ROUTE_STAGE_NAMES,
    _configure_mega_gate_profile_metadata,
)


MEGA_GATE_AIV_WORKER_CAPACITY = NUM_WORKERS_VECTOR
MEGA_GATE_ROUTE_TASK_TYPES = (
    TaskType.TASK_GATE_SOFTPLUS,
    TaskType.TASK_GATE_SQRT,
    TaskType.TASK_GATE_ADD_BIAS,
    TaskType.TASK_GATE_TOPK,
    TaskType.TASK_GATE_GATHER,
    TaskType.TASK_GATE_REDUCE_SUM,
    TaskType.TASK_GATE_ADD_EPSILON,
    TaskType.TASK_GATE_DIV,
    TaskType.TASK_GATE_MUL_SCALE,
    TaskType.TASK_GATE_CAST_INDEX,
)
MEGA_GATE_ROUTE_GRAD_TASK_TYPES = (
    TaskType.TASK_GATE_GRAD_MULS_SCALE,
    TaskType.TASK_GATE_GRAD_BROADCAST_DENOMINATOR,
    TaskType.TASK_GATE_GRAD_NEG,
    TaskType.TASK_GATE_GRAD_DIV_SELECTED,
    TaskType.TASK_GATE_GRAD_DIV_SELECTED_RATIO,
    TaskType.TASK_GATE_GRAD_MUL_CROSS,
    TaskType.TASK_GATE_GRAD_DIV_DIRECT,
    TaskType.TASK_GATE_GRAD_REDUCE_SUM,
    TaskType.TASK_GATE_GRAD_BROADCAST_ROW_SUM,
    TaskType.TASK_GATE_GRAD_ADD_SELECTED,
    TaskType.TASK_GATE_GRAD_ZEROS,
)
MEGA_GATE_ROUTE_GRAD_K1_TASK_TYPES = (
    TaskType.TASK_GATE_GRAD_MULS_SCALE,
    TaskType.TASK_GATE_GRAD_ZEROS,
)


@dataclass(frozen=True)
class MegaGatePlan:
    """Capacity-bounded RuntimeConfig shared by all input shapes on one NPU."""

    route_runtime: _PreparedMegaKernelRuntime
    route_grad_runtime: _PreparedMegaKernelRuntime
    route_grad_k1_runtime: _PreparedMegaKernelRuntime
    vision_mask_placeholder: torch.Tensor


def _build_pipeline_runtime_config(
    task_types: tuple[TaskType, ...],
    stage_names: tuple[str, ...],
    *,
    kernel_name: str,
) -> RuntimeConfigC:
    """Create descriptors broadcast to every active token-row worker."""
    if len(task_types) != len(stage_names):
        raise RuntimeError("MegaGate task types and profile stages must have equal length")
    runtime_config = allocate_runtime_config(len(task_types))
    runtime_config.task_num = len(task_types)
    runtime_config.num_workers = MEGA_GATE_AIV_WORKER_CAPACITY
    runtime_config.task_index_num[1] = len(task_types)
    for task_id, task_type in enumerate(task_types):
        task = TaskDescC()
        task.task_type = task_type
        task.task_aicore_type = TaskAiCoreType.TASK_AICORE_VECTOR
        task.trigger_event = EVENT_INVALID_ID
        task.dependent_event = EVENT_INVALID_ID
        task.task_index = 0
        task.task_split_num = MEGA_GATE_AIV_WORKER_CAPACITY
        task.tiling_data_offset = task_id
        task.profile_owner_id = INVALID_PROFILE_OWNER_ID
        runtime_config.all_tasks[task_id] = task
        runtime_config.vector_task_indices[task_id] = task_id
    _configure_mega_gate_profile_metadata(runtime_config, stage_names, kernel_name)
    return runtime_config


def _tensor_from_bytes(data: bytes, device: torch.device) -> torch.Tensor:
    """Copy an immutable Host runtime image to one NPU device."""
    array = np.frombuffer(bytearray(data), dtype=np.uint8).copy()
    return torch.from_numpy(array).to(device=device, dtype=torch.uint8)


def build_mega_gate_plan(device: torch.device) -> MegaGatePlan:
    """Build an eventless Route runtime for one NPU device.

    Args:
        device: NPU device receiving the serialized RuntimeConfig.

    Returns:
        Immutable plan with normal and profiled broadcast stage descriptors.
    """
    if device.type != "npu":
        raise ValueError(f"MegaGate plan requires an NPU device, got {device}")
    device_id = device.index
    if device_id is None:
        device_id = torch.npu.current_device()
    rank = dist.get_rank() if dist.is_initialized() else 0

    def prepare(runtime_config: RuntimeConfigC) -> _PreparedMegaKernelRuntime:
        """Serialize one normal/profiled descriptor pair on the target NPU."""
        return _prepare_mega_kernel_runtime_config(
            runtime_config,
            tensor_factory=lambda data: _tensor_from_bytes(data, device),
            profile_tensor_factory=_enable_runtime_config_tensor,
            rank=rank,
            device=device,
            device_id=device_id,
        )

    return MegaGatePlan(
        route_runtime=prepare(_build_pipeline_runtime_config(
            MEGA_GATE_ROUTE_TASK_TYPES,
            MEGA_GATE_ROUTE_STAGE_NAMES,
            kernel_name="HyperMegaGateRoute",
        )),
        route_grad_runtime=prepare(_build_pipeline_runtime_config(
            MEGA_GATE_ROUTE_GRAD_TASK_TYPES,
            MEGA_GATE_ROUTE_GRAD_STAGE_NAMES,
            kernel_name="HyperMegaGateRouteGrad",
        )),
        route_grad_k1_runtime=prepare(_build_pipeline_runtime_config(
            MEGA_GATE_ROUTE_GRAD_K1_TASK_TYPES,
            MEGA_GATE_ROUTE_GRAD_K1_STAGE_NAMES,
            kernel_name="HyperMegaGateRouteGrad",
        )),
        vision_mask_placeholder=torch.zeros((1,), device=device, dtype=torch.bool),
    )
