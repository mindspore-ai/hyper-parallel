# Copyright 2026 Huawei Technologies Co., Ltd.
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
"""Shape-bound RuntimeConfig plan for HyperMegaMhcGrad."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from hyper_parallel.core.multicore.profiler.profiling import (
    _PreparedMegaKernelRuntime,
    _ProfileSpec,
    _apply_mega_kernel_profile_graph,
    _prepare_mega_kernel_runtime_config,
)
from hyper_parallel.core.multicore.profiler.profiler import _enable_runtime_config_tensor
from hyper_parallel.core.multicore.scheduler.builder import build_runtime_config

from .graph import (
    DEFAULT_GRAD_TOKEN_TILE,
    build_mega_mhc_grad_graph,
    order_vector_tasks_by_stage,
    resolve_grad_token_tile,
)


@dataclass(frozen=True)
class MegaMhcGradPlan:
    """Own one backward schedule and its reusable event counters."""

    token_count: int
    hidden_size: int
    num_cube_cores: int
    num_vector_cores: int
    token_tile: int
    runtime: _PreparedMegaKernelRuntime
    event_counters: torch.Tensor


def _tensor_from_bytes(data: bytes, device: Any) -> torch.Tensor:
    """Copy serialized scheduling data into one NPU byte tensor."""
    array = np.frombuffer(bytearray(data), dtype=np.uint8).copy()
    return torch.from_numpy(array).to(device=device, dtype=torch.uint8)


def build_mega_mhc_grad_plan(
    token_count: int,
    hidden_size: int,
    device: Any,
    num_cube_cores: int,
    num_vector_cores: int,
    token_tile: int = DEFAULT_GRAD_TOKEN_TILE,
) -> MegaMhcGradPlan:
    """Build and materialize the token-tiled fused backward schedule.

    Args:
        token_count: Total number of flattened tokens.
        hidden_size: Hidden dimension of every token.
        device: NPU device used to materialize runtime tensors.
        num_cube_cores: Number of physical AIC cores.
        num_vector_cores: Number of physical AIV cores.
        token_tile: Requested AIV token tile.

    Returns:
        Reusable backward plan bound to the requested shape and device.
    """
    token_tile = resolve_grad_token_tile(token_count, token_tile)
    graph, topology = build_mega_mhc_grad_graph(
        token_count,
        hidden_size,
        token_tile,
        num_vector_cores=num_vector_cores,
    )
    runtime_config = build_runtime_config(
        graph, topology, rank_id=0, num_cube_cores=num_cube_cores
    )
    order_vector_tasks_by_stage(runtime_config)
    _apply_mega_kernel_profile_graph(
        runtime_config,
        graph,
        _ProfileSpec(kernel_name="HyperMegaMhcGrad", owner_label="NativeKernel"),
    )
    npu_device = torch.device(device)
    device_id = npu_device.index
    if device_id is None:
        device_id = torch.npu.current_device()

    def tensor_factory(data: bytes) -> torch.Tensor:
        """Materialize one serialized runtime buffer on the target NPU.

        Args:
            data: Serialized runtime bytes.

        Returns:
            Byte tensor resident on the target NPU.
        """
        return _tensor_from_bytes(data, npu_device)

    runtime = _prepare_mega_kernel_runtime_config(
        runtime_config,
        tensor_factory=tensor_factory,
        profile_tensor_factory=_enable_runtime_config_tensor,
        rank=0,
        device_id=device_id,
    )
    event_counters = torch.zeros(
        (runtime_config.event_capacity * 4,), dtype=torch.uint8, device=npu_device
    )
    return MegaMhcGradPlan(
        token_count,
        hidden_size,
        num_cube_cores,
        num_vector_cores,
        token_tile,
        runtime,
        event_counters,
    )
