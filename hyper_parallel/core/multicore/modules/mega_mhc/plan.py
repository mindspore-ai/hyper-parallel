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
"""Shape-bound RuntimeConfig plan for HyperMegaMhc."""

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

from .graph import DEFAULT_TOKEN_TILE, build_mega_mhc_graph, resolve_token_tile


@dataclass(frozen=True)
class MegaMhcPlan:
    """Own one token-shape schedule and its reusable event counters."""

    token_count: int
    hidden_size: int
    num_cube_cores: int
    token_tile: int
    runtime: _PreparedMegaKernelRuntime
    event_counters: torch.Tensor

    @property
    def runtime_config(self) -> torch.Tensor:
        """Return the profiling-disabled RuntimeConfig tensor."""
        return self.runtime.normal_tensor


def _tensor_from_bytes(data: bytes, device: Any) -> torch.Tensor:
    """Copy serialized Host scheduling data into one NPU byte tensor."""
    array = np.frombuffer(bytearray(data), dtype=np.uint8).copy()
    return torch.from_numpy(array).to(device=device, dtype=torch.uint8)


def build_mega_mhc_plan(
    token_count: int,
    hidden_size: int,
    device: Any,
    num_cube_cores: int,
    token_tile: int = DEFAULT_TOKEN_TILE,
) -> MegaMhcPlan:
    """Build and materialize a token-owned HyperMegaMhc schedule.

    Args:
        token_count: Total flattened token count.
        hidden_size: Hidden dimension.
        device: Target NPU device.
        num_cube_cores: Physical AIC core count.
        token_tile: Preferred token count per task.
    """
    token_tile = resolve_token_tile(token_count, token_tile, num_cube_cores)
    graph, topology = build_mega_mhc_graph(
        token_count,
        hidden_size,
        num_cube_cores,
        token_tile,
    )
    runtime_config = build_runtime_config(
        graph,
        topology,
        rank_id=0,
        num_cube_cores=num_cube_cores,
    )
    _apply_mega_kernel_profile_graph(
        runtime_config,
        graph,
        _ProfileSpec(kernel_name="HyperMegaMhc", owner_label="TokenPartition"),
    )
    npu_device = torch.device(device)
    device_id = npu_device.index
    if device_id is None:
        device_id = torch.npu.current_device()

    def tensor_factory(data: bytes) -> torch.Tensor:
        """Create one byte-backed runtime tensor on the selected NPU.

        Args:
            data: Serialized runtime bytes.
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
        (runtime_config.event_capacity * 4,),
        dtype=torch.uint8,
        device=npu_device,
    )
    return MegaMhcPlan(
        token_count=token_count,
        hidden_size=hidden_size,
        num_cube_cores=num_cube_cores,
        token_tile=token_tile,
        runtime=runtime,
        event_counters=event_counters,
    )
