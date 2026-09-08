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
"""Validated shape, topology, and hardware specification for MegaMoe."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Any

torch = import_module("torch")
dist = import_module("torch.distributed")

_COMMUNICATION_SPLIT = 128


@dataclass(frozen=True)
class MegaMoeSpec:
    """Shape, capacity, topology, and hardware values bound on first use."""

    local_num_tokens: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    expert_capacity_factor: float | None
    receive_capacity: int
    ep_size: int
    ep_group: Any | None
    rank_id: int
    num_cube_cores: int
    dispatch_split: int = _COMMUNICATION_SPLIT
    combine_split: int = _COMMUNICATION_SPLIT
    swiglu_split: int = _COMMUNICATION_SPLIT

    @property
    def local_experts(self) -> int:
        """Return experts owned by this expert-parallel rank."""
        return self.num_experts // self.ep_size

    @property
    def routed_slots(self) -> int:
        """Return routed rows produced by local Top-K expansion."""
        return self.local_num_tokens * self.top_k

    @property
    def capacity_is_lossless(self) -> bool:
        """Return whether capacity covers the worst possible route."""
        return self.expert_capacity_factor is None


def _align_capacity(capacity: int) -> int:
    """Align one receive capacity to the fixed communication split."""
    return (
        (capacity + _COMMUNICATION_SPLIT - 1)
        // _COMMUNICATION_SPLIT
        * _COMMUNICATION_SPLIT
    )


def _resolve_receive_capacity(
    expert_capacity_factor: float | None,
    routed_slots: int,
    ep_size: int,
) -> int:
    """Resolve lossless or explicitly bounded receive capacity."""
    if expert_capacity_factor is None:
        requested = ep_size * routed_slots
    else:
        requested = math.ceil(routed_slots * expert_capacity_factor)
    return _align_capacity(requested)


def bind_mega_moe_spec(
    specification: Mapping[str, Any],
    tensor: torch.Tensor,
) -> MegaMoeSpec:
    """Bind hardware and process-group values to the first NPU input.

    Args:
        specification: Constructor-validated static resource specification.
        tensor: First input tensor used to resolve device-specific limits.

    Returns:
        Bound specification cached for subsequent forwards.

    Raises:
        ValueError: If the device or EP process group does not match the specification.
        TypeError: If the activation dtype is unsupported.
        RuntimeError: If the device does not report a valid Cube-core count.
    """
    device = torch.device(tensor.device)
    if device.type != "npu":
        raise ValueError(f"MegaMoeExperts requires an NPU input, got {device}.")
    if tensor.dtype != torch.bfloat16:
        raise TypeError(
            f"MegaMoeExperts supports BF16 activations, got {tensor.dtype}."
        )

    ep_group = specification["ep_group"]
    ep_size = specification["ep_size"]
    if dist.is_initialized():
        group_size = dist.get_world_size(ep_group)
        rank_id = dist.get_rank(ep_group)
    else:
        if ep_group is not None:
            raise RuntimeError(
                "ep_group cannot be used before torch.distributed initialization."
            )
        group_size = 1
        rank_id = 0
    if group_size != ep_size:
        raise ValueError(
            f"ep_group world size ({group_size}) does not match ep_size ({ep_size})."
        )

    device_index = (
        device.index if device.index is not None else torch.npu.current_device()
    )
    device_limits = torch.npu.get_device_limit(device_index)
    num_cube_cores = device_limits.get("cube_core_num")
    if not isinstance(num_cube_cores, int) or num_cube_cores <= 0:
        raise RuntimeError(
            "torch.npu.get_device_limit() returned an invalid cube_core_num "
            f"for NPU {device_index}: {num_cube_cores!r}."
        )

    local_num_tokens = specification["local_num_tokens"]
    top_k = specification["top_k"]
    expert_capacity_factor = specification["expert_capacity_factor"]
    receive_capacity = _resolve_receive_capacity(
        expert_capacity_factor,
        local_num_tokens * top_k,
        ep_size,
    )
    return MegaMoeSpec(
        local_num_tokens=local_num_tokens,
        hidden_size=specification["hidden_size"],
        intermediate_size=specification["intermediate_size"],
        num_experts=specification["num_experts"],
        top_k=top_k,
        expert_capacity_factor=expert_capacity_factor,
        receive_capacity=receive_capacity,
        ep_size=ep_size,
        ep_group=ep_group,
        rank_id=rank_id,
        num_cube_cores=num_cube_cores,
    )
