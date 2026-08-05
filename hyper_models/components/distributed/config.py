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
"""Distributed strategy configuration — stubs for 06 §3.1 / §4.

Full implementation will move to this module from infrastructure.py.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MixedPrecisionPolicy:
    """FSDP2 mixed precision policy stub."""
    param_dtype: Any = None
    reduce_dtype: Any = None
    output_dtype: Any = None


@dataclass
class CPUOffloadPolicy:
    """FSDP2 CPU offload policy stub."""
    pin_memory: bool = False


@dataclass
class FSDP2Config:
    """FSDP2 strategy configuration (06 §4.1)."""
    dp_shard_size: int = 1
    replicate_params: list[str] = field(default_factory=list)
    activation_checkpointing: bool | str = False
    mp_policy: MixedPrecisionPolicy | None = None
    offload_policy: CPUOffloadPolicy | None = None
    reshard_after_forward: bool = True
    reshard_after_backward: bool = False
    defer_fsdp_grad_sync: bool = True
    enable_async_tensor_parallel: bool = False
    enable_compile: bool = False
    enable_fsdp2_prefetch: bool = False
    backward_prefetch_depth: int = 1
    forward_prefetch_depth: int = 1
    comm_fusion: bool = False
    comm_fusion_zero_copy: bool | None = None
        
    def __post_init__(self) -> None:
        """Validate topology sizes and prefetch depths."""
        if self.dp_shard_size < 1:
            raise ValueError("dp_shard_size must be greater than or equal to 1")
        if self.backward_prefetch_depth < 0:
            raise ValueError("backward_prefetch_depth must be greater than or equal to 0")
        if self.forward_prefetch_depth < 0:
            raise ValueError("forward_prefetch_depth must be greater than or equal to 0")
