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
"""Public API for optimizer state swap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from hyper_parallel.core.optimizer.swap_optimizer_base import (
    SwapOptimizer as _SwapOptimizer,
)
from hyper_parallel.core.optimizer.swap_optimizer_base import validate_state_keys


@dataclass(frozen=True)
class SwapOptimizerConfig:
    """Configuration for Adam/AdamW optimizer state swap.

    The runtime uses a fixed one-batch-ahead prefetch pipeline.

    Args:
        swap_times: Number of pipeline partitions.
        state_keys: Logical state keys to swap. ``None`` uses adapter defaults.
        min_numel: Tensor states smaller than this element count are not swapped.
        include_master_params: Whether optimizer-owned fp32 master params are swapped.
        packed_swap: Whether to use two packed A/B staging buffers. Defaults to
            ``True``; when ``False``, optimizer states are swapped tensor by
            tensor.
    """

    swap_times: int = 16
    state_keys: Optional[Sequence[str]] = None
    min_numel: int = 1024
    include_master_params: bool = False
    packed_swap: bool = True

    def __post_init__(self) -> None:
        if self.swap_times <= 0:
            raise ValueError("SwapOptimizerConfig.swap_times must be positive.")
        if self.min_numel < 0:
            raise ValueError("SwapOptimizerConfig.min_numel must be non-negative.")
        object.__setattr__(self, "state_keys", validate_state_keys(self.state_keys))


class SwapOptimizer:
    """Core facade that creates the Torch optimizer-state swap wrapper."""

    def __new__(cls, optimizer: Any, config: Optional[SwapOptimizerConfig] = None):
        return swap_optimizer(optimizer, config)


def swap_optimizer(optimizer: Any, config: Optional[SwapOptimizerConfig] = None) -> Any:
    """Wrap a supported Adam/AdamW optimizer with optimizer-state swap.

    Args:
        optimizer: Base optimizer instance.
        config: Swap optimizer configuration.

    Returns:
        Swap optimizer wrapper.

    Raises:
        ValueError: If the optimizer type is unsupported.
    """
    return _SwapOptimizer(optimizer, config or SwapOptimizerConfig())


def is_swap_optimizer(optimizer: Any) -> bool:
    """Return whether ``optimizer`` is a swap optimizer wrapper."""
    return bool(getattr(optimizer, "_is_swap_optimizer", False))
