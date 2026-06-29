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

from dataclasses import dataclass, field, replace
from typing import Any, Optional, Sequence

from hyper_parallel.core.optimizer.swap_optimizer_base import validate_state_keys
from hyper_parallel.platform import get_platform
from hyper_parallel.platform.platform import PlatformType


def _default_packed_swap() -> bool:
    """Return the packed swap default for the active backend."""
    return get_platform().platform_type == PlatformType.PYTORCH


def _is_mindformers_adamw(optimizer: Any) -> bool:
    """Return whether an optimizer is the supported MindFormers AdamW type."""
    optimizer_type = type(optimizer)
    return (
        optimizer_type.__name__ == "AdamW"
        and optimizer_type.__module__ == "mindformers.pynative.optimizer.adamw"
    )


def _resolve_packed_swap_default(
    optimizer: Any,
    config: "SwapOptimizerConfig",
    platform_type: PlatformType,
) -> "SwapOptimizerConfig":
    """Enable packed swap by default for MindFormers AdamW on MindSpore."""
    if (
        platform_type == PlatformType.MINDSPORE
        and not config.packed_swap_was_explicit
        and _is_mindformers_adamw(optimizer)
    ):
        return replace(config, packed_swap=True)
    return config


@dataclass(frozen=True)
class SwapOptimizerConfig:
    """Configuration for Adam/AdamW optimizer state swap.

    The runtime uses a fixed one-batch-ahead prefetch pipeline.

    Args:
        swap_times: Number of pipeline partitions.
        state_keys: Logical state keys to swap. ``None`` uses adapter defaults.
        min_numel: Tensor states smaller than this element count are not swapped.
        include_master_params: Whether optimizer-owned fp32 master params are swapped.
        packed_swap: Whether supported backends use two packed A/B staging buffers.
            Defaults to ``True`` on PyTorch and for MindFormers AdamW on
            MindSpore; other MindSpore optimizers default to ``False``. When
            ``False``, optimizer states are swapped tensor by tensor. An
            explicit value always takes precedence over these defaults.
    """

    swap_times: int = 16
    state_keys: Optional[Sequence[str]] = None
    min_numel: int = 1024
    include_master_params: bool = False
    packed_swap: Optional[bool] = None
    _packed_swap_explicit: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        packed_swap_explicit = self.packed_swap is not None
        object.__setattr__(self, "_packed_swap_explicit", packed_swap_explicit)
        if not packed_swap_explicit:
            object.__setattr__(self, "packed_swap", _default_packed_swap())
        if self.swap_times <= 0:
            raise ValueError("SwapOptimizerConfig.swap_times must be positive.")
        if self.min_numel < 0:
            raise ValueError("SwapOptimizerConfig.min_numel must be non-negative.")
        object.__setattr__(self, "state_keys", validate_state_keys(self.state_keys))

    @property
    def packed_swap_was_explicit(self) -> bool:
        """Return whether ``packed_swap`` was explicitly supplied by the caller."""
        return self._packed_swap_explicit


class SwapOptimizer:
    """Core facade that dispatches to the active backend implementation."""

    def __new__(cls, optimizer: Any, config: Optional[SwapOptimizerConfig] = None):
        return swap_optimizer(optimizer, config)


def swap_optimizer(optimizer: Any, config: Optional[SwapOptimizerConfig] = None) -> Any:
    """Wrap a supported Adam/AdamW optimizer with optimizer-state swap.

    Args:
        optimizer: Base optimizer instance.
        config: Swap optimizer configuration.

    Returns:
        Backend-specific swap optimizer wrapper.

    Raises:
        ValueError: If the active backend or optimizer type is unsupported.
    """
    platform = get_platform()
    cfg = config or SwapOptimizerConfig()
    cfg = _resolve_packed_swap_default(optimizer, cfg, platform.platform_type)
    return platform.get_swap_optimizer()(optimizer, cfg)


def is_swap_optimizer(optimizer: Any) -> bool:
    """Return whether ``optimizer`` is a swap optimizer wrapper."""
    return bool(getattr(optimizer, "_is_swap_optimizer", False))
