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

from hyper_parallel.core.optimizer.swap_optimizer_base import validate_state_keys


@dataclass(frozen=True)
class SwapOptimizerConfig:
    """Configuration for optimizer state swap.

    The runtime uses a fixed one-batch-ahead prefetch pipeline.

    Args:
        swap_times: Number of pipeline partitions.
        state_keys: Logical optimizer state keys to swap. ``None`` uses the
            defaults of the adapter matching the wrapped optimizer; an explicit
            value must only name keys that optimizer family actually owns.
        min_numel: Tensor states smaller than this element count are not swapped.
        packed_swap: Whether to use two packed A/B staging buffers. Defaults to
            ``True``; when ``False``, optimizer states are swapped tensor by
            tensor.
    """

    swap_times: int = 16
    state_keys: Optional[Sequence[str]] = None
    min_numel: int = 1024
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
    """Wrap a supported optimizer with optimizer-state swap.

    Public dispatcher: it resolves the optimizer family and defers to that
    family's factory, which owns adapter selection and the resulting error
    message.  Optimizer modules are imported lazily so loading this facade does
    not require every algorithm implementation.

    Args:
        optimizer: Base optimizer instance.
        config: Swap optimizer configuration.

    Returns:
        Swap optimizer wrapper.

    Raises:
        ValueError: If the optimizer type is unsupported.
    """
    resolved = config or SwapOptimizerConfig()
    if is_muon_optimizer(optimizer):
        from hyper_parallel.core.optimizer.swap_muon import (  # pylint: disable=import-outside-toplevel
            swap_muon,
        )
        return swap_muon(optimizer, resolved)

    from hyper_parallel.core.optimizer.swap_adam import (  # pylint: disable=import-outside-toplevel
        swap_adam,
    )
    return swap_adam(optimizer, resolved)


def is_swap_optimizer(optimizer: Any) -> bool:
    """Return whether ``optimizer`` is a swap optimizer wrapper."""
    return bool(getattr(optimizer, "_is_swap_optimizer", False))


def is_muon_optimizer(optimizer: Any) -> bool:
    """Return whether ``optimizer`` is a Muon leaf optimizer.

    The check is by concrete leaf type, never by class name or by inspecting the
    shape of a ``ChainedOptimizer``: a chain is not itself a Muon optimizer, and
    its leaves must each be wrapped on their own.
    """
    from hyper_parallel.core.optimizer.muon import (  # pylint: disable=import-outside-toplevel
        Muon,
    )
    return isinstance(optimizer, Muon)
