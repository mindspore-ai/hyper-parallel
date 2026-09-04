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
"""Root mean square normalization module."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
from hyper_parallel.components.checkpoint.weight_conversion import WeightConverter

from hyper_parallel.components.checkpoint.conversion_ops import AddScalar
from hyper_parallel.models.replacement import module_replacement
from hyper_parallel.components.functional import rms_norm


@module_replacement
class RMSNorm(torch.nn.Module):
    """NPU-accelerated replacement for direct-scale RMSNorm."""

    def __init__(
        self,
        hidden_size: int | None = None,
        eps: float = 1e-5,
        *,
        module: torch.nn.Module | None = None,
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
    ) -> None:
        """Build a direct-scale RMSNorm or replace an existing module.

        Args:
            hidden_size: Size of the normalized dimension for direct construction.
            eps: Epsilon added to the variance.
            module: Optional source module whose parameter layout is preserved.
            module_fqn: Fully qualified source-module name supplied by replacement.
            context: Replacement context supplied by Trainer.

        Raises:
            TypeError: If the source module does not expose a supported RMSNorm contract.
            ValueError: If neither a source module nor a positive hidden size is supplied.
        """
        super().__init__()
        del module_fqn, context
        if module is not None:
            if not isinstance(getattr(module, "weight", None), torch.nn.Parameter):
                raise TypeError("RMSNorm source module must expose an nn.Parameter weight")
            source_eps = getattr(module, "eps", None)
            if source_eps is None:
                source_eps = getattr(module, "variance_epsilon", None)
            if source_eps is None:
                raise TypeError(
                    "RMSNorm source module must expose eps or variance_epsilon"
                )
            self.eps = float(source_eps)
            self.weight = module.weight
            self.train(module.training)
            return
        if hidden_size is None or hidden_size <= 0:
            raise ValueError("RMSNorm hidden_size must be a positive integer")
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply NPU-accelerated RMS normalization."""
        return rms_norm(hidden_states, self.weight, self.eps)


@module_replacement
class OffsetRMSNorm(torch.nn.Module):
    """High-performance replacement for RMSNorm using ``1 + weight``."""

    def __init__(
        self,
        *,
        module: torch.nn.Module,
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
    ) -> None:
        """Build a direct-scale RMSNorm from a unit-offset source module."""
        super().__init__()
        del module_fqn, context
        if not hasattr(module, "weight") or not hasattr(module, "eps"):
            raise TypeError(
                "OffsetRMSNorm source module must expose weight and eps"
            )
        self.eps = module.eps
        weight = (
            torch.empty_like(module.weight)
            if module.weight.is_meta
            else module.weight.detach() + 1.0
        )
        self.weight = torch.nn.Parameter(
            weight,
            requires_grad=module.weight.requires_grad,
        )
        self.train(module.training)

    def reset_parameters(self) -> None:
        """Initialize the direct scale used by randomly initialized models."""
        torch.nn.init.ones_(self.weight)

    def make_transforms(self) -> list[WeightConverter]:
        """Convert the unit-offset weight to the NPU operator's direct scale."""
        return [
            WeightConverter(
                source_patterns=["weight"],
                target_patterns="weight",
                operations=[AddScalar(1.0)],
            )
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply unit-offset RMSNorm with the NPU operator."""
        return rms_norm(x, self.weight, self.eps)
