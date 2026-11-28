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
"""Dense Linear replacement baseline and strategy-bound low-precision Linear."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from hyper_parallel.components.quantization.functional.base_linear_func import (
    LinearStrategy,
    _LinearFunction,
)
from hyper_parallel.models.replacement import module_replacement


def _adopt_linear_parameters(target: nn.Linear, source: nn.Linear) -> None:
    """Reuse source Linear parameters without temporary allocation."""
    nn.Module.__init__(target)  # pylint: disable=non-parent-init-called,unnecessary-dunder-call
    target.in_features = source.in_features
    target.out_features = source.out_features
    target.register_parameter("weight", source.weight)
    target.register_parameter("bias", source.bias)


@module_replacement
class Linear(nn.Linear):
    """Baseline Dense replacement using ordinary torch linear computation."""

    def __init__(
        self,
        *,
        module: nn.Linear,
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
    ) -> None:
        """Reuse parameters without allocating or initializing a second weight set.

        Args:
            module: Source Linear with loaded weight and optional bias.
            module_fqn: Replacement path; unused by baseline computation.
            context: Replacement context; unused by baseline computation.
        """
        del module_fqn, context
        _adopt_linear_parameters(self, module)
        self.train(module.training)


class LowPrecisionLinear(nn.Linear):
    """Preserve Linear state while delegating compute to one bound strategy."""

    _hp_linear_compute_kind = "npu_quant"

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        strategy: LinearStrategy,
    ) -> None:
        """Create a low-precision Linear with newly allocated Parameters."""
        self._validate_strategy(strategy)
        super().__init__(in_features, out_features, bias=bias)
        self.strategy = strategy

    @staticmethod
    def _validate_strategy(strategy: LinearStrategy) -> None:
        """Require a plain strategy so module registration stays unchanged."""
        if isinstance(strategy, nn.Module) or not isinstance(strategy, LinearStrategy):
            raise TypeError("strategy must be a non-Module LinearStrategy instance")

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        strategy: LinearStrategy,
    ) -> "LowPrecisionLinear":
        """Create a shell retaining the source Parameters and training mode.

        Args:
            linear: Source Linear whose weight and optional bias are reused.
            strategy: Low-precision compute strategy bound to the replacement.

        Returns:
            A new module sharing the source Parameters without reinitializing them.
        """
        cls._validate_strategy(strategy)
        # object.__new__ is a static method; passing cls explicitly avoids the
        # allocation and initialization performed by nn.Linear.__init__.
        converted = cls.__new__(cls)  # pylint: disable=no-value-for-parameter
        _adopt_linear_parameters(converted, linear)
        converted.strategy = strategy
        converted.training = linear.training
        return converted

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # pylint: disable=redefined-builtin
        """Apply the selected low-precision compute and high-precision bias."""
        output = _LinearFunction.apply(input, self.weight, self.strategy)
        if self.bias is not None:
            output = output + self.bias
        return output


__all__ = ["Linear", "LowPrecisionLinear"]
