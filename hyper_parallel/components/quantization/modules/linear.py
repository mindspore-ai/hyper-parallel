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
"""Canonical shell and single replacement for low-precision Dense Linear."""

from collections.abc import Mapping
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.quantization.functional.base_linear_func import (
    LinearStrategy,
    _LinearFunction,
)
from hyper_parallel.components.quantization.functional.linear_strategy_factory import (
    build_linear_strategy,
)
from hyper_parallel.models.replacement import module_replacement


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
        """Create a no-allocation shell retaining the source Parameters."""

        cls._validate_strategy(strategy)
        converted = cls.__new__(cls)
        nn.Module.__init__(converted)  # pylint: disable=unnecessary-dunder-call
        converted.in_features = linear.in_features
        converted.out_features = linear.out_features
        converted.register_parameter("weight", linear.weight)
        converted.register_parameter("bias", linear.bias)
        converted.strategy = strategy
        converted.training = linear.training
        return converted

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # pylint: disable=redefined-builtin
        """Apply the selected low-precision compute and high-precision bias."""

        output = _LinearFunction.apply(input, self.weight, self.strategy)
        if self.bias is not None:
            output = output + self.bias
        return output


@module_replacement
def replace_linear(
    *,
    module: nn.Module,
    module_fqn: str,
    context: Mapping[str, Any],
) -> LowPrecisionLinear:
    """Replace one exact Dense Linear using the selected dtype policy."""

    if type(module) is not nn.Linear:  # pylint: disable=unidiomatic-typecheck
        raise TypeError(
            f"{module_fqn!r} must be exact nn.Linear, got {type(module).__name__}"
        )
    if context.get("pp"):
        raise NotImplementedError(
            "Low-precision Linear training is not yet supported with pipeline parallelism."
        )
    try:
        strategy = build_linear_strategy(
            context.get("low_precision"),
            in_features=module.in_features,
            out_features=module.out_features,
        )
    except ValueError as error:
        raise ValueError(
            f"Low-precision Linear target {module_fqn!r} is invalid: {error}"
        ) from error
    return LowPrecisionLinear.from_linear(module, strategy=strategy)


__all__ = ["LowPrecisionLinear", "replace_linear"]
