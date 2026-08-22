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
"""HiFloat8 Linear adapter and explicit replacement factory."""

from collections.abc import Mapping
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_models.components.model_transform.replacement import module_replacement
from hyper_models.components.training.low_precision.functional.hifloat8_linear_func import (
    hifloat8_linear,
)
from hyper_models.components.training.low_precision.module_adapter.linear import (
    QuantizedLinearBase,
)
from hyper_models.components.training.low_precision.ops.npu_hifloat8 import (
    validate_hifloat8_runtime,
)
from hyper_models.components.training.low_precision.quantizers.hifloat8 import (
    GRADIENT_FORMAT_MAX,
    INPUT_WEIGHT_FORMAT_MAX,
    HiFloat8Quantizer,
)


class HiFloat8Linear(QuantizedLinearBase):
    """Own Pangu-compatible X/W/G current-scaling role quantizers."""

    def _initialize_format(self) -> None:
        self.input_quantizer = HiFloat8Quantizer(
            fp8_max=INPUT_WEIGHT_FORMAT_MAX
        )
        self.weight_quantizer = HiFloat8Quantizer(
            fp8_max=INPUT_WEIGHT_FORMAT_MAX
        )
        self.grad_output_quantizer = HiFloat8Quantizer(
            fp8_max=GRADIENT_FORMAT_MAX
        )

    def _apply_low_precision(self, inputs: torch.Tensor) -> torch.Tensor:
        return hifloat8_linear(
            inputs,
            self.weight,
            self.input_quantizer,
            self.weight_quantizer,
            self.grad_output_quantizer,
        )


@module_replacement
def replace_hifloat8_linear(
    *,
    module: nn.Module,
    module_fqn: str,
    context: Mapping[str, Any],
) -> HiFloat8Linear:
    """Build a current-scaling HiFloat8 owner for one exact Dense Linear."""

    if type(module) is not nn.Linear:  # pylint: disable=unidiomatic-typecheck
        raise TypeError(
            f"{module_fqn!r} must be exact nn.Linear, got {type(module).__name__}"
        )
    if context.get("pp"):
        raise NotImplementedError(
            "HiFloat8 online training is not yet supported with pipeline parallelism."
        )
    validate_hifloat8_runtime()
    return HiFloat8Linear.from_linear(module)


__all__ = ["HiFloat8Linear", "replace_hifloat8_linear"]
