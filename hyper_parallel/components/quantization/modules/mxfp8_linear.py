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
"""MXFP8 Linear adapter and explicit replacement factory."""

from collections.abc import Mapping
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.models.replacement import module_replacement
from hyper_parallel.components.quantization.functional.mxfp8_linear_func import (
    mxfp8_linear,
)
from hyper_parallel.components.quantization.modules.linear import (
    QuantizedLinearBase,
)
from hyper_parallel.components.quantization.functional.npu_mxfp8 import (
    validate_npu_runtime,
)
from hyper_parallel.components.quantization.quantizers.mxfp8 import (
    MXFP8Quantizer,
)


class MXFP8Linear(QuantizedLinearBase):
    """Own MXFP8 quantization and Dense autograd selection."""

    def _initialize_format(self) -> None:
        self.quantizer = MXFP8Quantizer()

    def _apply_low_precision(self, inputs: torch.Tensor) -> torch.Tensor:
        return mxfp8_linear(inputs, self.weight, self.quantizer)


@module_replacement
def replace_mxfp8_linear(
    *,
    module: nn.Module,
    module_fqn: str,
    context: Mapping[str, Any],
) -> MXFP8Linear:
    """Build an MXFP8 owner for one exact, tile-aligned Dense Linear."""

    if type(module) is not nn.Linear:  # pylint: disable=unidiomatic-typecheck
        raise TypeError(
            f"{module_fqn!r} must be exact nn.Linear, got {type(module).__name__}"
        )
    if module.in_features % 32 or module.out_features % 32:
        raise ValueError(
            f"{module_fqn!r} is not MXFP8 tile aligned: "
            f"({module.out_features}, {module.in_features}) requires multiples of 32"
        )
    if context.get("pp"):
        raise NotImplementedError(
            "MXFP8 online training is not yet supported with pipeline parallelism."
        )
    validate_npu_runtime()
    return MXFP8Linear.from_linear(module)


__all__ = ["MXFP8Linear", "replace_mxfp8_linear"]
