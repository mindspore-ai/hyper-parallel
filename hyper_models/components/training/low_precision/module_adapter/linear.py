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
"""Common parameter-preserving base for low-precision Linear adapters."""

from abc import ABC, abstractmethod
from typing import TypeVar

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import

_LinearType = TypeVar("_LinearType", bound="QuantizedLinearBase")


class QuantizedLinearBase(nn.Linear, ABC):
    """Preserve Linear parameters while delegating format-specific compute."""

    _hp_linear_compute_kind = "npu_quant"

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ) -> None:
        """Create a format-specific low-precision Linear."""

        super().__init__(in_features, out_features, bias=True)
        self._initialize_format()

    @classmethod
    def from_linear(
        cls: type[_LinearType],
        linear: nn.Linear,
    ) -> _LinearType:
        """Create a no-allocation shell while retaining Parameter objects."""

        # Do not call nn.Linear.__init__: it allocates and initializes a full
        # temporary weight before we replace it. This remains safe for large
        # model conversion and meta-device construction.
        converted = cls.__new__(cls)
        nn.Module.__init__(converted)
        converted.in_features = linear.in_features
        converted.out_features = linear.out_features
        converted.register_parameter("weight", linear.weight)
        converted.register_parameter("bias", linear.bias)
        converted.training = linear.training
        converted._initialize_format()
        return converted

    @abstractmethod
    def _initialize_format(self) -> None:
        """Create format-specific quantizers and state."""

    @abstractmethod
    def _apply_low_precision(self, inputs: torch.Tensor) -> torch.Tensor:
        """Execute one format-specific bias-free Dense operation."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # pylint: disable=redefined-builtin
        """Apply low-precision compute and preserve the high-precision bias."""

        output = self._apply_low_precision(input)
        if self.bias is not None:
            output = output + self.bias
        return output


__all__ = ["QuantizedLinearBase"]
