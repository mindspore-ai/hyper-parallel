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
"""Dense A5 MXFP8 training module."""

from typing import TYPE_CHECKING, Optional

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_models.components.training.low_precision.functional import (
    npu_quant_linear,
)
from hyper_models.components.training.low_precision.ops import (
    NpuCapabilityError,
)
from hyper_models.components.training.low_precision.quantizers import (
    MXFP8Quantizer,
)

if TYPE_CHECKING:
    from hyper_models.components.training.low_precision.observer.bridge import (
        PrecisionDebugSession,
    )


class NpuQuantLinear(nn.Module):
    """Preserve Linear parameters while changing its GEMM boundary."""

    _hp_linear_compute_kind = "npu_quant"

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        fqn: str = "",
        quantizer: Optional[MXFP8Quantizer] = None,
    ) -> None:
        """Create an unbound low-precision Linear shell."""

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fqn = fqn
        self.quantizer = (
            quantizer if quantizer is not None else MXFP8Quantizer()
        )
        self._precision_debug_session: Optional["PrecisionDebugSession"] = None
        self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        fqn: str,
        quantizer: Optional[MXFP8Quantizer] = None,
    ) -> "NpuQuantLinear":
        """Replace a Linear while retaining its exact parameter objects."""

        converted = cls(
            linear.in_features,
            linear.out_features,
            fqn=fqn,
            quantizer=quantizer,
        )
        converted.weight = linear.weight
        converted.bias = linear.bias
        converted.training = linear.training
        return converted

    def set_precision_debug_session(
        self,
        session: "PrecisionDebugSession",
    ) -> None:
        """Bind the optional, training-owned precision diagnostic session."""
        self._precision_debug_session = session

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply A5 MXFP8 GEMMs and preserve the optional high-precision bias."""

        try:
            output = npu_quant_linear(
                inputs,
                self.weight,
                self.quantizer,
                observer=self._precision_debug_session,
                module_fqn=self.fqn,
            )
        except NpuCapabilityError as exc:
            target = self.fqn or "<unknown>"
            raise NpuCapabilityError(
                f"Low-precision target {target!r} cannot run: {exc}"
            ) from exc
        if self.bias is not None:
            output = output + self.bias
        return output
