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
"""Pangu-compatible HiFloat8 current-scaling quantizer."""

from typing import Any, Optional

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.quantization.functional.npu_hifloat8 import (
    HiFloat8NpuOps,
    _get_hifloat8_npu_ops,
)
from hyper_parallel.components.quantization.quantizers.base import Quantizer
from hyper_parallel.components.quantization.tensor.hifloat8_tensor import (
    HiFloat8Tensor,
)
from hyper_parallel.components.quantization.tensor.quantized_tensor import (
    QuantizedTensor,
)

INPUT_WEIGHT_FORMAT_MAX = 15.0
GRADIENT_FORMAT_MAX = 224.0
_SUPPORTED_FORMAT_MAX = (INPUT_WEIGHT_FORMAT_MAX, GRADIENT_FORMAT_MAX)


class HiFloat8Quantizer(Quantizer):
    """Quantize one HiFloat8 role with stateless current tensorwise scaling."""

    def __init__(
        self,
        *,
        fp8_max: float,
        npu_ops: Optional[HiFloat8NpuOps] = None,
    ) -> None:
        """Create one frozen-role quantizer compatible with Pangu's structure."""

        if float(fp8_max) not in _SUPPORTED_FORMAT_MAX:
            raise ValueError(
                "HiFloat8 current scaling supports format max 15 or 224, "
                f"got {fp8_max}."
            )
        self.fp8_max = float(fp8_max)
        self._npu_ops = npu_ops

    @property
    def npu_ops(self) -> HiFloat8NpuOps:
        """Return the injected or lazily resolved NPU operator adapter."""

        if self._npu_ops is None:
            self._npu_ops = _get_hifloat8_npu_ops()
        return self._npu_ops

    @property
    def quant_dtype(self) -> Any:
        """Return the runtime-frozen logical HiFloat8 dtype identifier."""

        return self.npu_ops.quant_dtype

    @staticmethod
    def _validate_input(
        tensor: torch.Tensor,
        *,
        group_list: Optional[torch.Tensor],
        group_list_type: int,
        rowwise: bool,
        colwise: bool,
    ) -> None:
        if not isinstance(tensor, torch.Tensor) or isinstance(
            tensor,
            QuantizedTensor,
        ):
            raise TypeError(
                "HiFloat8Quantizer expects a high-precision torch.Tensor."
            )
        if tensor.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError(
                "HiFloat8 current quantization requires float16 or bfloat16 "
                f"input, got {tensor.dtype}."
            )
        if tensor.ndim not in (2, 3):
            raise ValueError(
                "HiFloat8 quantization requires a matrix or grouped weight, "
                "got shape "
                f"{tuple(tensor.shape)}."
            )
        if not rowwise and not colwise:
            raise ValueError(
                "HiFloat8Quantizer requires rowwise=True or colwise=True."
            )
        if group_list_type not in (0, 1):
            raise ValueError(
                "HiFloat8Quantizer group_list_type must be 0 or 1, "
                f"got {group_list_type}."
            )
        if group_list is not None:
            if tensor.ndim != 2:
                raise ValueError(
                    "Grouped HiFloat8 quantization requires a 2D operand."
                )
            if not isinstance(group_list, torch.Tensor) or group_list.ndim != 1:
                raise ValueError(
                    "HiFloat8Quantizer group_list must be one-dimensional."
                )
            if group_list.dtype != torch.int64:
                raise TypeError(
                    "HiFloat8Quantizer group_list must use torch.int64."
                )

    def quantize(
        self,
        tensor: torch.Tensor,
        *,
        group_list: Optional[torch.Tensor] = None,
        group_list_type: int = 0,
        rowwise: bool = False,
        colwise: bool = False,
    ) -> HiFloat8Tensor:
        """Quantize Dense or grouped operands with current scaling.

        Two-dimensional activation/gradient operands use one tensorwise scale.
        GMM callers expand that scalar to one identical entry per expert.
        Three-dimensional expert weights use one scale per leading expert.
        """

        self._validate_input(
            tensor,
            group_list=group_list,
            group_list_type=group_list_type,
            rowwise=rowwise,
            colwise=colwise,
        )
        if tensor.ndim == 3:
            flattened = tensor.reshape(tensor.shape[0], -1)
            data, scale = self.npu_ops.dynamic_quant(
                flattened,
                dst_type_max=self.fp8_max,
                quant_mode="pertoken",
            )
            data = data.reshape_as(tensor)
        else:
            data, scale = self.npu_ops.dynamic_quant(
                tensor,
                dst_type_max=self.fp8_max,
                quant_mode="pertensor",
            )
        scale = torch.where(
            torch.isfinite(scale) & (scale > 0),
            scale,
            torch.ones_like(scale),
        )
        if group_list is not None and scale.numel() == 1:
            scale = scale.reshape(1).expand(group_list.numel()).contiguous()
        return HiFloat8Tensor(
            shape=tensor.shape,
            dtype=tensor.dtype,
            quantizer=self,
            row_data=data if rowwise else None,
            row_scale=scale if rowwise else None,
            col_data=data if colwise else None,
            col_scale=scale if colwise else None,
            device=tensor.device,
        )


__all__ = [
    "GRADIENT_FORMAT_MAX",
    "HiFloat8Quantizer",
    "INPUT_WEIGHT_FORMAT_MAX",
]
