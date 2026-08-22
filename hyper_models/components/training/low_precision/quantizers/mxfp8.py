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
"""MXFP8 quantizer for NPU dynamic block quantization."""

from typing import Optional

import torch  # pylint: disable=forbidden-backend-import

from hyper_models.components.training.low_precision.ops.npu_mxfp8 import (
    MXFP8NpuOps,
    _get_npu_ops,
)
from hyper_models.components.training.low_precision.quantizers.base import Quantizer
from hyper_models.components.training.low_precision.tensor import (
    MXFP8Tensor,
    QuantizedTensor,
)


class MXFP8Quantizer(Quantizer):
    """Build E4M3 data and E8M0 scales for MX block computation."""

    def __init__(
        self,
        quant_dtype: Optional[torch.dtype] = None,
        *,
        npu_ops: Optional[MXFP8NpuOps] = None,
    ) -> None:
        """Create a quantizer with lazy NPU operator resolution."""

        self.quant_dtype = quant_dtype or torch.float8_e4m3fn
        self._npu_ops = npu_ops

    @property
    def npu_ops(self) -> MXFP8NpuOps:
        """Return the injected or lazily resolved NPU operator adapter."""

        if self._npu_ops is None:
            self._npu_ops = _get_npu_ops()
        return self._npu_ops

    def quantize(
        self,
        tensor: torch.Tensor,
        *,
        group_list: Optional[torch.Tensor] = None,
        group_list_type: int = 0,
        rowwise: bool = False,
        colwise: bool = False,
    ) -> MXFP8Tensor:
        """Quantize a tensor into the requested MXFP8 directions."""

        if not isinstance(tensor, torch.Tensor) or isinstance(
            tensor,
            QuantizedTensor,
        ):
            raise TypeError(
                "MXFP8Quantizer expects a high-precision torch.Tensor."
            )
        if tensor.ndim < 2:
            raise ValueError(
                "MXFP8Quantizer expects at least two dimensions, but got "
                f"shape {tuple(tensor.shape)}."
            )
        if not rowwise and not colwise:
            raise ValueError(
                "MXFP8Quantizer requires rowwise=True or colwise=True."
            )
        if group_list_type not in (0, 1):
            raise ValueError(
                "MXFP8Quantizer group_list_type must be 0 or 1, "
                f"but got {group_list_type}."
            )
        if group_list is not None and (
            not isinstance(group_list, torch.Tensor) or group_list.ndim != 1
        ):
            raise ValueError(
                "MXFP8Quantizer group_list must be a one-dimensional Tensor."
            )

        row_data = None
        row_scale = None
        col_data = None
        col_scale = None
        if group_list is not None:
            if not colwise:
                raise ValueError(
                    "Grouped MXFP8 quantization requires colwise=True."
                )
            quant_group_list = (
                group_list
                if group_list_type == 0
                else torch.cumsum(group_list, dim=0)
            )
            if rowwise:
                row_data, row_scale = self.npu_ops.dynamic_mx_quant(
                    tensor,
                    axis=-1,
                    quant_dtype=self.quant_dtype,
                )
            col_data, col_scale = self.npu_ops.grouped_dynamic_mx_quant(
                tensor,
                quant_group_list,
                quant_dtype=self.quant_dtype,
            )
        elif rowwise and colwise:
            (
                row_data,
                row_scale,
                col_data,
                col_scale,
            ) = self.npu_ops.dynamic_mx_quant_dual_axis(
                tensor,
                quant_dtype=self.quant_dtype,
            )
        elif rowwise:
            row_data, row_scale = self.npu_ops.dynamic_mx_quant(
                tensor,
                axis=-1,
                quant_dtype=self.quant_dtype,
            )
        elif colwise:
            col_data, col_scale = self.npu_ops.dynamic_mx_quant(
                tensor,
                axis=-2,
                quant_dtype=self.quant_dtype,
            )
        return MXFP8Tensor(
            shape=tensor.shape,
            dtype=tensor.dtype,
            quantizer=self,
            row_data=row_data,
            row_scale=row_scale,
            col_data=col_data,
            col_scale=col_scale,
            device=tensor.device,
        )
