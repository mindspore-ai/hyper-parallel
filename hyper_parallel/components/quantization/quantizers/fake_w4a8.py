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
"""Fake-QAT W4A8 quantizer.

This quantizer is deliberately separate from ``W4A8Quantizer``.  Native W4A8
stores packed MXFP4/WeightNz data; fake W4A8 stores only the MXFP8 operands
that the all-MXFP8 GMM consumes after an on-the-fly MXFP4 quantize/dequantize
step.
"""

from typing import Optional

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.quantization.ops.npu_fake_w4a8 import (
    convert_mx_scale_to_axis2,
    expand_scale_to_per_row,
    fake_mxfp4_to_mxfp8,
    transform_grouped_scale,
    _get_fake_w4a8_npu_ops,
    FakeW4A8NpuOps,
)
from hyper_parallel.components.quantization.quantizers.base import Quantizer
from hyper_parallel.components.quantization.tensor import MXFP8Tensor, QuantizedTensor


_SUPPORTED_BLOCK_SIZES = (32, 128)


class FakeW4A8Quantizer(Quantizer):
    """Build fake MXFP4 weights and MXFP8 activation/backward operands."""

    def __init__(self, block_size: int = 32) -> None:
        if block_size not in _SUPPORTED_BLOCK_SIZES:
            raise ValueError(
                f"Fake W4A8 block_size must be 32 or 128, got {block_size}."
            )
        self.block_size = block_size

    @property
    def npu_ops(self) -> FakeW4A8NpuOps:
        return _get_fake_w4a8_npu_ops()

    @staticmethod
    def _boundaries(
        group_list: Optional[torch.Tensor],
        group_list_type: int,
    ) -> Optional[torch.Tensor]:
        if group_list is None:
            return None
        return group_list if group_list_type == 0 else torch.cumsum(group_list, dim=0)

    @staticmethod
    def _scale_to_fp32(scale: torch.Tensor) -> torch.Tensor:
        return torch.pow(2.0, scale.to(torch.float32) - 127.0)

    def _quantize_activation(
        self,
        tensor: torch.Tensor,
        *,
        rowwise: bool,
        colwise: bool,
        group_list: Optional[torch.Tensor],
        group_list_type: int,
    ) -> MXFP8Tensor:
        row_data = row_scale = col_data = col_scale = None
        boundaries = self._boundaries(group_list, group_list_type)
        if rowwise:
            row_data, row_scale = self.npu_ops.dynamic_mx_quant(
                tensor,
                axis=-1,
                block_size=self.block_size,
                quant_dtype=self.npu_ops.activation_dtype,
            )
            if self.block_size == 128:
                row_blocks = (tensor.shape[-1] + self.block_size - 1) // self.block_size
                row_scale = self._scale_to_fp32(row_scale)
                row_scale = row_scale.reshape(tensor.shape[0], -1)[..., :row_blocks]
        if colwise:
            if self.block_size == 32 and boundaries is not None:
                # This is the G-B grouped backward quantizer.  The operator
                # requires int32 group indices and has no scale_alg argument.
                col_data, col_scale = self.npu_ops.grouped_dynamic_mx_quant(
                    tensor,
                    boundaries,
                    block_size=32,
                    quant_dtype=self.npu_ops.activation_dtype,
                )
            else:
                # Block-128 has no grouped_dynamic_mx_quant contract.  Quantize
                # along the token axis and insert the expert-boundary rows in
                # the scale tensor consumed by group_type=2 GMM.
                col_data, col_scale = self.npu_ops.dynamic_mx_quant(
                    tensor,
                    axis=-2,
                    block_size=self.block_size,
                    quant_dtype=self.npu_ops.activation_dtype,
                )
                if boundaries is not None:
                    col_scale = transform_grouped_scale(
                        tensor,
                        col_scale,
                        boundaries,
                        self.block_size,
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

    def _quantize_weight(
        self,
        tensor: torch.Tensor,
        *,
        rowwise: bool,
        colwise: bool,
    ) -> MXFP8Tensor:
        if tensor.ndim != 3:
            raise ValueError(
                "Fake W4A8 GMM weights must be [experts,K,N], "
                f"got shape {tuple(tensor.shape)}."
            )
        if any(dimension % 32 for dimension in tensor.shape[1:]):
            raise ValueError(
                "Fake MXFP4 weight dimensions must be divisible by 32, "
                f"got shape {tuple(tensor.shape)}."
            )
        if any(dimension % self.block_size for dimension in tensor.shape[1:]):
            raise ValueError(
                "Fake MXFP8 GMM dimensions must be divisible by block_size, "
                f"got shape {tuple(tensor.shape)}, block_size={self.block_size}."
            )
        row_data = row_scale = col_data = col_scale = None

        def _quantize_source(source: torch.Tensor):
            packed, packed_scale = self.npu_ops.dynamic_mx_quant(
                source,
                axis=-1,
                block_size=32,
                quant_dtype=self.npu_ops.weight_dtype,
                round_mode="rint",
                scale_alg=0,
            )
            return fake_mxfp4_to_mxfp8(
                packed,
                packed_scale,
                mx8_block_h=self.block_size,
                mx8_block_w=self.block_size,
            )

        # Block-128 uses the A5 G-B contract directly: the forward view is
        # generated from the [E,K,N] source and dgrad consumes its transposed
        # view.  Its FP32 square scale grid is exactly what split_item=2
        # grouped matmul expects.
        if self.block_size == 128:
            fp8_data, fp8_grid = _quantize_source(tensor)
            if rowwise:
                row_data = fp8_data
                row_scale = fp8_grid.to(torch.float32)
            if colwise:
                col_data = fp8_data.transpose(-2, -1).contiguous()
                col_scale = fp8_grid.transpose(1, 2).contiguous().to(torch.float32)
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

        # The ordinary MXFP8 GMM contract has two directional views:
        # row-wise data/scales for the transposed-right dgrad and column-wise
        # data/scales for the forward NN matmul.  Quantize each view along its
        # actual contracting axis instead of reusing a transposed scale grid;
        # the latter makes the A5 kernel see mismatched weight/scale
        # transposition flags for NT.
        if rowwise:
            row_data, row_grid = _quantize_source(tensor)
            if self.block_size == 32:
                row_scale = expand_scale_to_per_row(row_grid)
            else:
                row_scale = row_grid.to(torch.float32)
        if colwise:
            col_fp8, col_grid = _quantize_source(tensor.transpose(-2, -1).contiguous())
            col_data = col_fp8.transpose(-2, -1).contiguous()
            if self.block_size == 32:
                col_grid = col_grid.permute(0, 2, 1).contiguous()
                col_scale = convert_mx_scale_to_axis2(
                    col_grid,
                    tuple(tensor.shape),
                )
            else:
                col_scale = col_grid.transpose(1, 2).contiguous().to(torch.float32)
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

    def quantize(
        self,
        tensor: torch.Tensor,
        *,
        rowwise: bool = False,
        colwise: bool = False,
        group_list: Optional[torch.Tensor] = None,
        group_list_type: int = 0,
        role: str = "activation",
    ) -> MXFP8Tensor:
        if not isinstance(tensor, torch.Tensor) or isinstance(tensor, QuantizedTensor):
            raise TypeError("FakeW4A8Quantizer expects a high-precision Tensor.")
        if not rowwise and not colwise:
            raise ValueError("FakeW4A8Quantizer requires rowwise or colwise=True.")
        if group_list_type not in (0, 1):
            raise ValueError("Fake W4A8 group_list_type must be 0 or 1.")
        if group_list is not None and (
            not isinstance(group_list, torch.Tensor) or group_list.ndim != 1
        ):
            raise ValueError("Fake W4A8 group_list must be one-dimensional.")
        if role == "weight":
            if group_list is not None:
                raise ValueError("Fake W4A8 weight quantization does not accept groups.")
            return self._quantize_weight(tensor, rowwise=rowwise, colwise=colwise)
        if role != "activation" or tensor.ndim != 2:
            raise ValueError("Fake W4A8 activation quantization expects a 2D Tensor.")
        return self._quantize_activation(
            tensor,
            rowwise=rowwise,
            colwise=colwise,
            group_list=group_list,
            group_list_type=group_list_type,
        )

    def quantize_activation(self, tensor: torch.Tensor, **kwargs) -> MXFP8Tensor:
        return self.quantize(tensor, role="activation", **kwargs)

    def quantize_weight(self, tensor: torch.Tensor, **kwargs) -> MXFP8Tensor:
        return self.quantize(tensor, role="weight", **kwargs)


__all__ = ["FakeW4A8Quantizer"]
