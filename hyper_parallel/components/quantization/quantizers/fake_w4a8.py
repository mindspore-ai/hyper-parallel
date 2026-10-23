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

from typing import Any, Optional

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.quantization.config import (
    GROUPED_NDIM,
    GROUP_LIST_COUNTS,
    GROUP_LIST_OFFSETS,
    MATRIX_NDIM,
    MX_BLOCK_SIZE_32,
    MX_SCALE_ALG_OCP,
    MX_SCALE_TILE,
    validate_mx_block_size,
)
from hyper_parallel.components.quantization.ops.npu_fake_w4a8 import (
    _get_fake_w4a8_npu_ops,
    FakeW4A8NpuOps,
)
from hyper_parallel.components.quantization.quantizers.base import Quantizer
from hyper_parallel.components.quantization.tensor import MXFP8Tensor, QuantizedTensor


class FakeW4A8Quantizer(Quantizer):
    """Build fake MXFP4 weights and MXFP8 activation/backward operands."""

    def __init__(
        self,
        block_size: int = MX_BLOCK_SIZE_32,
    ) -> None:
        """Create a fake-W4A8 quantizer for one supported MX block size."""
        self.block_size = validate_mx_block_size(block_size)

    @property
    def npu_ops(self) -> FakeW4A8NpuOps:
        """Return the process-local fake-W4A8 NPU adapter."""
        return _get_fake_w4a8_npu_ops()

    @staticmethod
    def _boundaries(
        group_list: Optional[torch.Tensor],
        group_list_type: int,
    ) -> Optional[torch.Tensor]:
        if group_list is None:
            return None
        return (
            group_list
            if group_list_type == GROUP_LIST_OFFSETS
            else torch.cumsum(group_list, dim=0)
        )

    def _quantize_activation(
        self,
        tensor: torch.Tensor,
        *,
        rowwise: bool,
        colwise: bool,
        group_list: Optional[torch.Tensor],
        group_list_type: int,
    ) -> MXFP8Tensor:
        """Quantize an activation in the requested row/column directions."""
        boundaries = self._boundaries(group_list, group_list_type)
        row_data, row_scale, col_data, col_scale = self._quantize_activation_block32(
            tensor,
            rowwise=rowwise,
            colwise=colwise,
            boundaries=boundaries,
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

    def _quantize_activation_axis(
        self,
        tensor: torch.Tensor,
        *,
        axis: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the shared dynamic-MX activation quantization operation."""

        return self.npu_ops.dynamic_mx_quant(
            tensor,
            axis=axis,
            block_size=self.block_size,
            quant_dtype=self.npu_ops.activation_dtype,
        )

    def _quantize_activation_block32(
        self,
        tensor: torch.Tensor,
        *,
        rowwise: bool,
        colwise: bool,
        boundaries: Optional[torch.Tensor],
    ) -> tuple[
        Optional[torch.Tensor], Optional[torch.Tensor],
        Optional[torch.Tensor], Optional[torch.Tensor],
    ]:
        """Quantize block-32 activations, including the grouped TN wgrad view."""

        row_data = row_scale = col_data = col_scale = None
        if rowwise:
            row_data, row_scale = self._quantize_activation_axis(tensor, axis=-1)
        if colwise and boundaries is not None:
            # The TN weight-gradient path needs grouped quantization here.
            # Its block-32 operator accepts cumulative boundaries as int32.
            col_data, col_scale = self.npu_ops.grouped_dynamic_mx_quant(
                tensor,
                boundaries,
                block_size=MX_BLOCK_SIZE_32,
                quant_dtype=self.npu_ops.activation_dtype,
            )
        elif colwise:
            col_data, col_scale = self._quantize_activation_axis(tensor, axis=-2)
        return row_data, row_scale, col_data, col_scale

    def _quantize_weight(
        self,
        tensor: torch.Tensor,
        *,
        rowwise: bool,
        colwise: bool,
    ) -> MXFP8Tensor:
        """Quantize a grouped master weight through the fake W4A8 path."""
        if tensor.ndim != GROUPED_NDIM:
            raise ValueError(
                "Fake W4A8 GMM weights must be [experts,K,N], "
                f"got shape {tuple(tensor.shape)}."
            )
        if any(dimension % MX_BLOCK_SIZE_32 for dimension in tensor.shape[1:]):
            raise ValueError(
                "Fake MXFP4 weight dimensions must be divisible by 32, "
                f"got shape {tuple(tensor.shape)}."
            )
        if any(dimension % self.block_size for dimension in tensor.shape[1:]):
            raise ValueError(
                "Fake MXFP8 GMM dimensions must be divisible by block_size, "
                f"got shape {tuple(tensor.shape)}, block_size={self.block_size}."
            )
        source, transposed_source = self._prepare_block32_weight_source(tensor)

        # Both layouts start by generating the same packed MXFP4 representation.
        # Only the source orientation and the following MXFP8 conversion differ.
        packed_mxfp4, mxfp4_scale = self._quantize_mxfp4(source)
        row_data, row_scale, col_data, col_scale = self._convert_weight_block32(
            packed_mxfp4,
            mxfp4_scale,
            transposed_source=transposed_source,
            rowwise=rowwise,
            colwise=colwise,
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

    def _quantize_mxfp4(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize one source orientation to packed MXFP4 and E8M0 scales."""

        return self.npu_ops.dynamic_mx_quant(
            tensor,
            axis=-1,
            block_size=MX_BLOCK_SIZE_32,
            quant_dtype=self.npu_ops.weight_dtype,
            round_mode="rint",
            scale_alg=MX_SCALE_ALG_OCP,
        )

    def _prepare_block32_weight_source(
        self,
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, bool]:
        """Select the MXFP4 source orientation required by the CANN converter."""

        rows, cols = tensor.shape[-2:]
        # CANN requires its penultimate axis to be 64-aligned. Prefer [E,K,N]
        # so MXFP4 quantization stays on N; transpose only when K is misaligned.
        transposed_source = rows % MX_SCALE_TILE != 0
        if transposed_source and cols % MX_SCALE_TILE != 0:
            raise ValueError(
                "Fake W4A8 block_size=32 requires at least one of the weight "
                "matrix dimensions to be divisible by 64 for "
                "the CANN MXFP4-to-block-MXFP8 conversion; "
                f"got shape {tuple(tensor.shape)}."
            )
        if transposed_source:
            return tensor.transpose(-2, -1).contiguous(), True
        return tensor, False

    def _convert_weight_block32(
        self,
        packed_mxfp4: torch.Tensor,
        mxfp4_scale: torch.Tensor,
        *,
        transposed_source: bool,
        rowwise: bool,
        colwise: bool,
    ) -> tuple[
        Optional[torch.Tensor], Optional[torch.Tensor],
        Optional[torch.Tensor], Optional[torch.Tensor],
    ]:
        """Convert packed MXFP4 through CANN's block-MXFP8 operator."""

        (
            fp8_data,
            scale1,
            scale2,
        ) = self.npu_ops.convert_packed_mxfp4_to_block_mxfp8(
            packed_mxfp4,
            mxfp4_scale,
        )

        if transposed_source:
            # For source [E,N,K], scale1 follows K (the original contracting
            # axis) and scale2 follows N. Map them back to [E,K,N] GMM views.
            fp8_data = fp8_data.transpose(-2, -1).contiguous()
            row_scale = scale2.transpose(1, 2).contiguous()
            col_scale = scale1.transpose(1, 2).contiguous()
        else:
            # For source [E,K,N], scale1 is [E,K,ceil(N/64),2] and scale2 is
            # [E,ceil(K/64),N,2], the GMM's rowwise and colwise layouts.
            row_scale = scale1
            col_scale = scale2
        return (
            fp8_data if rowwise else None,
            row_scale if rowwise else None,
            fp8_data if colwise else None,
            col_scale if colwise else None,
        )

    def quantize(
        self,
        tensor: torch.Tensor,
        *,
        rowwise: bool = False,
        colwise: bool = False,
        group_list: Optional[torch.Tensor] = None,
        group_list_type: int = GROUP_LIST_OFFSETS,
        role: str = "activation",
    ) -> MXFP8Tensor:
        """Build directional fake-MX storage for an activation or weight."""
        if not isinstance(tensor, torch.Tensor) or isinstance(tensor, QuantizedTensor):
            raise TypeError("FakeW4A8Quantizer expects a high-precision Tensor.")
        if not rowwise and not colwise:
            raise ValueError("FakeW4A8Quantizer requires rowwise or colwise=True.")
        if group_list_type not in (GROUP_LIST_OFFSETS, GROUP_LIST_COUNTS):
            raise ValueError("Fake W4A8 group_list_type must be 0 or 1.")
        if group_list is not None and (
            not isinstance(group_list, torch.Tensor) or group_list.ndim != 1
        ):
            raise ValueError("Fake W4A8 group_list must be one-dimensional.")
        if role == "weight":
            if group_list is not None:
                raise ValueError("Fake W4A8 weight quantization does not accept groups.")
            return self._quantize_weight(tensor, rowwise=rowwise, colwise=colwise)
        if role != "activation" or tensor.ndim != MATRIX_NDIM:
            raise ValueError("Fake W4A8 activation quantization expects a 2D Tensor.")
        return self._quantize_activation(
            tensor,
            rowwise=rowwise,
            colwise=colwise,
            group_list=group_list,
            group_list_type=group_list_type,
        )

    def quantize_activation(
        self, tensor: torch.Tensor, **kwargs: Any
    ) -> MXFP8Tensor:
        """Quantize a two-dimensional activation for fake-W4A8 GMM."""
        return self.quantize(tensor, role="activation", **kwargs)

    def quantize_weight(
        self, tensor: torch.Tensor, **kwargs: Any
    ) -> MXFP8Tensor:
        """Quantize grouped weights through the fake-MXFP4 path."""
        return self.quantize(tensor, role="weight", **kwargs)


__all__ = ["FakeW4A8Quantizer"]
