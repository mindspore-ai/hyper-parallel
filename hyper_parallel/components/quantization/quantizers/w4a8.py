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
"""Native W4A8 quantizer for grouped expert training."""

from typing import Optional

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.quantization.ops.npu_w4a8 import (
    transform_grouped_scale,
    W4A8NpuOps,
    _get_w4a8_npu_ops,
)
from hyper_parallel.components.quantization.quantizers.base import Quantizer
from hyper_parallel.components.quantization.tensor import (
    QuantizedTensor,
    W4A8Tensor,
)

_SUPPORTED_BLOCK_SIZES = (32, 128)
_ROLE_NAMES = ("activation", "weight")


class W4A8Quantizer(Quantizer):
    """Quantize MXFP4 weights and MXFP8 activations for native GMM."""

    def __init__(
        self,
        block_size: int = 32,
    ) -> None:
        """Create a block-size-specific native W4A8 quantizer."""

        if block_size not in _SUPPORTED_BLOCK_SIZES:
            raise ValueError(
                f"Native W4A8 block_size must be 32 or 128, got {block_size}."
            )
        self.block_size = block_size

    @property
    def npu_ops(self) -> W4A8NpuOps:
        """Return the process-local native W4A8 ops adapter."""

        return _get_w4a8_npu_ops()

    @staticmethod
    def _validate_group_list(
        group_list: Optional[torch.Tensor],
        group_list_type: int,
    ) -> None:
        if group_list_type not in (0, 1):
            raise ValueError(
                "W4A8 group_list_type must be 0 or 1, "
                f"but got {group_list_type}."
            )
        if group_list is not None and (
            not isinstance(group_list, torch.Tensor) or group_list.ndim != 1
        ):
            raise ValueError("W4A8 group_list must be a one-dimensional Tensor.")

    @staticmethod
    def _boundaries(
        group_list: Optional[torch.Tensor],
        group_list_type: int,
    ) -> Optional[torch.Tensor]:
        if group_list is None:
            return None
        return group_list if group_list_type == 0 else torch.cumsum(group_list, dim=0)

    def _quantize_activation(
        self,
        tensor: torch.Tensor,
        *,
        rowwise: bool,
        colwise: bool,
        group_list: Optional[torch.Tensor],
        group_list_type: int,
    ) -> W4A8Tensor:
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
                row_scale = self._scale_to_fp32(row_scale).reshape(
                    tensor.shape[0],
                    -1,
                )
        if colwise:
            if self.block_size == 32 and boundaries is not None:
                col_data, col_scale = self.npu_ops.grouped_dynamic_mx_quant(
                    tensor,
                    boundaries,
                    block_size=32,
                    quant_dtype=self.npu_ops.activation_dtype,
                )
            else:
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
        return W4A8Tensor(
            shape=tensor.shape,
            dtype=tensor.dtype,
            quantizer=self,
            row_data=row_data,
            row_scale=row_scale,
            col_data=col_data,
            col_scale=col_scale,
            device=tensor.device,
        )

    @staticmethod
    def _scale_to_fp32(scale: torch.Tensor) -> torch.Tensor:
        """Convert an E8M0 exponent tensor to the block-128 FP32 contract."""

        return torch.pow(2.0, scale.to(torch.float32) - 127)

    def _quantize_weight(
        self,
        tensor: torch.Tensor,
        *,
        rowwise: bool,
        colwise: bool,
    ) -> W4A8Tensor:
        if any(dimension % 32 for dimension in tensor.shape[1:]):
            raise ValueError(
                "Native MXFP4 weight dimensions must be divisible by 32, "
                f"got shape {tuple(tensor.shape)}."
            )
        row_data = row_scale = col_data = col_scale = None
        if rowwise:
            # Forward GMM consumes [E, K/2, N] packed NZ data. Quantizing the
            # live [E, N, K] source along K gives that native contract.
            row_data, row_scale = self.npu_ops.pack_mxfp4_weight(
                tensor.transpose(-2, -1).contiguous(),
                block_size=32,
            )
        if colwise:
            # Dgrad uses the other matrix direction, so its FP4 scales must be
            # generated independently from the [E, K, N] source.
            col_data, col_scale = self.npu_ops.pack_mxfp4_weight(
                tensor,
                block_size=32,
            )
        return W4A8Tensor(
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
    ) -> W4A8Tensor:
        """Quantize an activation or target-layout expert weight.

        Args:
            tensor: High-precision source tensor.
            rowwise: Keep the forward representation.
            colwise: Keep the representation required by backward.
            group_list: Optional cumulative expert boundaries or token counts.
            group_list_type: ``0`` for boundaries and ``1`` for token counts.
            role: ``"weight"`` quantizes a 3D MXFP4 expert weight with no
                grouping; ``"activation"`` quantizes a 2D activation with
                optional grouped boundaries.
        """

        if not isinstance(tensor, torch.Tensor) or isinstance(tensor, QuantizedTensor):
            raise TypeError("W4A8Quantizer expects a high-precision torch.Tensor.")
        if not rowwise and not colwise:
            raise ValueError("W4A8Quantizer requires rowwise=True or colwise=True.")
        if role not in _ROLE_NAMES:
            raise ValueError(f"Unsupported W4A8 quantization role {role!r}.")
        self._validate_group_list(group_list, group_list_type)
        if role == "weight":
            if tensor.ndim != 3:
                raise ValueError(
                    "Native W4A8 weights must be three-dimensional, "
                    f"got shape {tuple(tensor.shape)}."
                )
            if group_list is not None:
                raise ValueError("W4A8 weight quantization does not accept group_list.")
            return self._quantize_weight(
                tensor,
                rowwise=rowwise,
                colwise=colwise,
            )
        if tensor.ndim != 2:
            raise ValueError(
                "Native W4A8 activations must be two-dimensional, "
                f"got shape {tuple(tensor.shape)}."
            )
        return self._quantize_activation(
            tensor,
            rowwise=rowwise,
            colwise=colwise,
            group_list=group_list,
            group_list_type=group_list_type,
        )

    def quantize_activation(
        self,
        tensor: torch.Tensor,
        *,
        rowwise: bool = False,
        colwise: bool = False,
        group_list: Optional[torch.Tensor] = None,
        group_list_type: int = 0,
    ) -> W4A8Tensor:
        """Quantize an MXFP8 activation representation."""

        return self.quantize(
            tensor,
            rowwise=rowwise,
            colwise=colwise,
            group_list=group_list,
            group_list_type=group_list_type,
            role="activation",
        )

    def quantize_weight(
        self,
        tensor: torch.Tensor,
        *,
        rowwise: bool = False,
        colwise: bool = False,
    ) -> W4A8Tensor:
        """Quantize an expert weight into native Packed MXFP4 views."""

        return self.quantize(
            tensor,
            rowwise=rowwise,
            colwise=colwise,
            role="weight",
        )


__all__ = [
    "W4A8Quantizer",
]
