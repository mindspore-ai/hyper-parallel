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
"""Pangu-compatible directional storage for HiFloat8 tensors."""

from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.quantization.tensor.base import (
    QuantizedTensorStorage,
)
from hyper_parallel.components.quantization.tensor.quantized_tensor import (
    QuantizedTensor,
)

if TYPE_CHECKING:
    from hyper_parallel.components.quantization.quantizers.hifloat8 import (
        HiFloat8Quantizer,
    )


class HiFloat8TensorStorage(QuantizedTensorStorage):
    """Store Pangu-compatible row/column views of one HiFloat8 payload."""

    shape: Iterable[int]
    dtype: torch.dtype
    quantizer: "HiFloat8Quantizer"
    row_data: Optional[torch.Tensor]
    col_data: Optional[torch.Tensor]
    row_scale: Optional[torch.Tensor]
    col_scale: Optional[torch.Tensor]
    scale: Optional[torch.Tensor]

    def __init__(
        self,
        shape: Iterable[int],
        dtype: torch.dtype,
        *,
        quantizer: "HiFloat8Quantizer",
        row_data: Optional[torch.Tensor] = None,
        col_data: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
        row_scale: Optional[torch.Tensor] = None,
        col_scale: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialize directional fields using Pangu's HiF8 storage contract."""

        self.row_data = row_data
        self.col_data = col_data
        self.row_scale = row_scale if row_scale is not None else (
            scale if self.row_data is not None else None
        )
        self.col_scale = col_scale if col_scale is not None else (
            scale if self.col_data is not None else None
        )
        self.scale = (
            scale
            if scale is not None
            else self.row_scale
            if self.row_scale is not None
            else self.col_scale
        )
        if not hasattr(self, "shape"):
            self.shape = shape
            self.dtype = dtype
        self.quantizer = quantizer

    def update_usage(self, rowwise: bool = True, colwise: bool = True) -> None:
        """Release row/column references independently, matching Pangu."""

        if not rowwise:
            self.row_data = None
            self.row_scale = None
        if not colwise:
            self.col_data = None
            self.col_scale = None
        self.scale = self.row_scale if self.row_scale is not None else self.col_scale

    def is_rowwise(self) -> bool:
        """Return whether row-wise HiFloat8 data and scale are available."""

        return self.row_data is not None

    def is_colwise(self) -> bool:
        """Return whether column-wise HiFloat8 data and scale are available."""

        return self.col_data is not None

    def get_metadata(self) -> dict[str, Any]:
        """Return the same directional metadata fields as Pangu HiF8Tensor."""

        return {
            "row_data": self.row_data,
            "row_scale": self.row_scale,
            "col_data": self.col_data,
            "col_scale": self.col_scale,
            "quantizer": self.quantizer,
        }


class HiFloat8Tensor(HiFloat8TensorStorage, QuantizedTensor):
    """Expose Pangu-compatible HiFloat8 storage with a logical source dtype."""

    def __new__(
        cls,
        shape: Iterable[int],
        dtype: torch.dtype,
        *,
        quantizer: "HiFloat8Quantizer",
        row_data: Optional[torch.Tensor] = None,
        col_data: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
        row_scale: Optional[torch.Tensor] = None,
        col_scale: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
    ) -> "HiFloat8Tensor":
        """Create a wrapper from at least one directional HiFloat8 view."""

        del quantizer, scale, row_scale, col_scale
        physical_data = row_data if row_data is not None else col_data
        if physical_data is None:
            raise ValueError("HiFloat8Tensor requires row_data or col_data.")
        wrapper_device = physical_data.device if device is None else device
        return QuantizedTensor.__new__(
            cls,
            shape,
            dtype,
            device=wrapper_device,
            requires_grad=requires_grad,
        )

    def __init__(
        self,
        shape: Iterable[int],
        dtype: torch.dtype,
        *,
        quantizer: "HiFloat8Quantizer",
        row_data: Optional[torch.Tensor] = None,
        col_data: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
        row_scale: Optional[torch.Tensor] = None,
        col_scale: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
    ) -> None:
        """Initialize directional storage after wrapper allocation."""

        del device, requires_grad
        HiFloat8TensorStorage.__init__(
            self,
            shape,
            dtype,
            quantizer=quantizer,
            row_data=row_data,
            col_data=col_data,
            scale=scale,
            row_scale=row_scale,
            col_scale=col_scale,
        )

    def _new_with_transformed_data(
        self,
        transform: Callable[[torch.Tensor], torch.Tensor],
        *,
        requires_grad: bool,
    ) -> "HiFloat8Tensor":
        transformed: dict[int, torch.Tensor] = {}

        def _transform_optional(
            value: Optional[torch.Tensor],
        ) -> Optional[torch.Tensor]:
            if value is None:
                return None
            key = id(value)
            if key not in transformed:
                transformed[key] = transform(value)
            return transformed[key]

        return type(self)(
            shape=self.shape,
            dtype=self.dtype,
            quantizer=self.quantizer,
            row_data=_transform_optional(self.row_data),
            row_scale=_transform_optional(self.row_scale),
            col_data=_transform_optional(self.col_data),
            col_scale=_transform_optional(self.col_scale),
            requires_grad=requires_grad,
        )

    def __repr__(self) -> str:
        """Return logical shape and directional storage availability."""

        return (
            "HiFloat8Tensor("
            f"shape={tuple(self.shape)}, dtype={self.dtype}, "
            f"device={self.device}, rowwise={self.is_rowwise()}, "
            f"colwise={self.is_colwise()})"
        )
