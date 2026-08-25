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
"""MXFP8 directional storage and logical Tensor wrapper."""

from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.auto_models.components.training.low_precision.tensor.base import (
    QuantizedTensorStorage,
)
from hyper_parallel.auto_models.components.training.low_precision.tensor.quantized_tensor import (
    QuantizedTensor,
)

if TYPE_CHECKING:
    from hyper_parallel.auto_models.components.training.low_precision.quantizers.mxfp8 import (
        MXFP8Quantizer,
    )


class MXFP8TensorStorage(QuantizedTensorStorage):
    """Store row-wise and column-wise MXFP8 data with paired scales."""

    def __init__(
        self,
        shape: Iterable[int],
        dtype: torch.dtype,
        *,
        quantizer: "MXFP8Quantizer",
        row_data: Optional[torch.Tensor] = None,
        row_scale: Optional[torch.Tensor] = None,
        col_data: Optional[torch.Tensor] = None,
        col_scale: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialize one typed MXFP8 representation."""

        self._validate_pair("row", row_data, row_scale)
        self._validate_pair("column", col_data, col_scale)
        if row_data is None and col_data is None:
            raise ValueError(
                "MXFP8TensorStorage requires a row-wise or column-wise "
                "representation."
            )
        if not isinstance(self, torch.Tensor):
            self.shape = tuple(shape)
            self.dtype = dtype
        self.quantizer = quantizer
        self.row_data = row_data
        self.row_scale = row_scale
        self.col_data = col_data
        self.col_scale = col_scale

    @staticmethod
    def _validate_pair(
        direction: str,
        data: Optional[torch.Tensor],
        scale: Optional[torch.Tensor],
    ) -> None:
        if (data is None) != (scale is None):
            raise ValueError(
                f"MXFP8 {direction}-wise data and scale must either both be "
                "present or both be absent."
            )

    def update_usage(self, rowwise: bool = True, colwise: bool = True) -> None:
        """Release directional data that subsequent computation will not use."""

        if not rowwise:
            self.row_data = None
            self.row_scale = None
        if not colwise:
            self.col_data = None
            self.col_scale = None

    def is_rowwise(self) -> bool:
        """Return whether row-wise data and scale are available."""

        return self.row_data is not None and self.row_scale is not None

    def is_colwise(self) -> bool:
        """Return whether column-wise data and scale are available."""

        return self.col_data is not None and self.col_scale is not None

    def get_metadata(self) -> dict[str, Any]:
        """Return directional storage and quantizer metadata."""

        return {
            "row_data": self.row_data,
            "row_scale": self.row_scale,
            "col_data": self.col_data,
            "col_scale": self.col_scale,
            "quantizer": self.quantizer,
        }


class MXFP8Tensor(MXFP8TensorStorage, QuantizedTensor):
    """Expose MXFP8 storage as a logical Tensor with the original dtype."""

    def __new__(
        cls,
        shape: Iterable[int],
        dtype: torch.dtype,
        *,
        quantizer: "MXFP8Quantizer",
        row_data: Optional[torch.Tensor] = None,
        row_scale: Optional[torch.Tensor] = None,
        col_data: Optional[torch.Tensor] = None,
        col_scale: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
    ) -> "MXFP8Tensor":
        """Create an MXFP8 wrapper from at least one directional pair."""

        del quantizer, row_scale, col_scale
        physical_data = row_data if row_data is not None else col_data
        if physical_data is None:
            raise ValueError("MXFP8Tensor requires row_data or col_data.")
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
        quantizer: "MXFP8Quantizer",
        row_data: Optional[torch.Tensor] = None,
        row_scale: Optional[torch.Tensor] = None,
        col_data: Optional[torch.Tensor] = None,
        col_scale: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
    ) -> None:
        """Initialize directional storage after wrapper allocation."""

        del device, requires_grad
        MXFP8TensorStorage.__init__(
            self,
            shape,
            dtype,
            quantizer=quantizer,
            row_data=row_data,
            row_scale=row_scale,
            col_data=col_data,
            col_scale=col_scale,
        )

    def _new_with_transformed_data(
        self,
        transform: Callable[[torch.Tensor], torch.Tensor],
        *,
        requires_grad: bool,
    ) -> "MXFP8Tensor":
        def _transform_optional(
            value: Optional[torch.Tensor],
        ) -> Optional[torch.Tensor]:
            return None if value is None else transform(value)

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
        """Return logical shape and directional storage state."""

        return (
            "MXFP8Tensor("
            f"shape={tuple(self.shape)}, dtype={self.dtype}, "
            f"device={self.device}, rowwise={self.is_rowwise()}, "
            f"colwise={self.is_colwise()})"
        )
