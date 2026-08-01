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
"""A5 MXFP8 quantization and matrix-multiplication operator adapters."""

from functools import lru_cache
from typing import Any, Optional

import torch  # pylint: disable=forbidden-backend-import

from hyper_models.components.training.low_precision.tensor import MXFP8Tensor

_A5_DEVICE_MARKERS = ("Ascend950", "Ascend910_95")
_SUPPORTED_LAYOUTS = ("NN", "NT", "TN")


class NpuCapabilityError(RuntimeError):
    """Report that the active torch_npu stack lacks the A5 MXFP8 contract."""


class NpuOps:
    """Adapt the A5 MXFP8 torch_npu operator contract."""

    def __init__(self, torch_npu_module: Any) -> None:
        """Validate and retain an imported torch_npu module."""

        required = (
            "npu_dynamic_mx_quant",
            "npu_dynamic_mx_quant_with_dual_axis",
            "npu_quant_matmul",
            "float8_e8m0fnu",
        )
        missing = [
            name for name in required if not hasattr(torch_npu_module, name)
        ]
        if missing:
            version = getattr(torch_npu_module, "__version__", "unknown")
            raise NpuCapabilityError(
                "The active torch_npu stack does not provide the A5 MXFP8 "
                f"operator contract; missing {missing}, torch_npu={version}."
            )
        npu_handle = getattr(torch_npu_module, "npu", None)
        device_name = (
            npu_handle.get_device_name()
            if npu_handle is not None
            and hasattr(npu_handle, "get_device_name")
            else ""
        )
        if not any(marker in device_name for marker in _A5_DEVICE_MARKERS):
            raise NpuCapabilityError(
                "MXFP8 is supported only on Ascend 950PR/950DT (A5), "
                f"but the active device is {device_name or 'unknown'}."
            )
        self._torch_npu = torch_npu_module

    def dynamic_mx_quant(
        self,
        tensor: torch.Tensor,
        *,
        axis: int,
        quant_dtype: Optional[torch.dtype] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize one tensor along an MX block axis."""

        return self._torch_npu.npu_dynamic_mx_quant(
            tensor,
            axis=axis,
            dst_type=quant_dtype or torch.float8_e4m3fn,
            scale_alg=1,
        )

    def dynamic_mx_quant_dual_axis(
        self,
        tensor: torch.Tensor,
        *,
        quant_dtype: Optional[torch.dtype] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize one tensor into row-wise and column-wise MX storage."""

        return self._torch_npu.npu_dynamic_mx_quant_with_dual_axis(
            tensor,
            dst_type=quant_dtype or torch.float8_e4m3fn,
            scale_alg=1,
        )

    def quant_matmul(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        scale: torch.Tensor,
        *,
        pertoken_scale: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Execute an A5 MXFP8 matrix multiplication."""

        return self._torch_npu.npu_quant_matmul(
            x1,
            x2,
            scale,
            pertoken_scale=pertoken_scale,
            output_dtype=output_dtype,
            scale_dtype=self._torch_npu.float8_e8m0fnu,
            pertoken_scale_dtype=self._torch_npu.float8_e8m0fnu,
            group_sizes=[1, 1, 32],
        )

    def is_e8m0_dtype(self, dtype: torch.dtype) -> bool:
        """Return whether a scale dtype is this runtime's E8M0 representation."""
        return dtype == self._torch_npu.float8_e8m0fnu


@lru_cache(maxsize=1)
def _get_npu_ops() -> NpuOps:
    """Return the process-local A5 MXFP8 operator adapter."""

    try:
        import torch_npu  # pylint: disable=C0415
    except ImportError as exc:
        raise NpuCapabilityError(
            "A5 MXFP8 requires torch_npu, but it is not importable."
        ) from exc
    return NpuOps(torch_npu)


def validate_npu_runtime() -> None:
    """Fail during model setup when the MXFP8 runtime is unavailable."""

    _get_npu_ops()


def _transpose_scale(scale: torch.Tensor) -> torch.Tensor:
    """Transpose the two matrix dimensions of an MX scale tensor."""

    if scale.ndim < 2:
        raise ValueError(
            "MXFP8 scale transpose requires at least two dimensions, got "
            f"shape {tuple(scale.shape)}."
        )
    if scale.ndim == 2:
        return scale.transpose(0, 1)
    return scale.transpose(-3, -2)


def _left_operand(
    tensor: MXFP8Tensor,
    *,
    transpose: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if transpose:
        if not tensor.is_colwise():
            raise ValueError(
                "Transposed MXFP8 left operand requires column-wise data."
            )
        return (
            tensor.col_data.transpose(-1, -2),
            _transpose_scale(tensor.col_scale),
        )
    if not tensor.is_rowwise():
        raise ValueError(
            "Non-transposed MXFP8 left operand requires row-wise data."
        )
    return tensor.row_data, tensor.row_scale


def _right_operand(
    tensor: MXFP8Tensor,
    *,
    transpose: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if transpose:
        if not tensor.is_rowwise():
            raise ValueError(
                "Transposed MXFP8 right operand requires row-wise data."
            )
        return (
            tensor.row_data.transpose(-1, -2),
            _transpose_scale(tensor.row_scale),
        )
    if not tensor.is_colwise():
        raise ValueError(
            "Non-transposed MXFP8 right operand requires column-wise data."
        )
    return tensor.col_data, tensor.col_scale


def mxfp8_matmul(
    left: MXFP8Tensor,
    right: MXFP8Tensor,
    *,
    layout: str,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Multiply typed MXFP8 operands and return a high-precision Tensor."""

    if not isinstance(left, MXFP8Tensor) or not isinstance(
        right,
        MXFP8Tensor,
    ):
        raise TypeError("mxfp8_matmul requires two MXFP8Tensor operands.")
    if layout not in _SUPPORTED_LAYOUTS:
        raise ValueError(
            f"Unsupported MXFP8 layout {layout!r}; "
            f"expected one of {_SUPPORTED_LAYOUTS}."
        )

    left_data, left_scale = _left_operand(
        left,
        transpose=layout[0] == "T",
    )
    right_data, right_scale = _right_operand(
        right,
        transpose=layout[1] == "T",
    )
    return left.quantizer.npu_ops.quant_matmul(
        left_data,
        right_data,
        right_scale,
        pertoken_scale=left_scale,
        output_dtype=output_dtype or left.dtype,
    )
