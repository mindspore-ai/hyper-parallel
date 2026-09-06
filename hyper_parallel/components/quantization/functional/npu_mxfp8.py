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

from hyper_parallel.components.quantization.tensor import MXFP8Tensor

_A5_DEVICE_MARKERS = ("Ascend950", "Ascend910_95")
_SUPPORTED_LAYOUTS = ("NN", "NT", "TN")


class LowPrecisionCapabilityError(RuntimeError):
    """Report that the active torch_npu stack lacks the A5 MXFP8 contract."""


class MXFP8NpuOps:
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
            raise LowPrecisionCapabilityError(
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
            raise LowPrecisionCapabilityError(
                "MXFP8 is supported only on Ascend 950PR/950DT (A5), "
                f"but the active device is {device_name or 'unknown'}."
            )
        self._torch_npu = torch_npu_module

    def validate_grouped_matmul(self) -> None:
        """Fail when the runtime lacks either MXFP8 GMM-specific operator."""

        required = (
            "npu_grouped_dynamic_mx_quant",
            "npu_grouped_matmul",
        )
        missing = [
            name for name in required if not hasattr(self._torch_npu, name)
        ]
        if missing:
            version = getattr(self._torch_npu, "__version__", "unknown")
            raise LowPrecisionCapabilityError(
                "The active torch_npu stack does not provide the A5 MXFP8 "
                f"GMM contract; missing {missing}, torch_npu={version}."
            )

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

    def grouped_dynamic_mx_quant(
        self,
        tensor: torch.Tensor,
        group_list: torch.Tensor,
        *,
        quant_dtype: Optional[torch.dtype] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize the column direction independently for each expert."""

        grouped_quant = getattr(
            self._torch_npu,
            "npu_grouped_dynamic_mx_quant",
            None,
        )
        if grouped_quant is None:
            version = getattr(self._torch_npu, "__version__", "unknown")
            raise LowPrecisionCapabilityError(
                "The active torch_npu stack does not provide "
                "npu_grouped_dynamic_mx_quant, which MXFP8 GMM requires; "
                f"torch_npu={version}."
            )
        return grouped_quant(
            tensor,
            group_list.to(torch.int32),
            round_mode="rint",
            dst_type=quant_dtype or torch.float8_e4m3fn,
            blocksize=32,
            scale_alg=1,
        )

    def quant_grouped_matmul(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x2_scale: torch.Tensor,
        *,
        x1_scale: torch.Tensor,
        group_list: torch.Tensor,
        group_type: int,
        group_list_type: int,
        output_dtype: torch.dtype,
        bias: Optional[list[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Execute one A5 MXFP8 grouped matrix multiplication."""

        grouped_matmul = getattr(self._torch_npu, "npu_grouped_matmul", None)
        if grouped_matmul is None:
            version = getattr(self._torch_npu, "__version__", "unknown")
            raise LowPrecisionCapabilityError(
                "The active torch_npu stack does not provide "
                f"npu_grouped_matmul; torch_npu={version}."
            )
        return grouped_matmul(
            [x1],
            [x2],
            bias=[] if bias is None else bias,
            scale=[x2_scale],
            per_token_scale=[x1_scale],
            group_list=group_list,
            group_type=group_type,
            output_dtype=output_dtype,
            group_list_type=group_list_type,
            scale_dtype=self._torch_npu.float8_e8m0fnu,
            per_token_scale_dtype=self._torch_npu.float8_e8m0fnu,
            split_item=2,
        )[0]

    def is_e8m0_dtype(self, dtype: torch.dtype) -> bool:
        """Return whether a scale dtype is this runtime's E8M0 representation."""
        return dtype == self._torch_npu.float8_e8m0fnu


@lru_cache(maxsize=1)
def _get_npu_ops() -> MXFP8NpuOps:
    """Return the process-local A5 MXFP8 operator adapter."""

    try:
        import torch_npu  # pylint: disable=C0415
    except ImportError as exc:
        raise LowPrecisionCapabilityError(
            "A5 MXFP8 requires torch_npu, but it is not importable."
        ) from exc
    return MXFP8NpuOps(torch_npu)


def validate_npu_runtime() -> None:
    """Fail during model setup when the MXFP8 runtime is unavailable."""

    _get_npu_ops()


def validate_npu_gmm_runtime() -> None:
    """Fail during expert setup when the MXFP8 GMM runtime is unavailable."""

    _get_npu_ops().validate_grouped_matmul()


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


def mxfp8_grouped_matmul(
    left: MXFP8Tensor,
    right: MXFP8Tensor,
    *,
    layout: str,
    group_list: torch.Tensor,
    group_type: int = 0,
    group_list_type: int = 0,
    output_dtype: Optional[torch.dtype] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Multiply expert-grouped MXFP8 operands and return a high-precision Tensor.

    The layout follows ``Y = op(left) @ op(right)``. Forward and input-gradient
    GMM split the M axis with ``group_type=0``; weight-gradient GMM splits the
    contracting axis with ``group_type=2``.

    Args:
        left: Quantized left operand.
        right: Quantized right operand.
        layout: One of ``"NN"``, ``"NT"``, or ``"TN"``.
        group_list: Expert boundaries or per-expert token counts.
        group_type: Axis grouped by the NPU kernel.
        group_list_type: ``0`` for cumulative boundaries or ``1`` for counts.
        output_dtype: Optional high-precision output dtype.
        bias: Optional grouped bias tensor.

    Returns:
        The grouped matrix-multiplication result.
    """

    if not isinstance(left, MXFP8Tensor) or not isinstance(right, MXFP8Tensor):
        raise TypeError("mxfp8_grouped_matmul requires two MXFP8Tensor operands.")
    if layout not in _SUPPORTED_LAYOUTS:
        raise ValueError(
            f"Unsupported MXFP8 GMM layout {layout!r}; "
            f"expected one of {_SUPPORTED_LAYOUTS}."
        )
    if not isinstance(group_list, torch.Tensor) or group_list.ndim != 1:
        raise ValueError("MXFP8 GMM group_list must be a one-dimensional Tensor.")
    if group_type not in (0, 2):
        raise ValueError(
            f"MXFP8 GMM group_type must be 0 or 2, but got {group_type}."
        )
    if group_list_type not in (0, 1):
        raise ValueError(
            "MXFP8 GMM group_list_type must be 0 or 1, "
            f"but got {group_list_type}."
        )

    left_data, left_scale = _left_operand(
        left,
        transpose=layout[0] == "T",
    )
    right_data, right_scale = _right_operand(
        right,
        transpose=layout[1] == "T",
    )
    bias_list = None if bias is None else [bias]
    return left.quantizer.npu_ops.quant_grouped_matmul(
        left_data,
        right_data,
        right_scale,
        x1_scale=left_scale,
        group_list=group_list,
        group_type=group_type,
        group_list_type=group_list_type,
        output_dtype=output_dtype or left.dtype,
        bias=bias_list,
    )
