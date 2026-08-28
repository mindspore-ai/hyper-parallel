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
"""A5 HiFloat8 current-quantization, MM, and GMM adapters."""

from functools import lru_cache
from typing import Any, Optional

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.auto_models.components.training.low_precision.tensor.hifloat8_tensor import (
    HiFloat8Tensor,
)

_A5_DEVICE_MARKERS = ("Ascend950", "Ascend910_95")
_SUPPORTED_LAYOUTS = ("NN", "NT", "TN")


class HiFloat8NpuOps:
    """Adapt the verified A5 HiFloat8 torch_npu operator contract."""

    def __init__(self, torch_npu_module: Any) -> None:
        """Validate logical dtype, required operators, and target device."""

        required_operators = ("npu_dynamic_quant", "npu_quant_matmul")
        missing = [
            name
            for name in required_operators
            if not callable(getattr(torch_npu_module, name, None))
        ]
        if getattr(torch_npu_module, "hifloat8", None) is None:
            missing.append("hifloat8")
        if missing:
            version = getattr(torch_npu_module, "__version__", "unknown")
            raise RuntimeError(
                "The active torch_npu stack does not provide the A5 HiFloat8 "
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
            raise RuntimeError(
                "HiFloat8 is supported only on Ascend 950PR/950DT (A5), "
                f"but the active device is {device_name or 'unknown'}."
            )
        self._torch_npu = torch_npu_module

    @property
    def quant_dtype(self) -> Any:
        """Return the runtime's logical HiFloat8 dtype identifier."""

        return self._torch_npu.hifloat8

    def dynamic_quant(
        self,
        tensor: torch.Tensor,
        *,
        dst_type_max: float,
        quant_mode: str = "pertensor",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize one FP16/BF16 Tensor with the selected current scale."""

        return self._torch_npu.npu_dynamic_quant(
            tensor,
            dst_type=self._torch_npu.hifloat8,
            quant_mode=quant_mode,
            dst_type_max=float(dst_type_max),
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
        """Execute an A5 HiFloat8 MM with explicit logical input dtypes."""

        return self._torch_npu.npu_quant_matmul(
            x1,
            x2,
            scale,
            pertoken_scale=pertoken_scale,
            output_dtype=output_dtype,
            x1_dtype=self._torch_npu.hifloat8,
            x2_dtype=self._torch_npu.hifloat8,
        )

    def validate_grouped_matmul(self) -> None:
        """Fail when the runtime lacks the HiFloat8 GMM operator."""

        if not callable(getattr(self._torch_npu, "npu_grouped_matmul", None)):
            version = getattr(self._torch_npu, "__version__", "unknown")
            raise RuntimeError(
                "The active torch_npu stack does not provide "
                f"npu_grouped_matmul; torch_npu={version}."
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
        """Execute the verified A5 HiFloat8 grouped matmul contract."""

        self.validate_grouped_matmul()
        return self._torch_npu.npu_grouped_matmul(
            [x1],
            [x2],
            bias=[] if bias is None else bias,
            scale=[x2_scale],
            per_token_scale=[x1_scale],
            group_list=group_list,
            group_type=group_type,
            output_dtype=output_dtype,
            group_list_type=group_list_type,
            x_dtype=self._torch_npu.hifloat8,
            weight_dtype=self._torch_npu.hifloat8,
            split_item=3,
        )[0]


@lru_cache(maxsize=1)
def _get_hifloat8_npu_ops() -> HiFloat8NpuOps:
    """Return the process-local A5 HiFloat8 operator adapter."""

    try:
        import torch_npu  # pylint: disable=C0415
    except ImportError as exc:
        raise RuntimeError(
            "A5 HiFloat8 requires torch_npu, but it is not importable."
        ) from exc
    return HiFloat8NpuOps(torch_npu)


def validate_hifloat8_runtime() -> None:
    """Fail during model setup when the HiFloat8 runtime is unavailable."""

    _get_hifloat8_npu_ops()


def validate_hifloat8_gmm_runtime() -> None:
    """Fail during expert setup when the HiFloat8 GMM runtime is unavailable."""

    _get_hifloat8_npu_ops().validate_grouped_matmul()


def _left_operand(
    tensor: HiFloat8Tensor,
    *,
    transpose: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the Pangu-compatible left operand direction."""

    if transpose:
        if not tensor.is_colwise():
            raise ValueError(
                "Transposed HiFloat8 left operand requires column-wise data."
            )
        return tensor.col_data.transpose(-1, -2), tensor.col_scale
    if not tensor.is_rowwise():
        raise ValueError(
            "Non-transposed HiFloat8 left operand requires row-wise data."
        )
    return tensor.row_data, tensor.row_scale


def _right_operand(
    tensor: HiFloat8Tensor,
    *,
    transpose: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the Pangu-compatible right operand direction."""

    if transpose:
        if not tensor.is_rowwise():
            raise ValueError(
                "Transposed HiFloat8 right operand requires row-wise data."
            )
        return tensor.row_data.transpose(-1, -2), tensor.row_scale
    if not tensor.is_colwise():
        raise ValueError(
            "Non-transposed HiFloat8 right operand requires column-wise data."
        )
    return tensor.col_data, tensor.col_scale


def hifloat8_matmul(
    left: HiFloat8Tensor,
    right: HiFloat8Tensor,
    *,
    layout: str,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Multiply typed HiFloat8 operands and return a high-precision Tensor."""

    if not isinstance(left, HiFloat8Tensor) or not isinstance(
        right,
        HiFloat8Tensor,
    ):
        raise TypeError("hifloat8_matmul requires two HiFloat8Tensor operands.")
    if layout not in _SUPPORTED_LAYOUTS:
        raise ValueError(
            f"Unsupported HiFloat8 layout {layout!r}; "
            f"expected one of {_SUPPORTED_LAYOUTS}."
        )

    result_dtype = output_dtype or left.dtype
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
        output_dtype=result_dtype,
    )


def hifloat8_grouped_matmul(
    left: HiFloat8Tensor,
    right: HiFloat8Tensor,
    *,
    layout: str,
    group_list: torch.Tensor,
    group_type: int = 0,
    group_list_type: int = 0,
    output_dtype: Optional[torch.dtype] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Multiply expert-grouped HiFloat8 operands on A5.

    ``group_type=0`` splits the token/M axis for forward and dgrad, while
    ``group_type=2`` splits the contracting axis for wgrad.
    """

    if not isinstance(left, HiFloat8Tensor) or not isinstance(
        right,
        HiFloat8Tensor,
    ):
        raise TypeError(
            "hifloat8_grouped_matmul requires two HiFloat8Tensor operands."
        )
    if layout not in _SUPPORTED_LAYOUTS:
        raise ValueError(
            f"Unsupported HiFloat8 GMM layout {layout!r}; "
            f"expected one of {_SUPPORTED_LAYOUTS}."
        )
    if not isinstance(group_list, torch.Tensor) or group_list.ndim != 1:
        raise ValueError("HiFloat8 GMM group_list must be one-dimensional.")
    if group_list.dtype != torch.int64:
        raise TypeError("HiFloat8 GMM group_list must use torch.int64.")
    if group_type not in (0, 2):
        raise ValueError(
            f"HiFloat8 GMM group_type must be 0 or 2, got {group_type}."
        )
    if group_list_type not in (0, 1):
        raise ValueError(
            "HiFloat8 GMM group_list_type must be 0 or 1, "
            f"got {group_list_type}."
        )

    left_data, left_scale = _left_operand(
        left,
        transpose=layout[0] == "T",
    )
    right_data, right_scale = _right_operand(
        right,
        transpose=layout[1] == "T",
    )
    if left_scale.dtype != torch.float32 or right_scale.dtype != torch.float32:
        raise TypeError("HiFloat8 GMM scales must use torch.float32.")
    return left.quantizer.npu_ops.quant_grouped_matmul(
        left_data,
        right_data,
        right_scale,
        x1_scale=left_scale,
        group_list=group_list,
        group_type=group_type,
        group_list_type=group_list_type,
        output_dtype=output_dtype or left.dtype,
        bias=None if bias is None else [bias],
    )


__all__ = [
    "HiFloat8NpuOps",
    "hifloat8_grouped_matmul",
    "hifloat8_matmul",
    "validate_hifloat8_gmm_runtime",
    "validate_hifloat8_runtime",
]
