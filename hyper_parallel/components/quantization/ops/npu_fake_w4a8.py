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
"""NPU operators used by the fake-QAT W4A8 grouped-linear strategy.

The native W4A8 strategy is intentionally not used here.  Fake W4A8 keeps the
master parameter in its source dtype, quantizes it to MXFP4 for every forward,
decodes that result to an MXFP8 operand, and executes the regular all-MXFP8
grouped-matmul contract.  This mirrors the QAT path used by the original
grouped W4A8 implementation while keeping the native Packed-MXFP4 storage
and WeightNz adapter isolated in ``npu_w4a8.py``.
"""

from functools import lru_cache
from types import ModuleType
from typing import Callable, Optional

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.quantization.config import (
    GMM_SPLIT_SCALE,
    GROUP_LIST_OFFSETS,
    MX_BLOCK_SIZE_32,
)
from hyper_parallel.components.quantization.ops.npu_w4a8 import W4A8CapabilityError


_A5_DEVICE_MARKERS = ("Ascend950", "Ascend910_95")


class FakeW4A8CapabilityError(W4A8CapabilityError):
    """Report an unavailable fake-W4A8 NPU operator contract."""


class FakeW4A8NpuOps:
    """Adapt MX quantization, MXFP4 conversion, and MXFP8 grouped matmul."""

    def __init__(self, torch_npu_module: ModuleType) -> None:
        """Validate and retain the active torch_npu operator module."""
        required = (
            "npu_dynamic_mx_quant",
            "npu_grouped_dynamic_mx_quant",
            "npu_grouped_matmul",
            "float4_e2m1fn_x2",
            "float8_e4m3fn",
            "float8_e8m0fnu",
        )
        missing = [name for name in required if not hasattr(torch_npu_module, name)]
        if missing:
            version = getattr(torch_npu_module, "__version__", "unknown")
            raise FakeW4A8CapabilityError(
                "The active torch_npu stack does not provide fake W4A8 operators; "
                f"missing {missing}, torch_npu={version}."
            )
        npu_handle = getattr(torch_npu_module, "npu", None)
        try:
            device_name = (
                npu_handle.get_device_name()
                if npu_handle is not None and hasattr(npu_handle, "get_device_name")
                else ""
            )
        except (RuntimeError, TypeError, OSError):
            device_name = ""
        if not any(marker in device_name for marker in _A5_DEVICE_MARKERS):
            raise FakeW4A8CapabilityError(
                "Fake W4A8 requires an Ascend 950/A5 runtime, "
                f"got {device_name or 'unknown'}."
            )
        self._torch_npu = torch_npu_module

    @property
    def activation_dtype(self) -> torch.dtype:
        """Return the MXFP8 activation dtype exposed by torch_npu."""
        return self._torch_npu.float8_e4m3fn

    @property
    def weight_dtype(self) -> torch.dtype:
        """Return the packed MXFP4 weight dtype exposed by torch_npu."""
        return self._torch_npu.float4_e2m1fn_x2

    @property
    def e8m0_dtype(self) -> torch.dtype:
        """Return the E8M0 scale dtype used by the grouped-matmul API."""
        return self._torch_npu.float8_e8m0fnu

    def dynamic_mx_quant(
        self,
        tensor: torch.Tensor,
        *,
        axis: int,
        block_size: int,
        quant_dtype: torch.dtype,
        round_mode: Optional[str] = None,
        scale_alg: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize a tensor with the requested dynamic-MX orientation."""
        kwargs = {"axis": axis, "dst_type": quant_dtype, "block_size": block_size}
        if round_mode is not None:
            kwargs["round_mode"] = round_mode
        if scale_alg is not None:
            kwargs["scale_alg"] = scale_alg
        return self._torch_npu.npu_dynamic_mx_quant(tensor, **kwargs)

    def convert_packed_mxfp4_to_block_mxfp8(
        self,
        packed_mxfp4: torch.Tensor,
        mxfp4_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run CANN's converter and return MXFP8 data with two axis scales."""

        return _get_cann_mx_to_block_mx_quant_op()(
            packed_mxfp4,
            mxfp4_scale,
            dst_type=self.activation_dtype,
            x_type=self.weight_dtype,
        )

    def grouped_dynamic_mx_quant(
        self,
        tensor: torch.Tensor,
        group_list: torch.Tensor,
        *,
        block_size: int,
        quant_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize grouped activation ranges with block-32 MX metadata."""
        if block_size != MX_BLOCK_SIZE_32:
            raise ValueError("npu_grouped_dynamic_mx_quant is only valid for block_size=32.")
        return self._torch_npu.npu_grouped_dynamic_mx_quant(
            tensor,
            group_list.to(torch.int32),
            round_mode="rint",
            dst_type=quant_dtype,
            blocksize=MX_BLOCK_SIZE_32,
        )

    def grouped_matmul(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        right_scale: torch.Tensor,
        *,
        left_scale: torch.Tensor,
        group_list: torch.Tensor,
        group_type: int,
        block_size: int,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Execute one grouped matrix multiplication with fake-MX operands."""
        if block_size != MX_BLOCK_SIZE_32:
            raise ValueError("Fake W4A8 supports only block_size=32.")
        kwargs = {
            "bias": [],
            "scale": [right_scale],
            "per_token_scale": [left_scale],
            "group_list": group_list.to(torch.int64),
            "group_type": group_type,
            "output_dtype": output_dtype,
            "group_list_type": GROUP_LIST_OFFSETS,
            "split_item": GMM_SPLIT_SCALE,
            "scale_dtype": self.e8m0_dtype,
            "per_token_scale_dtype": self.e8m0_dtype,
        }
        return self._torch_npu.npu_grouped_matmul([left], [right], **kwargs)[0]


@lru_cache(maxsize=1)
def _get_fake_w4a8_npu_ops() -> FakeW4A8NpuOps:
    """Resolve the process-local fake-W4A8 NPU adapter."""

    try:
        import torch_npu  # pylint: disable=C0415
    except ImportError as exc:
        raise FakeW4A8CapabilityError(
            "Fake W4A8 requires torch_npu, but it is not importable."
        ) from exc
    return FakeW4A8NpuOps(torch_npu)


@lru_cache(maxsize=1)
def _get_cann_mx_to_block_mx_quant_op() -> Callable[
    ..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]
]:
    """Load the CANN operator behind the project-level conversion method."""

    try:
        from cann_ops_nn import (  # pylint: disable=C0415
            mx_to_block_mx_quant as cann_mx_to_block_mx_quant,
        )
    except (ImportError, OSError) as exc:
        raise FakeW4A8CapabilityError(
            "Fake W4A8 block_size=32 requires CANN 9.2 with "
            "cann_ops_nn.mx_to_block_mx_quant importable."
        ) from exc
    return cann_mx_to_block_mx_quant


def validate_fake_w4a8_gmm_runtime(block_size: int) -> None:
    """Validate fake-W4A8 operators and the block-32 CANN converter."""

    _get_fake_w4a8_npu_ops()
    if block_size != MX_BLOCK_SIZE_32:
        raise ValueError("Fake W4A8 supports only block_size=32.")
    _get_cann_mx_to_block_mx_quant_op()


__all__ = [
    "FakeW4A8CapabilityError",
    "FakeW4A8NpuOps",
    "validate_fake_w4a8_gmm_runtime",
]
