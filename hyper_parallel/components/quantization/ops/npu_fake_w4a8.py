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
from typing import Optional

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.quantization.ops.npu_w4a8 import (
    W4A8CapabilityError,
)


_A5_DEVICE_MARKERS = ("Ascend950", "Ascend910_95")
_SUPPORTED_BLOCK_SIZES = (32, 128)


class FakeW4A8CapabilityError(W4A8CapabilityError):
    """Report an unavailable fake-W4A8 NPU operator contract."""


def _decode_packed_mxfp4(packed: torch.Tensor) -> torch.Tensor:
    """Decode two E2M1 values in each packed byte without CPU round-trips."""

    if packed.dtype != torch.uint8:
        raise TypeError(
            "Packed MXFP4 data must be uint8, "
            f"got {packed.dtype}."
        )
    # E2M1 codes are ordered as positive values in the low/high nibbles and
    # their sign-bit variants in the upper half.  Keeping the table in the
    # fake-only module prevents the native path from regressing to a LUT
    # expansion while preserving the exact MXFP4 E2M1 decode.
    values = torch.tensor(
        (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0),
        dtype=torch.float16,
        device=packed.device,
    )
    low = values[(packed & 0x0F).long()]
    high = values[((packed >> 4) & 0x0F).long()]
    return torch.stack((low, high), dim=-1).flatten(-2, -1)


def fake_mxfp4_to_mxfp8(
    packed: torch.Tensor,
    scale: torch.Tensor,
    *,
    mx4_block_size: int = 32,
    mx8_block_h: int = 32,
    mx8_block_w: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode packed MXFP4 and requantize it into a full-MXFP8 operand.

    ``npu_dynamic_mx_quant`` emits one E8M0 exponent for each 32-value MXFP4
    block.  Fake W4A8 groups those blocks into the block-size square used by
    the ordinary MXFP8 GMM.  The returned scale is raw E8M0 for block-32 and
    is converted to FP32 for block-128, matching the active GMM contracts.
    """

    if packed.ndim != 3 or scale.ndim != 4 or scale.shape[-1] != 2:
        raise ValueError(
            "Fake W4A8 expects packed [E,H,K/2] and scale [E,H,K/64,2], "
            f"got packed={tuple(packed.shape)}, scale={tuple(scale.shape)}."
        )
    if mx4_block_size != 32 or mx8_block_h not in _SUPPORTED_BLOCK_SIZES:
        raise ValueError("Fake W4A8 only supports MXFP4=32 and MXFP8 blocks 32/128.")
    if packed.shape[-1] * 2 % mx4_block_size:
        raise ValueError("Packed MXFP4 contracting dimension must be divisible by 32.")
    if packed.shape[1] % mx8_block_h or packed.shape[-1] * 2 % mx8_block_w:
        raise ValueError(
            "Fake MXFP4-to-MXFP8 conversion requires dimensions divisible by "
            f"the output block ({mx8_block_h}, {mx8_block_w})."
        )

    decoded = _decode_packed_mxfp4(packed)
    # Keep the original MXFP4 E8M0 exponents until the output MXFP8 block is
    # formed.  The MXFP4-to-MXFP8 conversion chooses ``s8 = max(s4) - 6``:
    # the largest E2M1 value is 6, so the scaled E4M3 block stays within its
    # finite range while the GMM's E8M0 scale restores the same magnitude.
    # The native MXFP4 scale layout is ``[E, H, ceil(W/64), 2]``.  The
    # trailing pair stores the two 32-value exponents covered by each 64-value
    # tile; for an odd final tile (notably ``W=32``) the second entry is a
    # hardware padding value.  Flatten the pair and discard only those padded
    # entries before mapping scales to the actual 32-value MXFP4 blocks.
    e4_scale = scale.reshape(scale.shape[0], scale.shape[1], -1)
    e4_scale = e4_scale[..., : packed.shape[-1] * 2 // mx4_block_size]
    e4_scale = e4_scale.to(torch.float32)
    h_blocks = packed.shape[1] // mx8_block_h
    w = packed.shape[-1] * 2
    w_blocks = w // mx8_block_w
    scale_blocks = e4_scale.reshape(
        packed.shape[0], h_blocks, mx8_block_h, w_blocks,
        mx8_block_w // mx4_block_size,
    )
    scale_blocks = scale_blocks.unsqueeze(-1).expand(
        -1, -1, -1, -1, -1, mx4_block_size
    ).reshape(
        packed.shape[0], h_blocks, mx8_block_h, w_blocks, mx8_block_w
    ).permute(0, 1, 3, 2, 4)
    fp32_blocks = decoded.reshape(
        packed.shape[0], h_blocks, mx8_block_h, w_blocks, mx8_block_w
    ).permute(0, 1, 3, 2, 4)
    exponent = (scale_blocks.amax(dim=(-2, -1)) - 6.0).clamp(0.0, 255.0)
    exponent_diff = scale_blocks - exponent[..., None, None]
    scaled = fp32_blocks * torch.pow(2.0, exponent_diff)
    fp8_blocks = scaled.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    fp8 = fp8_blocks.permute(0, 1, 3, 2, 4).reshape(
        packed.shape[0], packed.shape[1], w
    )
    if mx8_block_h == 32:
        return fp8, exponent.to(torch.uint8)
    return fp8, torch.pow(2.0, exponent.to(torch.float32) - 127.0)


def expand_scale_to_per_row(scale_2d: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Expand ``[E,H/32,W/32]`` scales for the transposed dgrad operand."""

    if scale_2d.ndim != 3 or block_size != 32:
        raise ValueError("Fake block-32 scale expansion expects a 3D scale tensor.")
    _, h_blocks, w_blocks = scale_2d.shape
    expanded = scale_2d[:, :, None, :].expand(-1, -1, block_size, -1)
    per_row = expanded.reshape(scale_2d.shape[0], h_blocks * block_size, w_blocks)
    return per_row.reshape(scale_2d.shape[0], h_blocks * block_size, w_blocks // 2, 2)


def convert_mx_scale_to_axis2(
    scale_mxfp8: torch.Tensor,
    weight_shape: tuple[int, ...],
) -> torch.Tensor:
    """Convert square block scales to the GMM ``[E,K/64,N,2]`` layout."""

    if scale_mxfp8.ndim != 3 or len(weight_shape) != 3:
        raise ValueError("Fake block-32 weight scales require 3D inputs.")
    experts, rows, cols = weight_shape
    if scale_mxfp8.shape[0] != experts or rows % 64 or cols % 32:
        raise ValueError(
            "Fake block-32 weight scale dimensions do not match the GMM weight."
        )
    if scale_mxfp8.shape[1] * 32 != rows or scale_mxfp8.shape[2] * 32 != cols:
        raise ValueError("Fake block-32 scale grid does not match the weight shape.")
    reshaped = scale_mxfp8.reshape(experts, rows // 64, 2, cols // 32)
    top = reshaped[:, :, 0, :].unsqueeze(-1).expand(-1, -1, -1, 32)
    bottom = reshaped[:, :, 1, :].unsqueeze(-1).expand(-1, -1, -1, 32)
    return torch.stack((top.reshape(experts, rows // 64, cols),
                        bottom.reshape(experts, rows // 64, cols)), dim=-1)


def transform_grouped_scale(
    input_tensor: torch.Tensor,
    input_scale_mxfp8: torch.Tensor,
    group_list: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Insert zero scale rows at expert boundaries for group-type 2 GMM."""

    if input_scale_mxfp8.ndim != 3 or group_list.ndim != 1:
        raise ValueError("Grouped fake-MX scales require a 3D scale and 1D groups.")
    if block_size not in _SUPPORTED_BLOCK_SIZES:
        raise ValueError(f"Unsupported fake W4A8 block size {block_size}.")
    indices = torch.arange(group_list.shape[0], device=group_list.device, dtype=group_list.dtype)
    zero_rows = group_list // (block_size // 2) + indices
    repeated = input_scale_mxfp8.reshape(-1, input_scale_mxfp8.shape[1]).repeat_interleave(4, dim=0)
    packed = repeated.reshape(input_scale_mxfp8.shape[0] * 4, input_scale_mxfp8.shape[1], 2)
    scale_shape = input_tensor.shape[0] // (block_size // 2) + group_list.shape[0]
    all_indices = torch.arange(scale_shape, device=packed.device)
    non_zero = all_indices[~torch.isin(all_indices, zero_rows)]
    grouped = torch.zeros(
        scale_shape,
        input_scale_mxfp8.shape[1],
        2,
        dtype=packed.dtype,
        device=packed.device,
    )
    if packed.shape[0] < non_zero.shape[0]:
        raise ValueError("Grouped fake-MX scales do not cover all token blocks.")
    grouped[non_zero] = packed[: non_zero.shape[0]]
    return grouped


class FakeW4A8NpuOps:
    """Adapt dynamic MX quantization and ordinary MXFP8 grouped matmul."""

    def __init__(self, torch_npu_module: ModuleType) -> None:
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
        return self._torch_npu.float8_e4m3fn

    @property
    def weight_dtype(self) -> torch.dtype:
        return self._torch_npu.float4_e2m1fn_x2

    @property
    def e8m0_dtype(self) -> torch.dtype:
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
        kwargs = {"axis": axis, "dst_type": quant_dtype, "block_size": block_size}
        if round_mode is not None:
            kwargs["round_mode"] = round_mode
        if scale_alg is not None:
            kwargs["scale_alg"] = scale_alg
        return self._torch_npu.npu_dynamic_mx_quant(tensor, **kwargs)

    def grouped_dynamic_mx_quant(
        self,
        tensor: torch.Tensor,
        group_list: torch.Tensor,
        *,
        block_size: int,
        quant_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if block_size != 32:
            raise ValueError("npu_grouped_dynamic_mx_quant is only valid for block_size=32.")
        return self._torch_npu.npu_grouped_dynamic_mx_quant(
            tensor,
            group_list.to(torch.int32),
            round_mode="rint",
            dst_type=quant_dtype,
            blocksize=32,
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
        kwargs = {
            "bias": [],
            "scale": [right_scale],
            "per_token_scale": [left_scale],
            "group_list": group_list.to(torch.int64),
            "group_type": group_type,
            "output_dtype": output_dtype,
            "group_list_type": 0,
            "split_item": 3 if block_size == 32 or group_type == 2 else 2,
        }
        if block_size == 32 or group_type == 2:
            kwargs["scale_dtype"] = self.e8m0_dtype
            kwargs["per_token_scale_dtype"] = self.e8m0_dtype
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


def validate_fake_w4a8_gmm_runtime() -> None:
    """Fail during replacement if the fake-W4A8 operator contract is absent."""

    _get_fake_w4a8_npu_ops()


__all__ = [
    "FakeW4A8CapabilityError",
    "FakeW4A8NpuOps",
    "convert_mx_scale_to_axis2",
    "expand_scale_to_per_row",
    "fake_mxfp4_to_mxfp8",
    "transform_grouped_scale",
    "validate_fake_w4a8_gmm_runtime",
]
