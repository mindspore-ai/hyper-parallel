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
"""DeepSeek V4.1 Main KV E2M1 cache with E4M3 or BF16 group scales."""

import torch
from torch import Tensor

from hyper_parallel.components.quantization.config import (
    FP8_E4M3_MAX,
    MAIN_KV_GROUP_SIZE,
    MXFP4_MAX_VALUE,
)
from hyper_parallel.components.quantization.ops.npu_fp8 import allow_fully_written_cache_allocation

_MIN_E4M3_SCALE = 2.0 ** -9


def _validate_main_kv(kv: Tensor) -> None:
    """Group only the feature axis; ragged token counts need no padding."""
    if kv.ndim < 2 or not kv.is_floating_point():
        raise ValueError("Main KV quantization expects a floating tensor with [..., D] shape.")
    if kv.shape[-1] <= 0 or kv.shape[-1] % MAIN_KV_GROUP_SIZE:
        raise ValueError(
            f"Main KV last axis {kv.shape[-1]} is not divisible by group_size {MAIN_KV_GROUP_SIZE}; "
            "no cross-row padding is allowed"
        )


def _main_row_bytes(width: int, scale_format: str) -> int:
    """Return packed row width; BF16-scale rows follow the 32-byte cache ABI."""
    payload = width // 2 + width // MAIN_KV_GROUP_SIZE * (2 if scale_format == "e8m7" else 1)
    return (payload + 31) // 32 * 32 if scale_format == "e8m7" else payload


def pack_main_kv(kv: Tensor, scale_format: str = "e4m3") -> Tensor:
    """Quantize E2M1 rows and append E4M3 or BF16 group scales."""
    _validate_main_kv(kv)
    if scale_format not in ("e4m3", "e8m7"):
        raise ValueError(f"Unsupported Main KV scale format: {scale_format!r}.")
    if kv.device.type != "npu":
        raise RuntimeError("CSA2 Main KV quantization requires NPU tensors.")
    width = kv.shape[-1]
    if kv.numel() == 0:
        return torch.empty((*kv.shape[:-1], _main_row_bytes(width, scale_format)),
                           dtype=torch.uint8, device=kv.device)
    grouped = kv.float().reshape(*kv.shape[:-1], width // MAIN_KV_GROUP_SIZE, MAIN_KV_GROUP_SIZE)
    amax = torch.nan_to_num(grouped.abs(), nan=float("inf")).amax(dim=-1)
    scales = amax / MXFP4_MAX_VALUE
    if scale_format == "e4m3":
        scales = scales.clamp(min=_MIN_E4M3_SCALE, max=FP8_E4M3_MAX)
    else:
        scales = scales.clamp_min(torch.finfo(torch.bfloat16).tiny)
    scale_values = torch.where(torch.isinf(amax), float("nan"), scales)
    if scale_format == "e4m3":
        with allow_fully_written_cache_allocation():
            scales = scale_values.to(torch.float8_e4m3fn)
    else:
        scales = scale_values.to(torch.bfloat16)
    normalized = grouped / scales.float().unsqueeze(-1)
    if scale_format == "e8m7":
        # BF16 rounding can move amax/6 downward; clip the normalized tail.
        normalized = normalized.clamp(-MXFP4_MAX_VALUE, MXFP4_MAX_VALUE)
    import torch_npu  # pylint: disable=C0415
    data = torch_npu.npu_dtype_cast(normalized, torch_npu.float4_e2m1fn_x2)
    packed = torch.cat((data.reshape(*kv.shape[:-1], width // 2), scales.view(torch.uint8)), dim=-1)
    padding = _main_row_bytes(width, scale_format) - packed.shape[-1]
    return torch.nn.functional.pad(packed, (0, padding)) if padding else packed


def unpack_main_kv(packed: Tensor, logical_width: int, dtype: torch.dtype,
                   scale_format: str = "e4m3") -> Tensor:
    """Read one physical Main KV cache row and reconstruct its attention values."""
    if logical_width <= 0 or logical_width % MAIN_KV_GROUP_SIZE:
        raise ValueError("Main KV logical width must be positive and divisible by 16.")
    if scale_format not in ("e4m3", "e8m7"):
        raise ValueError(f"Unsupported Main KV scale format: {scale_format!r}.")
    if packed.dtype != torch.uint8 or packed.shape[-1] != _main_row_bytes(logical_width, scale_format):
        raise ValueError("Main KV packed row does not match its logical width.")
    if packed.device.type != "npu":
        raise RuntimeError("CSA2 Main KV dequantization requires NPU tensors.")
    if packed.numel() == 0:
        return torch.empty((*packed.shape[:-1], logical_width), dtype=dtype, device=packed.device)
    data = packed[..., :logical_width // 2].contiguous()
    scale_bytes = logical_width // MAIN_KV_GROUP_SIZE * (2 if scale_format == "e8m7" else 1)
    scale_dtype = torch.bfloat16 if scale_format == "e8m7" else torch.float8_e4m3fn
    scale = packed[..., logical_width // 2:logical_width // 2 + scale_bytes].contiguous().view(scale_dtype)
    import torch_npu  # pylint: disable=C0415
    values = torch_npu.npu_dtype_cast(data, dtype, input_dtype=torch_npu.float4_e2m1fn_x2)
    grouped = values.reshape(*data.shape[:-1], logical_width // MAIN_KV_GROUP_SIZE, MAIN_KV_GROUP_SIZE)
    if scale_format == "e8m7":
        if dtype == torch.bfloat16:
            # The cast owns this BF16 buffer; scaling it in place avoids a
            # second full-sized decoded KV allocation before attention.
            grouped.mul_(scale.unsqueeze(-1))
            return values
        return (grouped.float() * scale.float().unsqueeze(-1)).reshape(*data.shape[:-1], logical_width).to(dtype)
    if dtype == torch.bfloat16:
        grouped.mul_(scale.to(dtype).unsqueeze(-1))
        return values
    return (grouped * scale.to(dtype).unsqueeze(-1)).reshape(*data.shape[:-1], logical_width)
