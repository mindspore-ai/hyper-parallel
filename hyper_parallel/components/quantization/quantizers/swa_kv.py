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
"""DeepSeek V4.1 sliding-window KV E4M3 cache with E8M0 or BF16 scales."""

import torch
from torch import Tensor

from hyper_parallel.components.quantization.config import FP8_E4M3_MAX, SWA_KV_GROUP_SIZE
from hyper_parallel.components.quantization.ops.npu_fp8 import allow_fully_written_cache_allocation


def _validate_swa_kv(kv: Tensor) -> None:
    """Require BF16 rows with a complete group-32 channel layout."""
    if kv.ndim < 2 or kv.dtype != torch.bfloat16:
        raise ValueError("SWA FP8 cache requires BF16 input with [..., D] shape.")
    if kv.shape[-1] <= 0 or kv.shape[-1] % SWA_KV_GROUP_SIZE:
        raise ValueError("SWA KV last axis must be positive and divisible by 32.")


def _scale_slots(width: int) -> int:
    """Pad E8M0 scale bytes to the native operator's two-scale pairs."""
    groups = width // SWA_KV_GROUP_SIZE
    return groups + groups % 2


def _swa_row_bytes(width: int, scale_format: str) -> int:
    """Return packed row width, including BF16 cache alignment."""
    if scale_format == "e8m0":
        return width + _scale_slots(width)
    payload = width + 2 * width // SWA_KV_GROUP_SIZE
    return (payload + 31) // 32 * 32


def pack_swa_kv(kv: Tensor, scale_format: str = "e8m0") -> Tensor:
    """Quantize E4M3 rows with E8M0 or BF16 group scales."""
    _validate_swa_kv(kv)
    if scale_format not in ("e8m0", "e8m7"):
        raise ValueError(f"Unsupported SWA KV scale format: {scale_format!r}.")
    if kv.device.type != "npu":
        raise RuntimeError("CSA2 SWA KV quantization requires NPU tensors.")
    width = kv.shape[-1]
    if kv.numel() == 0:
        return torch.empty((*kv.shape[:-1], _swa_row_bytes(width, scale_format)),
                           dtype=torch.uint8, device=kv.device)
    if scale_format == "e8m0":
        import torch_npu  # pylint: disable=C0415
        flat = kv.reshape(-1, width).contiguous()
        with allow_fully_written_cache_allocation():
            data, scale = torch_npu.npu_dynamic_mx_quant(
                flat, dst_type=torch.float8_e4m3fn, block_size=SWA_KV_GROUP_SIZE, scale_alg=1,
            )
        return torch.cat((data.view(torch.uint8).reshape(*kv.shape[:-1], width),
                          scale.reshape(*kv.shape[:-1], -1)), dim=-1)
    grouped = kv.float().reshape(*kv.shape[:-1], width // SWA_KV_GROUP_SIZE, SWA_KV_GROUP_SIZE)
    amax = torch.nan_to_num(grouped.abs(), nan=float("inf")).amax(dim=-1)
    scale = (amax / FP8_E4M3_MAX).clamp_min(torch.finfo(torch.bfloat16).tiny).to(torch.bfloat16)
    normalized = (grouped / scale.float().unsqueeze(-1)).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    with allow_fully_written_cache_allocation():
        data = normalized.to(torch.float8_e4m3fn)
    packed = torch.cat((data.view(torch.uint8).reshape(*kv.shape[:-1], width), scale.view(torch.uint8)), dim=-1)
    padding = _swa_row_bytes(width, scale_format) - packed.shape[-1]
    return torch.nn.functional.pad(packed, (0, padding)) if padding else packed


def unpack_swa_kv(packed: Tensor, logical_width: int, scale_format: str = "e8m0") -> Tensor:
    """Reconstruct BF16 values from one physical sliding-window KV row."""
    if logical_width <= 0 or logical_width % SWA_KV_GROUP_SIZE:
        raise ValueError("SWA KV logical width must be positive and divisible by 32.")
    if scale_format not in ("e8m0", "e8m7"):
        raise ValueError(f"Unsupported SWA KV scale format: {scale_format!r}.")
    if packed.dtype != torch.uint8 or packed.shape[-1] != _swa_row_bytes(logical_width, scale_format):
        raise ValueError("SWA KV packed row does not match its logical width.")
    if packed.device.type != "npu":
        raise RuntimeError("CSA2 SWA KV dequantization requires NPU tensors.")
    if packed.numel() == 0:
        return torch.empty((*packed.shape[:-1], logical_width), dtype=torch.bfloat16, device=packed.device)
    data = packed[..., :logical_width].contiguous()
    if scale_format == "e8m0":
        scale = packed[..., logical_width:logical_width + _scale_slots(logical_width)]
        scale = scale.reshape(*data.shape[:-1], _scale_slots(logical_width) // 2, 2)
        import torch_npu  # pylint: disable=C0415
        values = torch_npu.npu_anti_mx_quant(
            data.reshape(-1, logical_width).view(torch.float8_e4m3fn),
            scale.reshape(-1, _scale_slots(logical_width) // 2, 2).contiguous(),
            src_type=torch.float8_e4m3fn, dst_type=15,
        )
        return values.reshape(*packed.shape[:-1], logical_width)
    scale_bytes = 2 * logical_width // SWA_KV_GROUP_SIZE
    scales = packed[..., logical_width:logical_width + scale_bytes].contiguous().view(torch.bfloat16)
    values = data.view(torch.float8_e4m3fn).to(torch.bfloat16)
    return (values.reshape(*data.shape[:-1], logical_width // SWA_KV_GROUP_SIZE, SWA_KV_GROUP_SIZE)
            * scales.unsqueeze(-1)).reshape(*data.shape[:-1], logical_width)
