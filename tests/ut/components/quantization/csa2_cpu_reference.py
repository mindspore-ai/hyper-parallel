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
"""CPU quantizer stand-ins for testing cache ownership and STE only."""

from unittest import TestCase
from unittest.mock import patch

import torch
from torch import Tensor

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _quantize_e2m1(value: Tensor) -> Tensor:
    """Round a small CPU tensor to E2M1 for cache-flow tests."""
    positive = _E2M1[:8]
    magnitude = value.abs()
    code = torch.zeros_like(magnitude, dtype=torch.uint8)
    for upper, (lower_value, upper_value) in enumerate(zip(positive[:-1], positive[1:]), 1):
        midpoint = (lower_value + upper_value) / 2
        moves_up = magnitude >= midpoint if upper % 2 == 0 else magnitude > midpoint
        code = torch.where(moves_up, upper, code)
    return code | torch.where(value < 0, 8, 0).to(torch.uint8)


def pack_main(kv: Tensor, scale_format: str = "e4m3") -> Tensor:
    """Encode Main rows on CPU for wrapper tests, independent of torch_npu."""
    width = kv.shape[-1]
    if width % 16:
        raise ValueError("Main KV width is not divisible by 16")
    group = kv.float().reshape(*kv.shape[:-1], width // 16, 16)
    scale_dtype = torch.float8_e4m3fn if scale_format == "e4m3" else torch.bfloat16
    row_bytes = width // 2 + width // 16 * (1 if scale_format == "e4m3" else 2)
    if scale_format == "e8m7":
        row_bytes = (row_bytes + 31) // 32 * 32
    if kv.numel() == 0:
        return torch.empty((*kv.shape[:-1], row_bytes), dtype=torch.uint8)
    scales = (group.abs().amax(-1) / 6).clamp(min=2 ** -9, max=448).to(scale_dtype)
    codes = _quantize_e2m1(group / scales.float().unsqueeze(-1)).reshape(*kv.shape[:-1], width)
    data = codes[..., 0::2] | (codes[..., 1::2] << 4)
    packed = torch.cat((data, scales.view(torch.uint8)), dim=-1)
    return torch.nn.functional.pad(packed, (0, row_bytes - packed.shape[-1]))


def unpack_main(packed: Tensor, width: int, dtype: torch.dtype, scale_format: str = "e4m3") -> Tensor:
    """Decode CPU Main stand-in bytes for cache-flow tests."""
    if packed.numel() == 0:
        return torch.empty((*packed.shape[:-1], width), dtype=dtype)
    data = packed[..., :width // 2]
    codes = torch.empty((*data.shape[:-1], width), dtype=torch.long)
    codes[..., 0::2] = (data & 15).long()
    codes[..., 1::2] = (data >> 4).long()
    scale_dtype = torch.float8_e4m3fn if scale_format == "e4m3" else torch.bfloat16
    scale_bytes = width // 16 * (1 if scale_format == "e4m3" else 2)
    scales = packed[..., width // 2:width // 2 + scale_bytes].contiguous().view(scale_dtype)
    return (torch.tensor(_E2M1, dtype=dtype)[codes].reshape(*data.shape[:-1], width // 16, 16)
            * scales.to(dtype).unsqueeze(-1)).reshape(*data.shape[:-1], width)


def pack_swa(kv: Tensor, scale_format: str = "e8m0") -> Tensor:
    """Encode local SWA rows on CPU for wrapper tests."""
    width = kv.shape[-1]
    if width % 32:
        raise ValueError("SWA KV width is not divisible by 32")
    slots = width // 32 + (width // 32) % 2
    row_bytes = width + slots if scale_format == "e8m0" else (width + 2 * width // 32 + 31) // 32 * 32
    if kv.numel() == 0:
        return torch.empty((*kv.shape[:-1], row_bytes), dtype=torch.uint8)
    grouped = kv.float().reshape(*kv.shape[:-1], width // 32, 32)
    maximum = grouped.abs().amax(-1)
    exponent = torch.ceil(torch.log2((maximum / 448).clamp_min(2 ** -127))).clamp(-127, 127)
    scale = torch.exp2(exponent) if scale_format == "e8m0" else (maximum / 448).clamp_min(2 ** -126).to(torch.bfloat16)
    data = (grouped / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    scale_bytes = (exponent + 127).to(torch.uint8) if scale_format == "e8m0" else scale.view(torch.uint8)
    if scale_format == "e8m0" and slots != width // 32:
        scale_bytes = torch.nn.functional.pad(scale_bytes, (0, 1))
    packed = torch.cat((data.view(torch.uint8).reshape(*kv.shape[:-1], width), scale_bytes), dim=-1)
    return torch.nn.functional.pad(packed, (0, row_bytes - packed.shape[-1]))


def unpack_swa(packed: Tensor, width: int, scale_format: str = "e8m0") -> Tensor:
    """Decode CPU SWA stand-in bytes for cache-flow tests."""
    if packed.numel() == 0:
        return torch.empty((*packed.shape[:-1], width), dtype=torch.bfloat16)
    data = packed[..., :width].view(torch.float8_e4m3fn).float()
    scales = (torch.exp2(packed[..., width:width + width // 32].float() - 127)
              if scale_format == "e8m0" else packed[..., width:width + 2 * width // 32]
              .contiguous().view(torch.bfloat16).float())
    return (data.reshape(*packed.shape[:-1], width // 32, 32)
            * scales.unsqueeze(-1)).reshape(*packed.shape[:-1], width).to(torch.bfloat16)


def install_cache_standins(test: TestCase) -> None:
    """Patch only the test's cache codec calls, never production dispatch."""
    module = "hyper_parallel.components.quantization.tensor.csa2_cache_tensor"
    for name, implementation in (("pack_main_kv", pack_main), ("unpack_main_kv", unpack_main),
                                 ("pack_swa_kv", pack_swa), ("unpack_swa_kv", unpack_swa)):
        patcher = patch(f"{module}.{name}", implementation)
        patcher.start()
        test.addCleanup(patcher.stop)
