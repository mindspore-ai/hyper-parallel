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
"""FP8/FP4 conversion contracts with explicit data/scale representations.

Conversions are non-differentiable. They do not install straight-through estimators
or change the quantizers selected by existing training strategies.
"""

from __future__ import annotations

from typing import Any, Protocol

import torch
from torch.nn import functional as F

from .._dependency import require_package, require_symbol
from ..registry import OpRegistry
from ..spec import OpSpec

QuantTensor = tuple[torch.Tensor, torch.Tensor]


class PerTokenCastFn(Protocol):
    """Quantize rows with one scaling factor per channel group."""

    def __call__(self, x: torch.Tensor | QuantTensor, fmt: str, num_per_channels: int,
                 **options: Any) -> QuantTensor:
        """Return FP8 data or packed FP4 bytes, and their scale tensor."""


class PerBlockCastFn(Protocol):
    """Quantize a matrix with one scale per rectangular block."""

    def __call__(self, x: torch.Tensor, fmt: str, block_size: tuple[int, int], **options: Any) -> QuantTensor:
        """Preserve backend scale layout and optional output buffers."""


class CastBackFn(Protocol):
    """Decode a data/scale pair into a floating matrix."""

    def __call__(self, x: QuantTensor, fmt: str, block_size: tuple[int, int], **options: Any) -> torch.Tensor:
        """Return BF16 or FP32 without creating a quantization gradient path."""


PER_TOKEN_CAST = OpSpec[PerTokenCastFn](config_key="quantization", variant="per_token")
PER_BLOCK_CAST = OpSpec[PerBlockCastFn](config_key="quantization", variant="per_block")
CAST_BACK = OpSpec[CastBackFn](config_key="quantization", variant="cast_back")


def _pack_fp4(values):
    levels = values.new_tensor((0, 0.5, 1, 1.5, 2, 3, 4, 6))
    distance = (values.abs().unsqueeze(-1) - levels).abs()
    codes = distance.argmin(-1)
    upper = (codes + 1).clamp_max(7)
    tied = distance.gather(-1, upper.unsqueeze(-1)).squeeze(-1) == distance.amin(-1)
    codes = torch.where(tied & (codes % 2 == 1), upper, codes)
    codes = codes.to(torch.uint8) | (torch.signbit(values).to(torch.uint8) << 3)
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).view(torch.int8)


def _unpack_fp4(data):
    raw = data.view(torch.uint8)
    codes = torch.stack((raw & 15, raw >> 4), dim=-1).flatten(-2).long()
    levels = torch.tensor((0, 0.5, 1, 1.5, 2, 3, 4, 6), device=data.device)
    return levels[codes & 7] * torch.where(codes < 8, 1.0, -1.0)


def _store_scale(scale, block, column_major, packed):
    if packed:
        exponent = (torch.log2(scale) + 127).to(torch.uint8)
        if column_major:
            exponent = F.pad(exponent.T.contiguous(), (0, exponent.shape[0] % 2))
            return exponent.view(torch.int16).T
        if block[1] == 1:
            exponent = F.pad(exponent, (0, 0, 0, exponent.shape[0] % 2))
            exponent = exponent.reshape(-1, 2, exponent.shape[1]).transpose(1, 2).contiguous().flatten(1)
        else:
            exponent = F.pad(exponent, (0, exponent.shape[1] % 2))
        return exponent.contiguous().view(torch.int16)
    return scale.T.contiguous().T if column_major else scale


def _read_scale(storage, data_shape, block):
    rows = (data_shape[0] + block[0] - 1) // block[0]
    columns = (data_shape[1] + block[1] - 1) // block[1]
    if storage.dtype == torch.int16:
        column_major = storage.stride(0) == 1 and storage.stride(1) != 1
        if column_major:
            exponent = storage.T.contiguous().view(torch.uint8).T
        elif block[1] == 1:
            exponent = storage.contiguous().view(torch.uint8).reshape(-1, columns, 2)
            exponent = exponent.transpose(1, 2).reshape(-1, columns)
        else:
            exponent = storage.contiguous().view(torch.uint8)
        scale = torch.pow(2.0, exponent.float() - 127)
    else:
        scale = storage.float()
    return scale[:rows, :columns].repeat_interleave(block[0], 0).repeat_interleave(block[1], 1)


@torch.no_grad()
def _native_cast_back(x, fmt, block_size, *, x_special_fmt=None):
    if x_special_fmt is not None:
        raise ValueError("special quantized formats are not supported")
    if fmt not in ("bf16", "fp32"):
        raise ValueError("cast_back output format must be bf16 or fp32")
    data, scale = x
    values = _unpack_fp4(data) if data.dtype in (torch.int8, torch.uint8) else data.float()
    expanded = _read_scale(scale, values.shape, block_size)
    return (values * expanded[:values.shape[0], :values.shape[1]]).to(
        torch.bfloat16 if fmt == "bf16" else torch.float32)


@torch.no_grad()
def _native_cast(x, fmt, block, *, use_tma_aligned_col_major_sf=False, round_sf=False,
                  use_packed_ue8m0=False, use_e4m3_sf=False, out=None):
    if fmt not in ("e4m3", "e2m1"):
        raise ValueError("quantization format must be e4m3 or e2m1")
    if use_packed_ue8m0 and not round_sf:
        raise ValueError("packed UE8M0 scales require round_sf")
    if use_e4m3_sf and round_sf:
        raise ValueError("E4M3 scales cannot be combined with round_sf")
    rows, columns = x.shape
    padded = F.pad(x.float(), (0, -columns % block[1], 0, -rows % block[0]))
    tiles = padded.reshape(padded.shape[0] // block[0], block[0], -1, block[1])
    maximum = 448.0 if fmt == "e4m3" else 6.0
    minimum = maximum * 2**-9 if use_e4m3_sf else (1e-4 if fmt == "e4m3" else 6.0 * 2**-126)
    amax = tiles.abs().amax((1, 3)).clamp_min(minimum)
    scale = amax / maximum
    if round_sf:
        scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    if use_e4m3_sf:
        scale = scale.to(torch.float8_e4m3fn)
    inverse = 1 / scale.float() if round_sf or use_e4m3_sf else torch.div(amax.new_tensor(maximum), amax)
    inverse = inverse.repeat_interleave(block[0], 0).repeat_interleave(block[1], 1)[:rows, :columns]
    values = x.float() * inverse
    data = values.to(torch.float8_e4m3fn) if fmt == "e4m3" else _pack_fp4(values)
    result = data, _store_scale(scale, block, use_tma_aligned_col_major_sf, use_packed_ue8m0)
    if out is not None:
        for target, value in zip(out, result):
            target.copy_(value)
        return out
    return result


def _native_per_token(x, fmt, num_per_channels, *, x_block_size=None, stochastic_cast=False, **options):
    if stochastic_cast:
        raise NotImplementedError("native quantization does not emulate TileKernels' stochastic device rounding")
    if isinstance(x, tuple):
        if x_block_size is None:
            raise ValueError("requantization requires x_block_size")
        x = _native_cast_back(x, "fp32", x_block_size)
    return _native_cast(x, fmt, (1, num_per_channels), **options)


def _native_per_block(x, fmt, block_size, **options):
    return _native_cast(x, fmt, block_size, **options)


def _load_symbol(name):
    kernels = require_package("tile_kernels.quant")
    require_symbol(kernels, name)
    return getattr(kernels, name)


def register_quantization(registry: OpRegistry) -> None:
    """Register conversion variants without changing existing training quantizers."""
    registry.register_family("quantization", specs=(PER_TOKEN_CAST, PER_BLOCK_CAST, CAST_BACK))
    for spec, native, symbol in ((PER_TOKEN_CAST, _native_per_token, "per_token_cast"),
                                 (PER_BLOCK_CAST, _native_per_block, "per_block_cast"),
                                 (CAST_BACK, _native_cast_back, "cast_back")):
        registry.register_impl(spec, name="native", devices=("cpu", "cuda", "npu"), load=lambda fn=native: fn)
        registry.register_impl(spec, name="tile_kernels", devices=("npu",), load=lambda name=symbol: _load_symbol(name))
