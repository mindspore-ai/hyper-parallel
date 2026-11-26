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
"""Engram's discrete hash and differentiable RMS/signed-root gate."""

from __future__ import annotations

from typing import Protocol

import torch

from .._dependency import require_package, require_symbol
from ..registry import OpRegistry
from ..spec import OpSpec


class EngramHashFn(Protocol):
    """Return int32 global embedding rows for all supplied layers."""

    def __call__(self, tokens: torch.Tensor, multipliers: torch.Tensor, vocab_sizes: torch.Tensor,
                 offsets: torch.Tensor, image_token_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Hash [tokens, ngram] histories; masked image rows return -1."""


class EngramGateFn(Protocol):
    """Fuse packed key/value features into residual streams."""

    def __call__(self, hidden: torch.Tensor, kv: torch.Tensor, weight_hidden: torch.Tensor,
                 weight_embed: torch.Tensor, clamp_value: float, eps: float,
                 image_token_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Preserve masked rows and return the activation dtype with first-order gradients."""


ENGRAM_HASH = OpSpec[EngramHashFn](config_key="engram", variant="hash")
ENGRAM_GATE = OpSpec[EngramGateFn](config_key="engram", variant="gate")


def _native_hash(tokens, multipliers, vocab_sizes, offsets, image_token_mask=None):
    """Original rolling-XOR eager hash, batched over layers."""
    products = tokens.long().unsqueeze(0) * multipliers.unsqueeze(1)
    rolling = products[..., 0]
    columns = []
    layer_offsets = offsets.unflatten(-1, vocab_sizes.shape[-2:])
    for index in range(1, tokens.shape[-1]):
        rolling = rolling.bitwise_xor(products[..., index])
        row = rolling.unsqueeze(-1) % vocab_sizes[:, None, index - 1].long()
        columns.append(row + layer_offsets[:, None, index - 1])
    result = torch.cat(columns, dim=-1).to(offsets.dtype)
    if image_token_mask is not None:
        result = result.masked_fill(image_token_mask.reshape(1, -1, 1), -1)
    return result


def _native_gate(hidden, kv, weight_hidden, weight_embed, clamp_value, eps, image_token_mask=None):
    """Original eager WKV gate, including its FP32 evaluation order."""
    streams, width = hidden.shape[-2:]
    key, value = kv.split((streams * width, width), dim=-1)
    key = key.float().unflatten(-1, (streams, width))
    x = hidden.float()
    weight = weight_hidden.float() * weight_embed.float()
    reciprocal_std = torch.rsqrt(x.square().mean(-1) + eps)
    reciprocal_std = reciprocal_std * torch.rsqrt(key.square().mean(-1) + eps)
    dot = (x * weight * key).sum(-1)
    dot = dot * reciprocal_std * width**-0.5
    root = dot.abs().clamp_min(clamp_value).sqrt()
    # Ascend uses the positive branch at zero, matching the existing Engram module.
    gate = torch.where(dot < 0, -root, root).sigmoid()
    output = (x + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(hidden.dtype)
    if image_token_mask is not None:
        output = torch.where(image_token_mask.reshape(hidden.shape[:-2])[..., None, None], hidden, output)
    return output


def _load_hash() -> EngramHashFn:
    kernels = require_package("tile_kernels.engram")
    require_symbol(kernels, "engram_hash")
    return kernels.engram_hash


def _load_gate() -> EngramGateFn:
    kernels = require_package("tile_kernels.modeling.engram")
    require_symbol(kernels, "engram_gate")
    return kernels.engram_gate


def register_engram(registry: OpRegistry) -> None:
    """Register Torch references and lazy TileKernels implementations."""
    registry.register_family("engram", specs=(ENGRAM_HASH, ENGRAM_GATE))
    for spec, native, loader in ((ENGRAM_HASH, _native_hash, _load_hash), (ENGRAM_GATE, _native_gate, _load_gate)):
        registry.register_impl(spec, name="native", devices=("cpu", "cuda", "npu"), load=lambda fn=native: fn)
        registry.register_impl(spec, name="tile_kernels", devices=("npu",), load=loader)
