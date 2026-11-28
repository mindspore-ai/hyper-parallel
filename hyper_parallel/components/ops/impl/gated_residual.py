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
"""Qwen gated-residual read/inject contract; this is distinct from mHC."""
from __future__ import annotations


import math
from typing import Protocol

import torch
from torch.nn import functional as F

from .._dependency import require_package, require_symbol
from ..registry import OpRegistry
from ..spec import OpSpec


class GatedResidualFn(Protocol):
    """Read C residual streams and optionally produce their injection gates.

    x is [..., C*H]; down/up/gate weights are [R,C*H], [C*H,R], [C,C*H].
    gamma is the additive RMS scale [C*H]. All supplied tensors are differentiable.
    Returns mixed [...,H] or (mixed, inject [...,C]); no inputs are mutated.
    """

    def __call__(
        self, x: torch.Tensor, down: torch.Tensor, up: torch.Tensor, gamma: torch.Tensor,
        gate: torch.Tensor | None = None, *, channels: int = 4, eps: float = 1e-6,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply grouped offset RMSNorm, low-rank read gates and optional write gates."""


GATED_RESIDUAL = OpSpec[GatedResidualFn](config_key="gated_residual")


def validate_inputs(
    x: torch.Tensor, down: torch.Tensor, up: torch.Tensor, gamma: torch.Tensor,
    gate: torch.Tensor | None, channels: int, eps: float,
) -> None:
    """Keep the same semantic checks for every backend."""
    if isinstance(channels, bool) or not isinstance(channels, int) or channels < 1:
        raise ValueError("channels must be a positive integer")
    if x.ndim < 2 or x.shape[-1] == 0 or x.shape[-1] % channels:
        raise ValueError("gated residual expects [..., channels * hidden_size]")
    width = x.shape[-1]
    if down.ndim != 2 or down.shape[1] != width or down.shape[0] < 1:
        raise ValueError("down weight must have shape [lowrank, channels * hidden_size]")
    if up.shape != (width, down.shape[0]) or gamma.shape != (width,):
        raise ValueError("up weight or grouped RMS scale has an incompatible shape")
    if gate is not None and gate.shape != (channels, width):
        raise ValueError("gate weight must have shape [channels, channels * hidden_size]")
    tensors = (x, down, up, gamma) + (() if gate is None else (gate,))
    if any(t.device != x.device or not t.is_floating_point() for t in tensors):
        raise ValueError("gated residual requires floating tensors on the same device")
    if down.dtype != x.dtype or up.dtype != x.dtype or (gate is not None and gate.dtype != x.dtype):
        raise ValueError("projection weights must have the input dtype")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")


def cann_rejection(x: torch.Tensor, down: torch.Tensor, channels: int) -> str | None:
    """Use the documented A2/A3 training bounds, also valid conservatively on A5."""
    hidden, rank = x.shape[-1] // channels, down.shape[0]
    if x.dtype not in (torch.float16, torch.bfloat16):
        return "CANN gated residual requires FP16 or BF16"
    if not 2 <= channels <= 10 or hidden % 256 or not 256 <= hidden <= 4096:
        return "CANN gated residual requires C=2..10 and H=256..4096 in multiples of 256"
    limit = 39664 - 16 * math.ceil(channels / 8)
    if rank % 64 or not 64 <= rank <= 640 or max(13 * hidden, 3 * hidden + 8 * rank, 7 * hidden + 6 * rank) > limit:
        return "shape exceeds the CANN gated-residual training workspace bounds"
    return None


def _project(normalized, down, up, gate, channels):
    """Keep HF's low-precision projection and elementwise rounding boundaries."""
    read = F.silu(F.linear(normalized, down) / channels)
    read = F.linear(read, up).sigmoid()
    shape = (*normalized.shape[:-1], channels, normalized.shape[-1] // channels)
    mixed = (read.reshape(shape) * normalized.reshape(shape)).mean(dim=-2)
    if gate is None:
        return mixed
    return mixed, 2 * (F.linear(normalized, gate) / channels).sigmoid()


def _native(x, down, up, gamma, gate=None, *, channels=4, eps=1e-6):
    values = x.float().unflatten(-1, (channels, x.shape[-1] // channels))
    normalized = values * (values.square().mean(-1, keepdim=True) + eps).rsqrt()
    normalized = (normalized.flatten(-2) * (1 + gamma.float())).to(x.dtype)
    return _project(normalized, down, up, gate, channels)


def _load_native() -> GatedResidualFn:
    return _native


def _load_cann() -> GatedResidualFn:
    # The top-level CANN namespace exposes the raw forward dispatcher. Its ops
    # namespace preserves the Python wrapper that owns saved state and backward.
    package = require_package("cann_ops_transformer.ops")
    require_symbol(package, "text_gated_residual")
    kernel = package.text_gated_residual
    # Resolve the selected implementation, including official vendor entry points.
    # Older inference-only packages must not silently detach the trainable path.
    require_symbol(require_package(kernel.__module__), "_text_gated_residual_backward")

    def _operation(x, down, up, gamma, gate=None, *, channels=4, eps=1e-6):
        leading = x.shape[:-1]
        result = kernel(x.reshape(1, -1, x.shape[-1]), down, up,
                        w_gate=gate, channels=channels, gamma=gamma.float(), eps=eps)
        if gate is None:
            return result.reshape(*leading, x.shape[-1] // channels)
        mixed, inject = result
        return mixed.reshape(*leading, x.shape[-1] // channels), inject.reshape(*leading, channels)

    return _operation


def register_gated_residual(registry: OpRegistry) -> None:
    """Register eager and complete CANN read/inject backends."""
    registry.register_family("gated_residual", specs=(GATED_RESIDUAL,))
    registry.register_impl(GATED_RESIDUAL, name="native", devices=("cpu", "cuda", "npu"), load=_load_native)
    registry.register_impl(GATED_RESIDUAL, name="cann", devices=("npu",), load=_load_cann)
