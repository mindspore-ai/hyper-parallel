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
"""Packed and DeepSeek FP32 clamped SwiGLU implementations."""

from typing import Protocol

import torch
from torch.nn import functional as F

from ..registry import OpRegistry
from ..spec import OpSpec
from .._dependency import require_package, require_symbol


class SwiGLUFn(Protocol):
    """Apply SiLU(gate) * up to packed Gate/Up projections."""

    def __call__(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Split the packed projection evenly along the given dimension."""


class ClampedSwiGLUFn(Protocol):
    """Compute a FP32 activation from separate, optionally clamped projections."""

    def __call__(self, x: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
        """Clamp the projections for positive limits and return the FP32 result."""


SWIGLU = OpSpec[SwiGLUFn](config_key="swiglu")
SWIGLU_CLAMPED = OpSpec[ClampedSwiGLUFn](config_key="swiglu", variant="clamped_fp32")


def _native(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if x.shape[dim] % 2:
        raise ValueError("SwiGLU requires an evenly packed Gate/Up dimension")
    gate, up = x.chunk(2, dim=dim)
    return F.silu(gate) * up


def _clamp(x: torch.Tensor, up: torch.Tensor, limit: float) -> tuple[torch.Tensor, torch.Tensor]:
    gate, up = x.float(), up.float()
    if limit > 0:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return gate, up


def _native_clamped(x: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    gate, up = _clamp(x, up, limit)
    return F.silu(gate) * up


def _load_native() -> SwiGLUFn:
    return _native


def _load_native_clamped() -> ClampedSwiGLUFn:
    return _native_clamped


def _load_npu() -> SwiGLUFn:
    require_symbol(require_package("torch_npu"), "npu_swiglu")
    # The existing implementation imports the optional NPU package.
    from hyper_parallel.components.functional.swiglu import swiglu  # pylint: disable=import-outside-toplevel
    return swiglu


def _load_npu_clamped() -> ClampedSwiGLUFn:
    kernel = _load_npu()

    def swiglu_clamped(x: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
        """Clamp separate projections in FP32 and apply the packed activation kernel."""
        gate, up = _clamp(x, up, limit)
        return kernel(torch.cat((gate, up), dim=-1), dim=-1)

    return swiglu_clamped


def register_swiglu(registry: OpRegistry) -> None:
    """Register packed and clamped SwiGLU under the same selection key."""
    registry.register_family("swiglu", specs=(SWIGLU, SWIGLU_CLAMPED))
    registry.register_impl(SWIGLU, name="native", devices=("cuda", "cpu", "npu"), load=_load_native)
    registry.register_impl(SWIGLU, name="torch_npu", devices=("npu",), load=_load_npu)
    registry.register_impl(
        SWIGLU_CLAMPED, name="native", devices=("cuda", "cpu", "npu"), load=_load_native_clamped,
    )
    registry.register_impl(SWIGLU_CLAMPED, name="torch_npu", devices=("npu",), load=_load_npu_clamped)
