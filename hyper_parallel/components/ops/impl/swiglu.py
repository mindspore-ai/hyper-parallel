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

from ..factory import _select_for_call
from ..registry import OpRegistry
from ..spec import OpSpec
from .._dependency import require_package, require_symbol


class SwiGLUFn(Protocol):
    """Compute SiLU(gate) * up from two evenly packed projection halves.

    Split x into gate and up along dim. This operator performs activation and
    multiplication only; projections and parameters remain with the caller.
    Both packed halves receive gradients. The split dimension must have even size.
    """

    def __call__(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Split the packed projection evenly along the given dimension.

        Args:
            x: Packed gate/up projection with an even size along dim.
            dim: Axis containing the packed projection halves.

        Returns:
            Activated states with the split dimension halved.
        """


class ClampedSwiGLUFn(Protocol):
    """Compute SwiGLU from separate projections in FP32, with optional clamping.

    Cast gate and up to FP32. If limit > 0, clamp gate from above at limit
    and clamp up to [-limit, limit]; otherwise leave both unclamped. Return
    SiLU(gate) * up in FP32. Clamping follows Torch gradient semantics.
    """

    def __call__(self, x: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
        """Clamp the projections for positive limits and return the FP32 result.

        Args:
            x: Gate projection tensor.
            up: Up projection tensor matching x.
            limit: Positive clamp magnitude, or a nonpositive value to disable clamping.

        Returns:
            FP32 activated states with the projection shape.
        """


SWIGLU = OpSpec[SwiGLUFn](config_key='swiglu')
SWIGLU_CLAMPED = OpSpec[ClampedSwiGLUFn](config_key='swiglu', variant='clamped_fp32')


def swiglu(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Split the packed projection evenly along the given dimension.

    Args:
        x: Packed gate/up projection with an even size along dim.
        dim: Axis containing the packed projection halves.

    Returns:
        Activated states with the split dimension halved.

    Notes:
        Compute SiLU(gate) * up from two evenly packed projection halves.

        Split x into gate and up along dim. This operator performs activation and
        multiplication only; projections and parameters remain with the caller.
        Both packed halves receive gradients. The split dimension must have even size.
    """
    operation = _select_for_call(SWIGLU, device=x.device)
    return operation(x, dim)


def swiglu_clamped(x: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    """Clamp the projections for positive limits and return the FP32 result.

    Args:
        x: Gate projection tensor.
        up: Up projection tensor matching x.
        limit: Positive clamp magnitude, or a nonpositive value to disable clamping.

    Returns:
        FP32 activated states with the projection shape.

    Notes:
        Compute SwiGLU from separate projections in FP32, with optional clamping.

        Cast gate and up to FP32. If limit > 0, clamp gate from above at limit
        and clamp up to [-limit, limit]; otherwise leave both unclamped. Return
        SiLU(gate) * up in FP32. Clamping follows Torch gradient semantics.
    """
    operation = _select_for_call(SWIGLU_CLAMPED, device=x.device)
    return operation(x, up, limit)


def _native(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if x.shape[dim] % 2:
        raise ValueError("SwiGLU requires an evenly packed Gate/Up dimension")
    gate, up = x.chunk(2, dim=dim)
    # Packed slices are strided; contiguous SiLU preserves separate-projection
    # backward rounding for source modules that require exact optimizer parity.
    return F.silu(gate.contiguous()) * up


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
    extension = require_package("torch_npu")
    require_symbol(extension, "npu_swiglu")
    kernel = extension.npu_swiglu

    def run(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Preserve native's rounded SiLU intermediate before multiplying by Up."""
        if x.shape[dim] % 2:
            raise ValueError("SwiGLU requires an evenly packed Gate/Up dimension")
        if x.dtype not in (torch.float16, torch.bfloat16):
            return kernel(x, dim=dim)
        gate, up = x.chunk(2, dim=dim)
        # A single fused product rounds only once; native rounds SiLU as well.
        activated = kernel(torch.cat((gate, torch.ones_like(gate)), dim=dim), dim=dim)
        return activated * up

    return run


def _load_npu_clamped() -> ClampedSwiGLUFn:
    kernel = _load_npu()

    def run(x: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
        """Clamp separate projections in FP32 and apply the packed activation kernel."""
        gate, up = _clamp(x, up, limit)
        return kernel(torch.cat((gate, up), dim=-1), dim=-1)

    return run


def register_swiglu(registry: OpRegistry) -> None:
    """Register packed and clamped SwiGLU under the same selection key."""
    registry.register_family("swiglu", specs=(SWIGLU, SWIGLU_CLAMPED))
    registry.register_impl(SWIGLU, name="native", devices=("cuda", "cpu", "npu"), load=_load_native)
    registry.register_impl(SWIGLU, name="torch_npu", devices=("npu",), load=_load_npu)
    registry.register_impl(
        SWIGLU_CLAMPED, name="native", devices=("cuda", "cpu", "npu"), load=_load_native_clamped,
    )
    registry.register_impl(SWIGLU_CLAMPED, name="torch_npu", devices=("npu",), load=_load_npu_clamped)
