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
"""Depthwise causal convolution followed by SiLU."""

from __future__ import annotations

from functools import partial
from typing import Protocol

import torch
from torch.nn import functional as F

from hyper_parallel.components.functional.linear_attention_convolution import fused_causal_conv1d
from hyper_parallel.components.functional.linear_attention_packed import (
    PackedSequenceInfo,
    packed_causal_conv1d_reference,
)

from .._dependency import require_package, require_symbol
from ..registry import OpRegistry
from ..spec import OpSpec


class CausalConv1dFn(Protocol):
    """Apply causal depthwise convolution to dense or document-packed tokens."""

    def __call__(
        self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None,
        info: PackedSequenceInfo | None = None, *, dilation: int = 1,
    ) -> torch.Tensor:
        """Apply [C,1,W] weights and SiLU, resetting at packed document boundaries."""


CAUSAL_CONV1D = OpSpec[CausalConv1dFn](config_key="causal_conv1d")


def _native(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None,
    info: PackedSequenceInfo | None = None, *, dilation: int = 1,
) -> torch.Tensor:
    if info is not None:
        return packed_causal_conv1d_reference(x, weight, bias, info, dilation=dilation)
    padding = (weight.shape[-1] - 1) * dilation
    output = torch.conv1d(x.transpose(1, 2), weight, bias, padding=padding, dilation=dilation, groups=x.shape[-1])
    return F.silu(output[:, :, :x.shape[1]].transpose(1, 2))


def _load_triton() -> CausalConv1dFn:
    require_package("triton.backends.ascend")
    require_symbol(require_package("fla.modules.conv"), "causal_conv1d")
    return partial(fused_causal_conv1d, backend="triton")


def _load_ascendc() -> CausalConv1dFn:
    kernels = require_package("fla_npu.ops.ascendc")
    require_symbol(kernels, "causal_conv1d_fn")
    require_symbol(kernels, "causal_conv1d_bwd")
    return partial(fused_causal_conv1d, backend="ascendc")


def register_causal_conv1d(registry: OpRegistry) -> None:
    """Register dense and packed short convolution with a shared Torch reference."""
    registry.register_family("causal_conv1d", specs=(CAUSAL_CONV1D,))
    registry.register_impl(CAUSAL_CONV1D, name="native", devices=("cpu", "cuda", "npu"), load=lambda: _native)
    registry.register_impl(CAUSAL_CONV1D, name="triton_ascend", devices=("npu",), load=_load_triton)
    registry.register_impl(CAUSAL_CONV1D, name="ascendc", devices=("npu",), load=_load_ascendc)
