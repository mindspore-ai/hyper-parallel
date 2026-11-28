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

from types import SimpleNamespace
from typing import Protocol

import torch
from torch.nn import functional as F

from hyper_parallel.components.functional.linear_attention_convolution import fused_causal_conv1d, packed_causal_conv1d
from hyper_parallel.components.modules.attention.linear_attention_utils.sequence_layout import (
    PackedSequenceInfo,
)

from .._dependency import require_package, require_symbol
from ..factory import _select_for_call
from ..registry import OpRegistry
from ..spec import OpSpec


class CausalConv1dFn(Protocol):
    """Apply a depthwise causal convolution followed by SiLU.

    The final filter tap multiplies the current token. Earlier taps use left
    zero padding and the requested dilation. Packed documents reset convolution
    history at every local boundary; there is no history exchange across ranks.
    """

    def __call__(
        self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None,
        info: PackedSequenceInfo | None = None, *, dilation: int = 1,
    ) -> torch.Tensor:
        """Apply [C,1,W] weights and SiLU, resetting at packed document boundaries.

        Args:
            x: Token states [batch, sequence, channels]; packed inputs use batch one.
            weight: Depthwise filter [channels, 1, width].
            bias: Optional per-channel bias [channels].
            info: Optional local packed-document metadata matching the token sequence.
            dilation: Positive spacing between adjacent filter taps.

        Returns:
            Convolved and activated states with the input layout.
        """


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    info: PackedSequenceInfo | None = None,
    *,
    dilation: int = 1,
) -> torch.Tensor:
    """Apply local depthwise causal convolution and SiLU to dense or packed tokens.

    Args:
        x: Token states [batch, sequence, channels]; packed inputs use batch one.
        weight: Depthwise filter [channels, 1, width].
        bias: Optional per-channel bias [channels].
        info: Optional local packed-document metadata matching the token sequence.
        dilation: Positive spacing between adjacent filter taps.

    Returns:
        Convolved and activated states with the input layout.

    Notes:
        Apply a depthwise causal convolution followed by SiLU.

        The final filter tap multiplies the current token. Earlier taps use left
        zero padding and the requested dilation. Packed documents reset convolution
        history at every local boundary; there is no history exchange across ranks.
    """
    operation = _select_for_call(CAUSAL_CONV1D, device=x.device)
    return operation(x, weight, bias, info, dilation=dilation)


CAUSAL_CONV1D = OpSpec[CausalConv1dFn](config_key='causal_conv1d')


def _native(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None,
    info: PackedSequenceInfo | None = None, *, dilation: int = 1,
) -> torch.Tensor:
    """Apply native causal depthwise convolution and SiLU with document resets."""
    if info is not None:
        convolution = SimpleNamespace(
            groups=x.shape[-1], in_channels=x.shape[-1], out_channels=x.shape[-1], stride=(1,),
            kernel_size=(weight.shape[-1],), dilation=(dilation,), weight=weight, bias=bias,
        )
        return packed_causal_conv1d(x, convolution, info, weight=weight, bias=bias)
    padding = (weight.shape[-1] - 1) * dilation
    output = torch.conv1d(x.transpose(1, 2), weight, bias, padding=padding, dilation=dilation, groups=x.shape[-1])
    return F.silu(output[:, :, :x.shape[1]].transpose(1, 2))


def _load_triton() -> CausalConv1dFn:
    require_package("triton.backends.ascend")
    require_symbol(require_package("fla.modules.conv"), "causal_conv1d")
    return _implementation("triton")


def _load_ascendc() -> CausalConv1dFn:
    kernels = require_package("fla_npu.ops.ascendc")
    require_symbol(kernels, "causal_conv1d_fn")
    require_symbol(kernels, "causal_conv1d_bwd")
    return _implementation("ascendc")


def register_causal_conv1d(registry: OpRegistry) -> None:
    """Register dense and packed short convolution with a shared Torch reference."""
    registry.register_family("causal_conv1d", specs=(CAUSAL_CONV1D,))
    registry.register_impl(CAUSAL_CONV1D, name="native", devices=("cpu", "cuda", "npu"), load=lambda: _native)
    registry.register_impl(CAUSAL_CONV1D, name="triton_ascend", devices=("npu",), load=_load_triton)
    registry.register_impl(CAUSAL_CONV1D, name="ascendc", devices=("npu",), load=_load_ascendc)


def _validate_fused_inputs(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None, *, dilation: int = 1, backend: str
) -> None:
    """Reject unsupported inputs in the selected fused implementation."""
    reason = 'fused convolution requires nonempty batches and dilation one' if dilation != 1 or not x.numel() else None
    blocked = ()
    if (
        weight.shape[-1] not in (2, 3, 4)
        or x.shape[-1] % 16
        or x.dtype not in (torch.bfloat16, torch.float16)
        or (weight.dtype != x.dtype)
        or (bias is not None and bias.dtype != x.dtype)
    ):
        blocked = (('ascendc', 'requires width 2/3/4, 16-aligned channels and matching BF16/FP16 dtypes'),)
    if reason is not None:
        raise ValueError(reason)
    for name, message in blocked:
        if name == {'triton': 'triton_ascend'}.get(backend, backend):
            raise ValueError(message)


def _implementation(backend: str) -> CausalConv1dFn:
    """Bind a fused implementation and validate its inputs on every invocation."""

    def run(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        info: PackedSequenceInfo | None = None,
        *,
        dilation: int = 1,
    ) -> torch.Tensor:
        """Execute the selected fused convolution without changing backend."""
        _validate_fused_inputs(x, weight, bias, dilation=dilation, backend=backend)
        return fused_causal_conv1d(x, weight, bias, info, dilation=dilation, backend=backend)

    return run
