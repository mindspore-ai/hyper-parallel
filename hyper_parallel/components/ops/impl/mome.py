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
"""Masked causal depthwise convolution and residual for MOME."""

from __future__ import annotations

from typing import Protocol

import torch
from torch.nn import functional as F

from .._dependency import require_package, require_symbol
from ..registry import OpRegistry
from ..spec import OpSpec


class MomeFn(Protocol):
    """Add causal depthwise-convolution outputs at masked positions."""

    def __call__(
        self, x: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply [C, 1, K] weights to [B, T, C] states and gate the residual update with a [B, T] mask."""


MOME = OpSpec[MomeFn](config_key="mome")


def native_mome(
    x: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Use a depthwise Torch convolution and mask its outputs."""
    padding = weight.shape[-1] - 1
    padded = F.pad(x.transpose(1, 2), (padding, 0))
    mixed = torch.conv1d(padded, weight, bias, groups=x.shape[-1]).transpose(1, 2)
    return x + mixed * mask.unsqueeze(-1).to(mixed.dtype)


def _load_omni() -> MomeFn:
    """Load the optional Omni masked convolution and residual adapter."""
    require_package("omni_training_custom_ops")
    require_symbol(torch.ops.custom, "npu_aggregate_hidden")
    require_symbol(torch.ops.custom, "npu_aggregate_hidden_grad")
    # The autograd bridge imports the optional Omni extension.
    from hyper_parallel.components.functional.aggregate_hidden import (  # pylint: disable=import-outside-toplevel
        aggregate_hidden,
    )

    def run(
        x: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the fused causal update and the reference output mask."""
        padding = weight.shape[-1] - 1
        padded = F.pad(x, (0, 0, padding, 0))
        # The reference convolves all input positions and masks the final update.
        kernel_mask = F.pad(torch.ones_like(mask), (padding, 0), value=False)
        mixed = aggregate_hidden(
            padded.transpose(0, 1).contiguous(), weight.squeeze(1).transpose(0, 1), kernel_mask,
        )[padding:].transpose(0, 1).contiguous()
        if bias is not None:
            mixed = mixed + bias.to(mixed.dtype)
        return x + mixed * mask.unsqueeze(-1).to(mixed.dtype)

    return run


def register_mome(registry: OpRegistry) -> None:
    """Register the Torch reference and the Ascend Omni implementation."""
    registry.register_family("mome", specs=(MOME,))
    registry.register_impl(MOME, name="native", devices=("cpu", "cuda", "npu"), load=lambda: native_mome)
    registry.register_impl(MOME, name="omni", devices=("npu",), load=_load_omni)
