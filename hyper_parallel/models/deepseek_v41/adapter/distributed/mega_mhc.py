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
"""Declarative HyperMegaMhc replacement for DeepSeek-V4.1 training."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.modules.mhc import pipelined_mhc_post
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import (
    DeepseekV41PipelinedHyperConnection,
)
from hyper_parallel.models.replacement import module_replacement


def _rms_norm_eps(module: nn.Module) -> float:
    """Read the source coefficient RMSNorm epsilon."""
    value = getattr(module, "variance_epsilon", getattr(module, "eps", None))
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("DeepSeek-V4.1 mHC input_norm must expose a finite positive epsilon")
    return float(value)


def _reference_shifted_boundary(
        module: DeepseekV41PipelinedHyperConnection,
        previous_output: torch.Tensor,
        residual: torch.Tensor,
        previous_pre_mix: torch.Tensor,
        previous_post_mix: torch.Tensor,
        previous_residual_mix: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
) -> tuple[torch.Tensor, ...]:
    """Execute the shifted protocol with ordinary Torch operations for CPU tests."""
    new_residual = pipelined_mhc_post(
        previous_output,
        residual,
        previous_post_mix,
        previous_residual_mix,
    )
    next_pre, next_post, next_residual_mix = module(new_residual)
    mixed_input = (previous_pre_mix.unsqueeze(-1) * new_residual.float()).sum(dim=2)
    reciprocal_std = torch.rsqrt(mixed_input.square().mean(dim=-1, keepdim=True) + norm_eps)
    block_input = (mixed_input * reciprocal_std).to(residual.dtype) * norm_weight
    return new_residual, next_pre, next_post, next_residual_mix, block_input


@module_replacement
class DeepseekV41HyperMegaMhc(DeepseekV41PipelinedHyperConnection):
    """Retain V4.1 parameter names while executing a shifted MegaKernel boundary."""

    def __init__(
            self,
            *,
            module: nn.Module,
            module_fqn: str = "",
            context: Mapping[str, Any] | None = None,
            token_tile: int = 32,
    ) -> None:
        """Adopt source parameters and validate the initial fused-kernel contract."""
        del module_fqn
        if any((context or {}).get(axis, False) for axis in ("tp", "cp", "pp")):
            raise ValueError("DeepSeek-V4.1 HyperMegaMhc requires TP=CP=PP=1")
        if isinstance(token_tile, bool) or not isinstance(token_tile, int) or token_tile <= 0:
            raise ValueError("token_tile must be a positive integer")
        super().__init__(module)
        if self.hc_mult != 4:
            raise ValueError(f"HyperMegaMhc requires hc_mult=4, got {self.hc_mult}")
        if self.hc_sinkhorn_iters != 20:
            raise ValueError(
                "HyperMegaMhc requires hc_sinkhorn_iters=20, "
                f"got {self.hc_sinkhorn_iters}"
            )
        self.norm_eps = _rms_norm_eps(self.input_norm)
        self.token_tile = token_tile

    def advance(
            self,
            previous_output: torch.Tensor,
            residual: torch.Tensor,
            previous_pre_mix: torch.Tensor,
            previous_post_mix: torch.Tensor,
            previous_residual_mix: torch.Tensor,
            norm_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Advance one shifted mHC boundary and predict the next mappings."""
        if residual.device.type == "cpu":
            return _reference_shifted_boundary(
                self,
                previous_output,
                residual,
                previous_pre_mix,
                previous_post_mix,
                previous_residual_mix,
                norm_weight,
                self.norm_eps,
            )
        if residual.device.type != "npu":
            raise TypeError("DeepSeek-V4.1 HyperMegaMhc supports only NPU execution")
        # Multicore is an optional native dependency unless this replacement is selected.
        from hyper_parallel.core.multicore.modules.mega_mhc.function import (  # pylint: disable=C0415
            hyper_mega_mhc,
        )

        return hyper_mega_mhc(
            previous_output,
            residual,
            previous_pre_mix,
            previous_post_mix,
            previous_residual_mix,
            self.fn.float(),
            self.scale.float(),
            self.base.float(),
            norm_weight,
            hc_eps=self.hc_eps,
            norm_eps=self.norm_eps,
            num_iters=self.hc_sinkhorn_iters,
            token_tile=self.token_tile,
        )


__all__ = ["DeepseekV41HyperMegaMhc"]
