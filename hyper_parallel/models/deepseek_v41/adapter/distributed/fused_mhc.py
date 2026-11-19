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
"""Independent native Pre/Post operators with the V4.1 delayed-pre protocol."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import math
from typing import Any

import torch
from torch import nn

from hyper_parallel.components.functional.mhc_coefficients import mhc_coefficients
from hyper_parallel.components.functional.mhc_post import mhc_post, weighted_stream_sum
from hyper_parallel.components.functional.rms_norm import rms_norm
from hyper_parallel.components.functional.sinkhorn import sinkhorn
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41PipelinedHyperConnection
from hyper_parallel.models.replacement import module_replacement


@module_replacement
class DeepseekV41FusedMhc(DeepseekV41PipelinedHyperConnection):
    """Preserve checkpoint parameters and shifted state without HyperMulticore.

    Native Pre predicts coefficients; Sinkhorn, delayed pre aggregation,
    input RMSNorm and Post use separate torch-npu operators. No padding
    or HyperMulticore dependency is required.
    """

    fp32_parameter_names = ("fn", "base", "scale")

    def __init__(self, *, module: nn.Module, module_fqn: str = "",
                 context: Mapping[str, Any] | None = None) -> None:
        """Adopt existing parameters and validate the native coefficient contract."""
        del module_fqn, context
        super().__init__(module)
        if self.hc_mult != 4 or self.hc_sinkhorn_iters != 20:
            raise ValueError("Fused V4.1 mHC requires four streams and 20 Sinkhorn iterations")
        self.norm_eps = getattr(self.input_norm, "variance_epsilon", getattr(self.input_norm, "eps", None))
        if (isinstance(self.norm_eps, bool) or not isinstance(self.norm_eps, (int, float))
                or not math.isfinite(self.norm_eps) or self.norm_eps <= 0):
            raise ValueError("Fused V4.1 mHC requires a finite positive input_norm epsilon")

    def _apply(
            self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True,
    ) -> DeepseekV41FusedMhc:
        """Preserve FP32 coefficients and gradients while following device conversion."""
        protected = set()
        for name in self.fp32_parameter_names:
            parameter = getattr(self, name)
            protected.add(id(parameter))
            if parameter.grad is not None:
                protected.add(id(parameter.grad))

        def apply(tensor: torch.Tensor) -> torch.Tensor:
            """Avoid rounding protected values through the requested lower precision."""
            converted = fn(tensor)
            if id(tensor) not in protected or converted.dtype == torch.float32:
                return converted
            if tensor.is_meta:
                return converted.float()
            return tensor.to(device=converted.device, dtype=torch.float32)

        return super()._apply(apply, recurse=recurse)

    def forward(self, hidden_streams: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Generate coefficients without consuming the new pre prematurely."""
        if hidden_streams.device.type == "cpu":
            return super().forward(hidden_streams)
        phi = self.fn.float()
        gamma = getattr(self.input_norm, "weight", None)
        if gamma is not None:
            phi = phi * gamma.float().reshape(1, -1)
        pre, post, logits = mhc_coefficients(hidden_streams, phi, self.scale.float(), self.base.float(),
                                          norm_eps=self.norm_eps, hc_eps=self.hc_eps)
        combine = sinkhorn(logits, self.hc_sinkhorn_iters, self.hc_eps, backend="torch_npu")
        return pre, post, combine

    @staticmethod
    def post(output: torch.Tensor, residual: torch.Tensor,
             post: torch.Tensor, combine: torch.Tensor) -> torch.Tensor:
        """Finish a pending Post, including Engram and final decoder boundaries."""
        streams = residual.shape[-2]
        return mhc_post(output, residual.flatten(2), post, combine, streams, backend="torch_npu").unflatten(
            -1, (streams, residual.shape[-1]),
        )

    def advance(self, previous_output: torch.Tensor, residual: torch.Tensor,
                previous_pre_mix: torch.Tensor, previous_post_mix: torch.Tensor,
                previous_residual_mix: torch.Tensor, norm_weight: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Complete Post, predict coefficients, then aggregate using the OLD pre."""
        residual = self.post(previous_output, residual, previous_post_mix, previous_residual_mix)
        pre, post, combine = self(residual)
        mixed = weighted_stream_sum(residual, previous_pre_mix)
        if residual.device.type == "npu":
            block_input = rms_norm(mixed, norm_weight, self.norm_eps)
        else:
            normalized = mixed.float() * torch.rsqrt(mixed.float().square().mean(dim=-1, keepdim=True)
                                                    + self.norm_eps)
            block_input = normalized.to(residual.dtype) * norm_weight
        return residual, pre, post, combine, block_input


__all__ = ["DeepseekV41FusedMhc"]
