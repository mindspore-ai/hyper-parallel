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
"""Pipelined mHC coefficient and post computation."""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any
import torch
from torch import nn
from torch.nn import functional as F
from hyper_parallel.components.ops import mhc_post, mhc_pre_apply, mhc_split_mixes, sinkhorn
from ._utils import _validate_settings


class PipelinedMhcModule(nn.Module):
    """Coefficient module for cross-sublayer pipelined mHC."""

    def __init__(
        self,
        *,
        module: nn.Module | None = None,
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
        config: Any = None,
    ) -> None:
        """Reuse the source module's ``fn/base/scale`` parameter layout."""
        super().__init__()
        del module_fqn, context
        if (module is None) == (config is None):
            raise ValueError("provide exactly one of config or module")
        if config is not None:
            _validate_settings(config.num_stream, config.hidden_size, config.sinkhorn_iters,
                               config.hc_eps, config.norm_eps)
            self.hc_mult = config.num_stream
            width = config.hidden_size * config.num_stream
            count = (config.num_stream + 2) * config.num_stream
            self.input_norm = nn.RMSNorm(width, eps=config.norm_eps, dtype=torch.float32)
            self.fn = nn.Parameter(torch.empty(count, width, dtype=torch.float32))
            self.base = nn.Parameter(torch.empty(count, dtype=torch.float32))
            self.scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
            self.hc_sinkhorn_iters, self.hc_eps = config.sinkhorn_iters, config.hc_eps
            self.reset_parameters()
            return
        required = ("input_norm", "fn", "base", "scale", "hc_mult")
        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            raise TypeError(f"pipelined MHC source is missing required attributes: {missing}")
        self.input_norm = module.input_norm
        self.fn = module.fn
        self.base = module.base
        self.scale = module.scale
        if not all(isinstance(value, nn.Parameter) for value in (self.fn, self.base, self.scale)):
            raise TypeError("pipelined MHC fn/base/scale must be Parameters")
        self.hc_mult = int(module.hc_mult)
        self.hc_sinkhorn_iters = int(module.hc_sinkhorn_iters)
        self.hc_eps = float(module.hc_eps)
        expected_mix = (self.hc_mult + 2) * self.hc_mult
        if tuple(self.fn.shape)[0] != expected_mix or self.base.numel() != expected_mix:
            raise ValueError("pipelined MHC fn/base shapes do not match hc_mult")
        if self.scale.numel() != 3:
            raise ValueError("pipelined MHC scale must contain three values")
        if self.fn.ndim != 2 or self.fn.shape[1] % self.hc_mult:
            raise ValueError("pipelined MHC fn must be a matrix with stream-divisible input width")
        _validate_settings(self.hc_mult, self.fn.shape[1] // self.hc_mult,
                              self.hc_sinkhorn_iters, self.hc_eps)
        self.train(module.training)

    def reset_parameters(self) -> None:
        """Initialize direct-build coefficient parameters after materialization."""
        self.input_norm.reset_parameters()
        nn.init.normal_(self.fn, std=0.02)
        nn.init.zeros_(self.base)
        nn.init.constant_(self.scale, 0.01)

    def make_transforms(self) -> list:
        """Declare that this member keeps parameter names and layouts unchanged.

        Returns:
            An empty converter list, matching members whose initialization is
            fully owned by ``reset_parameters``.
        """
        return []

    def forward(
        self,
        hidden_streams: torch.Tensor,
        pre_mix: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Produce next coefficients, optionally collapsing streams with the supplied pre mix."""
        flattened = self.input_norm(hidden_streams.flatten(start_dim=2).float())
        mix = F.linear(  # pylint: disable=not-callable
            flattened, self.fn.float()
        )
        pre, post, residual = mhc_split_mixes(mix, self.scale, self.base, self.hc_mult, epsilon=self.hc_eps)
        residual = sinkhorn(residual, self.hc_sinkhorn_iters, self.hc_eps)
        if pre_mix is not None:
            return mhc_pre_apply(hidden_streams, pre_mix), pre, post, residual
        return pre, post, residual


def pipelined_mhc_post(
    sublayer_output: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    combine: torch.Tensor,
) -> torch.Tensor:
    """Apply the shared high-performance mHC post path to 4-D streams."""
    num_stream = residual.shape[-2]
    flattened = mhc_post(
        sublayer_output,
        residual.flatten(start_dim=2),
        post,
        combine,
        num_stream,
    )
    return flattened.unflatten(-1, (num_stream, residual.shape[-1]))
