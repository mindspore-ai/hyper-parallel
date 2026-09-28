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
"""Model-facing Torch module for shifted BF16 HyperMegaMhc."""

from __future__ import annotations

from typing import Any

import torch

from .function import hyper_mega_mhc
from .golden import MHC_MAPPINGS, MHC_SINKHORN_ITERS, MHC_STREAMS, MegaMhcOutputs
from .graph import DEFAULT_TOKEN_TILE


class HyperMegaMhc(torch.nn.Module):
    """Own the current layer's mapping predictor and input RMSNorm weight."""

    def __init__(
        self,
        hidden_size: int,
        *,
        hc_eps: float = 1e-6,
        norm_eps: float = 1e-6,
        num_iters: int = MHC_SINKHORN_ITERS,
        token_tile: int = DEFAULT_TOKEN_TILE,
        device: Any | None = None,
    ) -> None:
        """Initialize the fixed-four-stream mHC parameters."""
        super().__init__()
        if not isinstance(hidden_size, int) or isinstance(hidden_size, bool) or hidden_size <= 0:
            raise ValueError(f"hidden_size must be a positive integer, got {hidden_size!r}.")
        if hidden_size % 128:
            raise ValueError(f"hidden_size must be divisible by 128, got {hidden_size}.")
        if num_iters != MHC_SINKHORN_ITERS:
            raise ValueError(f"initial HyperMegaMhc requires num_iters={MHC_SINKHORN_ITERS}, got {num_iters}.")
        if not isinstance(token_tile, int) or isinstance(token_tile, bool) or token_tile <= 0:
            raise ValueError(f"token_tile must be a positive integer, got {token_tile!r}.")
        self.hidden_size = hidden_size
        self.hc_eps = float(hc_eps)
        self.norm_eps = float(norm_eps)
        self.num_iters = num_iters
        self.token_tile = token_tile
        factory_kwargs = {"device": device, "dtype": torch.float32}
        self.phi = torch.nn.Parameter(
            torch.empty((MHC_MAPPINGS, MHC_STREAMS * hidden_size), **factory_kwargs)
        )
        self.alpha = torch.nn.Parameter(torch.ones((3,), **factory_kwargs))
        self.bias = torch.nn.Parameter(torch.zeros((MHC_MAPPINGS,), **factory_kwargs))
        self.norm_weight = torch.nn.Parameter(
            torch.ones((hidden_size,), device=device, dtype=torch.bfloat16)
        )
        torch.nn.init.normal_(
            self.phi,
            mean=0.0,
            std=(MHC_STREAMS * hidden_size) ** -0.5,
        )

    def forward(
        self,
        previous_output: torch.Tensor,
        residual: torch.Tensor,
        previous_pre_mix: torch.Tensor,
        previous_post_mix: torch.Tensor,
        previous_residual_mix: torch.Tensor,
    ) -> MegaMhcOutputs:
        """Advance one shifted mHC block boundary and predict the next mappings.

        Args:
            previous_output: Previous block output.
            residual: Previous residual streams.
            previous_pre_mix: Previous pre-mix coefficients.
            previous_post_mix: Previous post-mix coefficients.
            previous_residual_mix: Previous residual-mix coefficients.
        """
        return hyper_mega_mhc(
            previous_output,
            residual,
            previous_pre_mix,
            previous_post_mix,
            previous_residual_mix,
            self.phi,
            self.alpha,
            self.bias,
            self.norm_weight,
            hc_eps=self.hc_eps,
            norm_eps=self.norm_eps,
            num_iters=self.num_iters,
            token_tile=self.token_tile,
        )
