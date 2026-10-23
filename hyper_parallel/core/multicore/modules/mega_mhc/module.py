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

from collections.abc import Callable
from typing import Any

import torch

from .._precision import preserve_fp32_parameters
from ..mega_mhc_common import (
    MHC_GRAD_MAX_HIDDEN_SIZE,
    MHC_GRAD_MIN_TOKENS,
    MHC_MAPPINGS,
    MHC_SINKHORN_ITERS,
    MHC_STREAMS,
    validate_mega_mhc_hidden_size,
)
from ..mega_mhc_grad.function import _resolve_core_counts
from .function import hyper_mega_mhc
from .graph import DEFAULT_TOKEN_TILE
from .state import MegaMhcOutput, MhcState


class MegaMhc(torch.nn.Module):
    """Own the current layer's mapping predictor and input RMSNorm weight.

    Note:
        This trainable module requires the native backward hidden-size contract.
        Forward-only callers can use ``hyper_mega_mhc(..., need_backward=False)``.
    """

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
        validate_mega_mhc_hidden_size(hidden_size, need_backward=True)
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

    def _apply(self, fn: Callable, recurse: bool = True) -> MegaMhc:
        """Keep mapping predictors FP32 when the enclosing model changes dtype or device."""
        return super()._apply(preserve_fp32_parameters(self, fn, ("phi", "alpha", "bias")), recurse=recurse)

    @classmethod
    def from_module(cls, module: torch.nn.Module) -> MegaMhc:
        """Bind an existing four-stream, shifted mHC module without replacing its Parameters.

        Args:
            module: Source with hidden_size, hc_eps, norm_eps, num_iters and Parameters
                phi [24, 4H], alpha [3], bias [24], norm_weight [H]. Predictor Parameters must be FP32.

        Returns:
            Boundary sharing all original Parameters and mathematical configuration.
        """
        configuration = {}
        for name in ("hidden_size", "hc_eps", "norm_eps", "num_iters"):
            if not hasattr(module, name):
                raise ValueError(f"source mHC module must expose {name}")
            configuration[name] = getattr(module, name)
        phi = getattr(module, "phi", None)
        if not isinstance(phi, torch.nn.Parameter):
            raise TypeError("source phi must be an FP32 Parameter [24, 4H]")
        result = cls(**configuration, device=phi.device)
        for name in ("phi", "alpha", "bias", "norm_weight"):
            parameter = getattr(module, name, None)
            expected = getattr(result, name)
            if (not isinstance(parameter, torch.nn.Parameter) or parameter.shape != expected.shape
                    or parameter.dtype != expected.dtype or parameter.device != expected.device):
                raise ValueError(f"source {name} must match shape={tuple(expected.shape)}, dtype={expected.dtype}")
            setattr(result, name, parameter)
        return result

    @staticmethod
    def execution_requirements(device: Any | None = None) -> dict[str, Any]:
        """Report native shape and precision limits, including the selected chip's training minimum.

        Args:
            device: NPU device to query. Without one, return generic limits with no chip-specific minimum.

        Returns:
            Native support conditions. A None minimum means the NPU has not been queried.
        """
        minimum = None
        if device is not None and torch.device(device).type == "npu":
            from_index = torch.device(device).index
            from_index = torch.npu.current_device() if from_index is None else from_index
            _, vectors = _resolve_core_counts(from_index)
            minimum = max(MHC_GRAD_MIN_TOKENS, vectors)
        return {"device_type": "npu", "activation_dtype": torch.bfloat16, "mapping_dtype": torch.float32,
                "streams": MHC_STREAMS, "sinkhorn_iterations": MHC_SINKHORN_ITERS,
                "hidden_size_multiple": 128, "maximum_training_hidden_size": MHC_GRAD_MAX_HIDDEN_SIZE,
                "minimum_training_tokens": minimum}

    def forward(
        self,
        previous_output: torch.Tensor | MhcState,
        residual: torch.Tensor | None = None,
        previous_pre_mix: torch.Tensor | None = None,
        previous_post_mix: torch.Tensor | None = None,
        previous_residual_mix: torch.Tensor | None = None,
    ) -> MegaMhcOutput:
        """Advance one shifted mHC block boundary and predict the next mappings.

        Args:
            previous_output: Explicit MhcState, or previous block output for the legacy five-tensor call.
            residual: Previous residual streams.
            previous_pre_mix: Previous pre-mix coefficients.
            previous_post_mix: Previous post-mix coefficients.
            previous_residual_mix: Previous residual-mix coefficients.
        """
        if isinstance(previous_output, MhcState):
            individual = (residual, previous_pre_mix, previous_post_mix, previous_residual_mix)
            if any(value is not None for value in individual):
                raise ValueError("Pass either MhcState or five individual state tensors")
            previous_output, residual, previous_pre_mix, previous_post_mix, previous_residual_mix = previous_output
        return MegaMhcOutput(*hyper_mega_mhc(
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
        ))


HyperMegaMhc = MegaMhc
