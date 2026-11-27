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
from math import isfinite
import torch
from torch import nn
from torch.nn import functional as F
from hyper_parallel.components.ops import mhc_post, sinkhorn
from hyper_parallel.core.dtensor.dtensor import DTensor
from ._utils import _validate_settings


class _UnweightedRMSNorm(nn.Module):
    """Normalize coefficient predictors without introducing an extra learned scale."""

    def __init__(self, eps: float) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Preserve the source FP32 variance and input-dtype reciprocal scaling order."""
        return values * torch.rsqrt(values.float().square().mean(-1, keepdim=True) + self.eps).to(values.dtype)


class PipelinedMhcModule(nn.Module):
    """Coefficient module for cross-sublayer pipelined mHC."""

    def __init__(
        self,
        *,
        module: nn.Module | None = None,
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
        config: Any = None,
        token_tile: int = 32,
    ) -> None:
        """Reuse the source module's ``fn/base/scale`` parameter layout."""
        super().__init__()
        del module_fqn, context
        if isinstance(token_tile, bool) or not isinstance(token_tile, int) or token_tile < 1:
            raise ValueError("token_tile must be a positive integer")
        self.token_tile = token_tile
        if (module is None) == (config is None):
            raise ValueError("provide exactly one of config or module")
        source = config if config is not None else module
        self.init_scale = getattr(source, "init_scale", 0.01 if config is not None else 1.0)
        self.init_method_std = getattr(source, "init_method_std", 0.02)
        if (isinstance(self.init_scale, bool) or not isinstance(self.init_scale, (int, float))
                or not isfinite(self.init_scale)):
            raise ValueError("init_scale must be finite")
        if (isinstance(self.init_method_std, bool) or not isinstance(self.init_method_std, (int, float))
                or not isfinite(self.init_method_std) or self.init_method_std <= 0):
            raise ValueError("init_method_std must be positive and finite")
        if config is not None:
            self._init_from_config(config)
        else:
            self._adopt_source(module)

    def _init_from_config(self, config: Any) -> None:
        """Allocate coefficient state with the declared normalization and initialization policy."""
        _validate_settings(config.num_stream, config.hidden_size, config.sinkhorn_iters,
                           config.hc_eps, config.norm_eps)
        self.hc_mult = config.num_stream
        width = config.hidden_size * config.num_stream
        count = (config.num_stream + 2) * config.num_stream
        self.input_norm = (nn.RMSNorm(width, eps=config.norm_eps, dtype=torch.float32)
                           if getattr(config, "use_gamma", True) else _UnweightedRMSNorm(config.norm_eps))
        self.fn = nn.Parameter(torch.empty(count, width, dtype=torch.float32))
        self.base = nn.Parameter(torch.empty(count, dtype=torch.float32))
        self.scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_sinkhorn_iters, self.hc_eps = config.sinkhorn_iters, config.hc_eps
        self.reset_parameters()

    def _adopt_source(self, module: nn.Module) -> None:
        """Adopt source coefficient matrices and validate stream-divisible mixing dimensions."""
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

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Initialize direct-build coefficient parameters after materialization."""
        # Adopted parameter-free norms may not expose an initialization hook.
        reset_norm = getattr(self.input_norm, "reset_parameters", None)
        if callable(reset_norm):
            reset_norm()
        nn.init.normal_(self.fn, std=self.init_method_std)
        nn.init.zeros_(self.base)
        self.scale.fill_(self.init_scale)

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Produce the coefficients consumed by the following sublayer.

        Args:
            hidden_streams: Residual streams with trailing stream and hidden dimensions.

        Returns:
            Pre-mixing, post-mixing, and residual-mixing coefficients.
        """
        flattened = self.input_norm(hidden_streams.flatten(start_dim=2).float())
        mix = F.linear(  # pylint: disable=not-callable
            flattened, self.fn.float()
        )
        num_stream = self.hc_mult
        pre, post, residual = mix.split(
            [num_stream, num_stream, num_stream * num_stream], dim=-1
        )
        pre_bias, post_bias, residual_bias = self.base.split(
            [num_stream, num_stream, num_stream * num_stream]
        )
        pre_scale, post_scale, residual_scale = self.scale.unbind(0)
        pre = torch.sigmoid(pre * pre_scale + pre_bias) + self.hc_eps
        post = 2 * torch.sigmoid(post * post_scale + post_bias)
        residual = residual.view(*residual.shape[:-1], num_stream, num_stream)
        residual = residual * residual_scale + residual_bias.view(num_stream, num_stream)
        residual = sinkhorn(residual, self.hc_sinkhorn_iters, self.hc_eps)
        return pre, post, residual


def pipelined_mhc_post(
    sublayer_output: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    combine: torch.Tensor,
) -> torch.Tensor:
    """Apply mHC post locally and retain the input batch/sequence placements.

    Args:
        sublayer_output: Sublayer output with batch/sequence-local rows.
        residual: Residual streams with replicated stream and feature dimensions.
        post: Output mixing coefficients with the same batch/sequence layout.
        combine: Residual mixing matrix with the same batch/sequence layout.

    Returns:
        Mixed streams retaining the input batch/sequence DTensor placements.
    """
    inputs = (sublayer_output, residual, post, combine)
    reference = next((value for value in inputs if isinstance(value, DTensor)), None)
    if reference is not None:
        # Native bindings return local tensors; keep their reshape and autograd
        # inside one local boundary before restoring the distributed metadata.
        sublayer_output, residual, post, combine = (
            value.to_local() if isinstance(value, DTensor) else value for value in inputs
        )
    num_stream = residual.shape[-2]
    flattened = mhc_post(
        sublayer_output,
        residual.flatten(start_dim=2),
        post,
        combine,
        num_stream,
    )
    output = flattened.unflatten(-1, (num_stream, residual.shape[-1]))
    if reference is not None:
        return DTensor.from_local(output, reference.device_mesh, reference.placements)
    return output


class ShiftedPipelinedMhcModule(PipelinedMhcModule):
    """Advance a pending residual update, next coefficients and next sublayer normalization together."""

    fp32_parameter_names = ("fn", "base", "scale")

    def __init__(self, *, config: Any = None, module: nn.Module | None = None,
                 module_fqn: str = "", context: Mapping[str, Any] | None = None,
                 token_tile: int = 32) -> None:
        """Retain the ordinary coefficient state keys for the shifted training protocol."""
        super().__init__(config=config, module=module, module_fqn=module_fqn, context=context, token_tile=token_tile)
        if tuple(self.input_norm.parameters()):
            raise ValueError("Shifted pipelined mHC requires unweighted coefficient normalization")
        if self.hc_mult != 4 or self.hc_sinkhorn_iters != 20:
            raise ValueError("Shifted pipelined mHC requires four streams and twenty Sinkhorn iterations")
        if isinstance(token_tile, bool) or not isinstance(token_tile, int) or token_tile < 1:
            raise ValueError("token_tile must be a positive integer")
        self.token_tile = token_tile
        self.norm_eps = getattr(self.input_norm, "eps", getattr(self.input_norm, "variance_epsilon", None))
        if self.norm_eps is None:
            raise ValueError("Shifted pipelined mHC requires the source normalization epsilon")

    def _apply(self, fn: Any, recurse: bool = True) -> nn.Module:
        """Keep coefficient, normalization and gradient storage in FP32 during dtype conversion."""
        protected = {id(parameter) for parameter in self.parameters()}
        protected.update(id(parameter.grad) for parameter in self.parameters() if parameter.grad is not None)

        def preserve(tensor: torch.Tensor) -> torch.Tensor:
            """Keep protected coefficients in FP32 without a lower-precision round trip."""
            converted = fn(tensor)
            if id(tensor) not in protected or converted.dtype == torch.float32:
                return converted
            return converted.float() if tensor.is_meta else tensor.to(device=converted.device, dtype=torch.float32)

        return super()._apply(preserve, recurse=recurse)

    def advance(self, previous_output: torch.Tensor, residual: torch.Tensor,
                previous_pre_mix: torch.Tensor, previous_post_mix: torch.Tensor,
                previous_residual_mix: torch.Tensor, norm_weight: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Evaluate the source shifted boundary with ordinary differentiable Torch operations."""
        updated = pipelined_mhc_post(previous_output, residual, previous_post_mix, previous_residual_mix)
        pre, post, mixing = self(updated)
        collapsed = (previous_pre_mix.unsqueeze(-1) * updated.float()).sum(dim=2)
        reciprocal_std = torch.rsqrt(collapsed.square().mean(dim=-1, keepdim=True) + self.norm_eps)
        block_input = (collapsed * reciprocal_std).to(residual.dtype) * norm_weight
        return updated, pre, post, mixing, block_input


class NativeShiftedPipelinedMhcModule(ShiftedPipelinedMhcModule):
    """Use the native multicore kernel with the same coefficient and external-normalization ownership."""

    def advance(self, previous_output: torch.Tensor, residual: torch.Tensor,
                previous_pre_mix: torch.Tensor, previous_post_mix: torch.Tensor,
                previous_residual_mix: torch.Tensor, norm_weight: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Execute the explicitly selected NPU kernel without changing state keys or silently falling back."""
        if residual.device.type != "npu":
            raise ValueError("Native shifted pipelined mHC requires NPU tensors")
        # The multicore dependency is optional until its execution member is called.
        from hyper_multicore.modules.mega_mhc.function import hyper_mega_mhc  # pylint: disable=import-outside-toplevel

        return hyper_mega_mhc(previous_output, residual, previous_pre_mix, previous_post_mix,
                              previous_residual_mix, self.fn.float(), self.scale.float(), self.base.float(),
                              norm_weight, hc_eps=self.hc_eps, norm_eps=self.norm_eps,
                              num_iters=self.hc_sinkhorn_iters, token_tile=self.token_tile)
