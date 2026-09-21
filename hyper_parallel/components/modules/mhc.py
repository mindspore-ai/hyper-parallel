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
"""Reusable manifold-constrained hyper-connection replacement modules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
import torch.nn.functional as F  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.functional.mhc_post import mhc_post
from hyper_parallel.components.functional.mhc_pre import mhc_pre
from hyper_parallel.components.functional.sinkhorn import sinkhorn, sinkhorn_knopps
from hyper_parallel.models.replacement import module_replacement


def _required_attribute(module: nn.Module, name: str) -> Any:
    """Return a required source-module attribute."""
    if not hasattr(module, name):
        raise TypeError(f"{module.__class__.__name__} is missing required attribute '{name}'")
    return getattr(module, name)


def _required_config_value(module: nn.Module, name: str) -> Any:
    """Return a required value from the source module or its config."""
    if hasattr(module, name):
        return getattr(module, name)
    config = getattr(module, "config", None)
    if config is None or not hasattr(config, name):
        raise TypeError(f"{module.__class__.__name__} is missing required MHC setting '{name}'")
    return getattr(config, name)


def _num_stream(module: nn.Module) -> int:
    """Return and validate the source module's residual-stream count."""
    value = getattr(module, "num_stream", None)
    if value is None:
        value = _required_config_value(module, "mhc_num_stream")
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"MHC num_stream must be a positive integer, but got {value}")
    return value


def _parameter(module: nn.Module, name: str) -> nn.Parameter:
    """Return a required source parameter."""
    value = _required_attribute(module, name)
    if not isinstance(value, nn.Parameter):
        raise TypeError(f"MHC attribute '{name}' must be an nn.Parameter")
    return value


def _validate_parameter_layout(phi: nn.Linear, parameters: tuple[nn.Parameter, ...]) -> None:
    """Validate that MHC parameters share the projection's device and dtype."""
    for parameter in parameters:
        if parameter.device != phi.weight.device or parameter.dtype != phi.weight.dtype:
            raise ValueError("MHC projection and branch parameters must share device and dtype")


@module_replacement
class MhcPreModule(nn.Module):
    """Prepare hidden states and mixing coefficients for an MHC-wrapped block."""

    def __init__(
        self,
        *,
        module: nn.Module,
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
    ) -> None:
        """Build the high-performance MHC pre module from an existing module.

        Args:
            module: Source MHC pre module with the existing parameter layout.
            module_fqn: Fully qualified source-module name supplied by replacement.
            context: Replacement context supplied by Trainer.

        Raises:
            TypeError: If required source attributes or settings are missing.
            ValueError: If parameter shapes or runtime settings are incompatible.
        """
        super().__init__()
        del module_fqn, context
        self.config = getattr(module, "config", None)
        self.num_stream = _num_stream(module)
        self.layer_number = getattr(module, "layer_number", 1)

        phi = _required_attribute(module, "phi")
        if not isinstance(phi, nn.Linear) or phi.bias is not None:
            raise TypeError("MhcPreModule requires a bias-free nn.Linear phi projection")
        if phi.in_features % self.num_stream != 0:
            raise ValueError("MHC pre phi input size must be divisible by num_stream")
        if phi.out_features != (self.num_stream + 2) * self.num_stream:
            raise ValueError("MHC pre phi output size must equal (num_stream + 2) * num_stream")
        self.phi = phi
        self.branch_alpha = _parameter(module, "branch_alpha")
        self.branch_beta = _parameter(module, "branch_beta")
        if self.branch_alpha.numel() != 3:
            raise ValueError("MHC pre branch_alpha must contain three values")
        expected_beta_size = 2 * self.num_stream + self.num_stream * self.num_stream
        if self.branch_beta.numel() != expected_beta_size:
            raise ValueError("MHC pre branch_beta has an incompatible size")

        self.mhc_use_gamma = bool(_required_config_value(module, "mhc_use_gamma"))
        parameters = [self.branch_alpha, self.branch_beta]
        if self.mhc_use_gamma:
            self.norm_gamma = _parameter(module, "norm_gamma")
            if self.norm_gamma.numel() != phi.in_features:
                raise ValueError("MHC pre norm_gamma size must equal the phi input size")
            parameters.append(self.norm_gamma)
        elif hasattr(module, "norm_gamma"):
            raise ValueError("MHC pre source has norm_gamma while mhc_use_gamma is disabled")
        _validate_parameter_layout(phi, tuple(parameters))

        self.hc_eps = float(_required_attribute(module, "hc_eps"))
        self.norm_eps = float(_required_attribute(module, "norm_eps"))
        self.mhc_recur_norm = int(_required_config_value(module, "mhc_recur_norm"))
        if self.hc_eps <= 0 or self.norm_eps <= 0:
            raise ValueError("MHC eps values must be positive")
        if self.mhc_recur_norm <= 0:
            raise ValueError("mhc_recur_norm must be a positive integer")
        self.train(module.training)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Prepare hidden states and return the MHC mixing coefficients."""
        gamma = self.norm_gamma if self.mhc_use_gamma else None
        return mhc_pre(
            x,
            self.phi.weight,
            self.branch_alpha,
            self.branch_beta,
            self.num_stream,
            self.mhc_recur_norm,
            self.norm_eps,
            self.hc_eps,
            gamma,
        )


@module_replacement
class MhcPostModule(nn.Module):
    """Combine transformed and residual streams after an MHC-wrapped block."""

    def __init__(
        self,
        *,
        module: nn.Module,
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
    ) -> None:
        """Build the high-performance MHC post module from an existing module.

        Args:
            module: Source MHC post module.
            module_fqn: Fully qualified source-module name supplied by replacement.
            context: Replacement context supplied by Trainer.
        """
        super().__init__()
        del module_fqn, context
        self.config = getattr(module, "config", None)
        self.num_stream = _num_stream(module)
        self.train(module.training)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        h_post: torch.Tensor,
        h_res: torch.Tensor,
    ) -> torch.Tensor:
        """Mix the wrapped block output into the residual streams."""
        return mhc_post(
            x,
            residual,
            h_post,
            h_res,
            self.num_stream,
        )


@module_replacement
class PipelinedMhcModule(nn.Module):
    """Coefficient module for cross-sublayer pipelined mHC."""

    def __init__(
        self,
        *,
        module: nn.Module,
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
    ) -> None:
        """Reuse the source module's ``fn/base/scale`` parameter layout."""
        super().__init__()
        del module_fqn, context
        required = ("input_norm", "fn", "base", "scale", "hc_mult")
        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            raise TypeError(f"pipelined MHC source is missing required attributes: {missing}")
        self.input_norm = module.input_norm
        self.fn = module.fn
        self.base = module.base
        self.scale = module.scale
        self.hc_mult = int(module.hc_mult)
        self.hc_sinkhorn_iters = int(module.hc_sinkhorn_iters)
        self.hc_eps = float(module.hc_eps)
        expected_mix = (self.hc_mult + 2) * self.hc_mult
        if tuple(self.fn.shape)[0] != expected_mix or self.base.numel() != expected_mix:
            raise ValueError("pipelined MHC fn/base shapes do not match hc_mult")
        if self.scale.numel() != 3:
            raise ValueError("pipelined MHC scale must contain three values")
        self.train(module.training)

    def forward(
        self,
        hidden_streams: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Produce the coefficients consumed by the following sublayer."""
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
        if hidden_streams.device.type == "npu" and hasattr(torch.ops.custom, "npu_sinkhorn"):
            residual = sinkhorn(residual, self.hc_sinkhorn_iters, self.hc_eps)
        else:
            residual = sinkhorn_knopps(residual, self.hc_sinkhorn_iters, self.hc_eps)
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
