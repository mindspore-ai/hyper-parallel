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
"""Parameterized mHC pre computation."""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any
import torch
from torch import nn
from hyper_parallel.components.ops import mhc_pre
from ._utils import (_validate_settings, _num_stream, _required_attribute, _required_config_value,
                         _parameter, _validate_parameter_layout)


class MhcPreModule(nn.Module):
    """Prepare hidden states and mixing coefficients for an MHC-wrapped block."""

    def __init__(
        self,
        *,
        module: nn.Module | None = None,
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
        config: Any = None,
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
        if (module is None) == (config is None):
            raise ValueError("provide exactly one of config or module")
        if config is not None:
            _validate_settings(config.num_stream, config.hidden_size, config.sinkhorn_iters,
                               config.hc_eps, config.norm_eps, config.use_gamma)
            self.config = config
            self.num_stream = config.num_stream
            width = config.hidden_size * config.num_stream
            count = (config.num_stream + 2) * config.num_stream
            self.phi = nn.Linear(width, count, bias=False, dtype=torch.float32)
            self.branch_alpha = nn.Parameter(torch.empty(3, dtype=torch.float32))
            self.branch_beta = nn.Parameter(torch.empty(count, dtype=torch.float32))
            self.mhc_use_gamma = config.use_gamma
            if config.use_gamma:
                self.norm_gamma = nn.Parameter(torch.empty(width, dtype=torch.float32))
            self.hc_eps, self.norm_eps = config.hc_eps, config.norm_eps
            self.mhc_recur_norm = config.sinkhorn_iters
            self.reset_parameters()
            return
        self.config = getattr(module, "config", None)
        self.num_stream = _num_stream(module)

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
        _validate_settings(self.num_stream, phi.in_features // self.num_stream,
                     self.mhc_recur_norm, self.hc_eps, self.norm_eps, self.mhc_use_gamma)
        self.train(module.training)

    def reset_parameters(self) -> None:
        """Initialize direct-build projection and coefficient scales explicitly."""
        nn.init.normal_(self.phi.weight, std=0.02)
        nn.init.constant_(self.branch_alpha, 0.01)
        nn.init.zeros_(self.branch_beta)
        if self.mhc_use_gamma:
            nn.init.ones_(self.norm_gamma)

    def make_transforms(self) -> list:
        """Declare that this member keeps parameter names and layouts unchanged.

        Returns:
            An empty converter list, matching members whose initialization is
            fully owned by ``reset_parameters``.
        """
        return []

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
