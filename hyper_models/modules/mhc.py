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

from collections.abc import Mapping
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_models.components.model_transform import module_replacement
from hyper_models.ops import mhc_post, mhc_post_process, mhc_pre


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
        self.mhc_hpre_renorm = bool(_required_config_value(module, "mhc_hpre_renorm"))
        self.use_mhc_ascendc_pre = bool(_required_config_value(module, "use_mhc_ascendc_pre"))
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
            self.mhc_hpre_renorm,
            self.use_mhc_ascendc_pre,
        )


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
        self.use_mhc_ascendc_post = bool(_required_config_value(module, "use_mhc_ascendc_post"))
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
            self.use_mhc_ascendc_post,
        )


class MhcPostProcessModule(nn.Module):
    """Merge all residual streams at the end of an MHC stack."""

    def __init__(
        self,
        *,
        module: nn.Module,
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
    ) -> None:
        """Build the high-performance final MHC module from an existing module.

        Args:
            module: Source MHC post-process module with the existing parameter layout.
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
            raise TypeError("MhcPostProcessModule requires a bias-free nn.Linear phi projection")
        if phi.in_features % self.num_stream != 0:
            raise ValueError("MHC post-process phi input size must be divisible by num_stream")
        if phi.out_features != self.num_stream:
            raise ValueError("MHC post-process phi output size must equal num_stream")
        self.phi = phi
        self.branch_alpha = _parameter(module, "branch_alpha")
        self.branch_beta = _parameter(module, "branch_beta")
        if self.branch_alpha.numel() != 1:
            raise ValueError("MHC post-process branch_alpha must contain one value")
        if self.branch_beta.numel() != self.num_stream:
            raise ValueError("MHC post-process branch_beta size must equal num_stream")

        self.mhc_use_gamma = bool(_required_config_value(module, "mhc_use_gamma"))
        parameters = [self.branch_alpha, self.branch_beta]
        if self.mhc_use_gamma:
            self.norm_gamma = _parameter(module, "norm_gamma")
            if self.norm_gamma.numel() != phi.in_features:
                raise ValueError("MHC post-process norm_gamma size must equal the phi input size")
            parameters.append(self.norm_gamma)
        elif hasattr(module, "norm_gamma"):
            raise ValueError("MHC post-process source has norm_gamma while mhc_use_gamma is disabled")
        _validate_parameter_layout(phi, tuple(parameters))

        self.hc_eps = float(_required_attribute(module, "hc_eps"))
        self.norm_eps = float(_required_attribute(module, "norm_eps"))
        self.mhc_hpre_renorm = bool(_required_config_value(module, "mhc_hpre_renorm"))
        if self.hc_eps <= 0 or self.norm_eps <= 0:
            raise ValueError("MHC eps values must be positive")
        self.train(module.training)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Merge all residual streams into one hidden state."""
        gamma = self.norm_gamma if self.mhc_use_gamma else None
        return mhc_post_process(
            x,
            self.phi.weight,
            self.branch_alpha,
            self.branch_beta,
            self.num_stream,
            self.norm_eps,
            self.hc_eps,
            gamma,
            self.mhc_hpre_renorm,
        )
