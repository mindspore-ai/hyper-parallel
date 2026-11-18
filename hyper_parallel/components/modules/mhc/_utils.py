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
"""Validation shared by mHC implementations."""
from __future__ import annotations

from math import isfinite
from typing import Any

from torch import nn


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
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
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


def _validate_settings(num_stream: int, hidden_size: int | None = None, sinkhorn_iters: int = 20,
                       hc_eps: float = 1e-6, norm_eps: float = 1e-6, use_gamma: bool = False) -> None:
    """Validate logical settings independently of the caller's config class."""
    sizes = {"num_stream": num_stream}
    if hidden_size is not None:
        sizes.update(hidden_size=hidden_size, sinkhorn_iters=sinkhorn_iters)
    for name, value in sizes.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (("hc_eps", hc_eps), ("norm_eps", norm_eps)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not isinstance(use_gamma, bool):
        raise TypeError("use_gamma must be bool")
