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
"""Common Attention Residual module family and sequence-layout contract."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from hyper_parallel.components.functional.attention_residuals import block_attention_residuals
from hyper_parallel.components.modules.family import ModuleFamily
from hyper_parallel.components.modules.module_family import build_family, register_module_family
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.distributed.module_binding import ParallelModuleSpec
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.models.replacement import module_replacement


_BACKEND_ALIASES = {"eager": "eager", "ascendc": "ascendc", "torch": "eager", "cann": "ascendc"}


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    """Read an optional setting from a model config or source module."""
    return getattr(config, name, default) if config is not None else default


def _resolve_backend(value: str) -> str:
    """Normalize legacy backend names to the shared functional API."""
    if not isinstance(value, str) or value not in _BACKEND_ALIASES:
        raise ValueError("AttentionResidual backend must be eager or ascendc")
    return _BACKEND_ALIASES[value]


def _validate_dimensions(hidden_size: Any, eps: Any, initializer_range: Any) -> None:
    """Validate dimensions and initialization scalars before allocating parameters."""
    if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size < 1:
        raise ValueError("AttentionResidual hidden_size must be a positive integer")
    if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not isfinite(eps) or eps <= 0:
        raise ValueError("AttentionResidual eps must be finite and positive")
    if (isinstance(initializer_range, bool) or not isinstance(initializer_range, (int, float))
            or not isfinite(initializer_range) or initializer_range <= 0):
        raise ValueError("AttentionResidual initializer_range must be finite and positive")


class AttentionResidualModule(nn.Module):
    """Mix the current residual with the completed residual blocks for each token."""

    def __init__(
        self,
        *,
        config: Any = None,
        module: nn.Module | None = None,
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
    ) -> None:
        """Construct from model dimensions or adopt a compatible source module.

        Args:
            config: Config with ``hidden_size`` and ``rms_norm_eps`` (or ``eps``), and optional
                ``attention_residual_backend`` (``eager`` or ``ascendc``) and
                ``attention_residual_recompute`` settings. The shorter ``attn_res_backend`` and
                ``attn_res_recompute`` names are accepted as aliases.
            module: Source module exposing ``projection`` and ``norm_weight`` Parameters.
            module_fqn: Replacement path supplied by the module replacement executor.
            context: Logical construction context supplied by the family builder.
        """
        super().__init__()
        del module_fqn, context
        if (config is None) == (module is None):
            raise ValueError("provide exactly one of config or module")

        source_config = config if config is not None else getattr(module, "config", None)
        self.eps = _config_value(
            source_config,
            "rms_norm_eps",
            _config_value(source_config, "eps", _config_value(module, "eps")),
        )
        initializer_range = _config_value(
            source_config,
            "initializer_range",
            _config_value(module, "initializer_range", 0.02),
        )
        configured_backend = _config_value(
            source_config,
            "attention_residual_backend",
            _config_value(source_config, "attn_res_backend", _config_value(module, "backend", "eager")),
        )
        if configured_backend is None:
            configured_backend = _config_value(module, "backend", "eager")
        self.backend = _resolve_backend(configured_backend)
        self.recompute = _config_value(
            source_config,
            "attention_residual_recompute",
            _config_value(source_config, "attn_res_recompute", _config_value(module, "recompute", False)),
        )
        if not isinstance(self.recompute, bool):
            raise TypeError("AttentionResidual recompute must be bool")

        if module is not None:
            projection = getattr(module, "projection", None)
            norm_weight = getattr(module, "norm_weight", None)
            if not isinstance(projection, nn.Parameter) or not isinstance(norm_weight, nn.Parameter):
                raise TypeError("AttentionResidual source must expose projection and norm_weight Parameters")
            hidden_size = projection.shape[-1] if projection.ndim in (1, 2) else None
            _validate_dimensions(hidden_size, self.eps, initializer_range)
            if tuple(projection.shape) not in ((hidden_size,), (1, hidden_size)):
                raise ValueError("AttentionResidual projection must have shape [H] or [1, H]")
            if tuple(norm_weight.shape) != (hidden_size,):
                raise ValueError("AttentionResidual norm_weight must have shape [H]")
            self.projection = projection
            self.norm_weight = norm_weight
            self.initializer_range = float(initializer_range)
            self.train(module.training)
            return

        hidden_size = _config_value(config, "hidden_size")
        _validate_dimensions(hidden_size, self.eps, initializer_range)
        self.initializer_range = float(initializer_range)
        self.projection = nn.Parameter(torch.empty(1, hidden_size, dtype=torch.float32))
        self.norm_weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Initialize the pseudo-query to zero and RMS scale to one."""
        if not getattr(self.projection, "_is_hf_initialized", False):
            nn.init.zeros_(self.projection)
        if not getattr(self.norm_weight, "_is_hf_initialized", False):
            nn.init.ones_(self.norm_weight)

    def make_transforms(self) -> list:
        """Keep the projection parameter names and shapes unchanged."""
        return []

    def forward(self, prefix: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """Mix ``[B,S,H]`` prefix states with ``[B,S,N,H]`` completed blocks."""
        if self.recompute and self.training and torch.is_grad_enabled():
            return checkpoint(self._mix, prefix, history, use_reentrant=False)
        return self._mix(prefix, history)

    def _mix(self, prefix: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """Flatten token rows for the shared functional implementation, then restore shape."""
        shape = prefix.shape
        result = block_attention_residuals(
            prefix.reshape(-1, shape[-1]),
            history.reshape(shape[0] * shape[1], history.shape[2], shape[-1]),
            self.projection,
            self.norm_weight,
            norm_eps=self.eps,
            backend=self.backend,
        )
        return result.reshape(shape)


def attention_residual_sharding(
    *, module: nn.Module, module_fqn: str, member: Any, context: Any,
) -> ModuleShardingSpec:
    """Declare replicated parameters and the CP/TP-sequence token layout."""
    del module, module_fqn, member
    replicated = {
        axis: Replicate()
        for axis in context.mesh_dim_names
        if axis in context.axis_sizes
    }
    token_layout = dict(replicated)
    if context.axis_sizes.get("cp", 1) > 1:
        token_layout["cp"] = Shard(1)
    if context.axis_sizes.get("tp", 1) > 1 and context.sequence_parallel:
        token_layout["tp"] = Shard(1)
    return ModuleShardingSpec(
        params={name: dict(replicated) for name in ("projection", "norm_weight")},
        in_src={"prefix": dict(token_layout), "history": dict(token_layout)},
        in_dst={"prefix": dict(token_layout), "history": dict(token_layout)},
        out_src={"output": dict(token_layout)},
        out_dst={"output": dict(token_layout)},
        out_names=("output",),
        is_boundary=True,
    )


def _same_placements(actual: Mapping[str, Any] | None, expected: Mapping[str, Any]) -> bool:
    """Treat omitted mesh axes as replicated, matching planner semantics."""
    actual = actual or {}
    return all(
        actual.get(axis, Replicate()) == expected.get(axis, Replicate())
        for axis in set(actual) | set(expected)
    )


def validate_attention_residual_layout(*, module: nn.Module, sharding: Any, context: Any) -> None:
    """Reject a plan that violates the Attention Residual parameter or token contract."""
    expected = attention_residual_sharding(module=module, module_fqn="", member=None, context=context)
    if not sharding.is_boundary or set(sharding.params or {}) != set(expected.params):
        raise ValueError("AttentionResidual must own both projection parameters at a module boundary")
    for name, placement in expected.params.items():
        if not _same_placements(sharding.params[name], placement):
            raise ValueError(f"unsupported AttentionResidual parameter layout for {name!r}")
    for field in ("in_src", "in_dst", "out_src", "out_dst"):
        actual = getattr(sharding, field) or {}
        required = getattr(expected, field) or {}
        if set(actual) != set(required) or any(
            not _same_placements(actual[name], placement) for name, placement in required.items()
        ):
            raise ValueError(f"unsupported AttentionResidual {field} layout")
    if tuple(sharding.out_names or ()) != ("output",):
        raise ValueError("AttentionResidual output contract must contain only 'output'")


def attention_residual_parallel_spec(*, module: nn.Module, member: Any) -> ParallelModuleSpec:
    """Expose the layout validator without binding communication resources."""
    del module, member
    return ParallelModuleSpec(
        "common.attention_residual.local",
        validate=validate_attention_residual_layout,
    )


@module_replacement
def AttentionResidual(  # pylint: disable=invalid-name
    *,
    config: Any = None,
    module: nn.Module | None = None,
    module_fqn: str = "",
    context: Mapping[str, Any] | None = None,
    mesh: Any = None,
) -> nn.Module:
    """Construct Attention Residual from config or a source module.

    Backend and recompute choices are configured through the model config rather than
    constructor overrides.
    """
    return build_family(
        AttentionResidual,
        config=config,
        module=module,
        module_fqn=module_fqn,
        context=context,
        mesh=mesh,
    )


register_module_family(ModuleFamily(
    name="attention_residual",
    baseline=AttentionResidual,
    variants={"local": "hyper_parallel.components.modules.common.attention_residual.AttentionResidualModule"},
    default_variant="local",
    sharding=attention_residual_sharding,
    parallel_spec=attention_residual_parallel_spec,
))


__all__ = ["AttentionResidual", "AttentionResidualModule"]
