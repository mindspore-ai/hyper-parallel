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
"""Small complete CPU modules used by construction and distribution acceptance tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch
from torch import nn

from hyper_parallel.components.modules.module_family import (
    ModuleFamily, SelectorSpec, VariantRule, build_family, get_module_family, register_module_family,
)
from hyper_parallel.models.replacement import ModuleReplacementRequest, module_replacement


@dataclass(frozen=True)
class LinearConfig:
    """Global dimensions and dtype, independent of any source module."""

    width: int = 4
    output_size: int = 3
    dtype: torch.dtype = torch.float32


class LocalLinear(nn.Module):
    """Complete ordinary algorithm and parameter registration for a test family."""

    def __init__(
        self, *, config: LinearConfig | None = None, module: nn.Linear | None = None,
        module_fqn: str = "", context: Mapping[str, Any] | None = None,
    ) -> None:
        """Create logical weights or retain a source module's parameters."""
        super().__init__()
        del context
        if module is not None:
            self.weight = module.weight
            self.bias = module.bias
            self.train(module.training)
        else:
            self.weight = nn.Parameter(torch.empty(config.output_size, config.width, dtype=config.dtype))
            self.bias = nn.Parameter(torch.empty(config.output_size, dtype=config.dtype))
            self.reset_parameters()
        self.module_fqn = module_fqn

    def reset_parameters(self) -> None:
        """Deterministic initialization works on real and meta tensors."""
        with torch.no_grad():
            self.weight.fill_(0.125)
            self.bias.fill_(0.25)

    # Preserve the nn.Linear input keyword across local and parallel replacements.
    def forward(self, input: torch.Tensor) -> torch.Tensor:  # pylint: disable=redefined-builtin
        """Evaluate the complete mathematical operation.

        Args:
            input: Input activation tensor.
        """
        return input @ self.weight.t() + self.bias


class ParallelLinear(LocalLinear):
    """Separate concrete class for context-dependent construction tests."""


class MegaLinear(LocalLinear):
    """Separate concrete class for allowlisted fused selection tests."""


class StatefulLocalLinear(LocalLinear):
    """Config-only member with persistent buffers and additional checkpoint state."""

    def __init__(self, *, config: LinearConfig) -> None:
        """Create the same logical weights plus local algorithm state."""
        super().__init__(config=config)
        self.register_buffer("output_scale", torch.ones(config.output_size, dtype=config.dtype))
        self.samples_seen = 0

    def get_extra_state(self) -> dict[str, int]:
        """Export algorithm state that is independent of tensor layout."""
        return {"samples_seen": self.samples_seen}

    def set_extra_state(self, state: dict[str, int]) -> None:
        """Restore the counter used by the stateful algorithm.

        Args:
            state: Saved algorithm counters.
        """
        self.samples_seen = state["samples_seen"]

    # Preserve the nn.Linear input keyword across local and parallel replacements.
    def forward(self, input: torch.Tensor) -> torch.Tensor:  # pylint: disable=redefined-builtin
        """Apply the stored output calibration to ordinary linear computation.

        Args:
            input: Input activations.
        """
        return super().forward(input) * self.output_scale


class StatefulParallelLinear(StatefulLocalLinear):
    """Parallel selection fixture with the same config-only state contract."""


class LocalLinearBlock(nn.Module):
    """A family member containing another family and a shared module alias."""

    def __init__(self, *, config: LinearConfig, context: Mapping[str, Any] | None = None) -> None:
        """Build a nested family through its public construction interface."""
        super().__init__()
        self.projection = LinearFamily(config=config, context=context)
        self.alias = self.projection

    # Keep the input port shared with this fixture family's Linear layout provider.
    def forward(self, input: torch.Tensor) -> torch.Tensor:  # pylint: disable=redefined-builtin
        """Use the nested family through an ordinary child call.

        Args:
            input: Input activations.
        """
        return self.projection(input)


class ParallelLinearBlock(LocalLinearBlock):
    """Parallel parent member retaining independent nested family ownership."""


# Public family entries retain constructor-style names for stable target paths.
def LinearBlockFamily(  # pylint: disable=invalid-name
    *, config: LinearConfig, context: Mapping[str, Any] | None = None,
) -> nn.Module:
    """Construct the nested test family without source replacement."""
    return build_family(LinearBlockFamily, config=config, context=context)


# Public family entries retain constructor-style names for stable target paths.
@module_replacement
def LinearFamily(  # pylint: disable=invalid-name
    *, config: LinearConfig | None = None, module: nn.Module | None = None,
    module_fqn: str = "", context: Mapping[str, Any] | None = None,
    kernel: str = "default", precision: str = "inherit",
) -> nn.Module:
    """One public callable shared by direct and source construction."""
    return build_family(
        LinearFamily, config=config, module=module, module_fqn=module_fqn, context=context,
        kernel=kernel, precision=precision,
    )


def register_linear_family(*, sharding: Callable | None = None, axis: str = "ep") -> None:
    """Register a fresh test declaration when no fixture has registered it yet."""
    if get_module_family(LinearFamily) is not None:
        return
    prefix = __name__
    rules = (
        VariantRule({"kernel": "default", "precision": "inherit", "parallel": True}, "parallel"),
        VariantRule({"kernel": "megakernel", "precision": "inherit", "parallel": False}, "mega"),
        VariantRule({"kernel": "megakernel", "precision": "inherit", "parallel": True}, "mega"),
    )
    register_module_family(ModuleFamily(
        name="test_linear_family", baseline=LinearFamily,
        variants={"default": f"{prefix}.LocalLinear", "parallel": f"{prefix}.ParallelLinear",
                  "mega": f"{prefix}.MegaLinear"},
        selector_specs={"kernel": SelectorSpec(str, "default"), "precision": SelectorSpec(str, "inherit")},
        variant_rules=rules, default_variant="default", parallel_axis=axis, sharding=sharding,
    ))


@dataclass(frozen=True)
class FamilyModelConfig:
    """Complete test-model facts that do not require an HF base class."""

    model_type: str = "test_family_model"
    architectures: tuple[str, ...] = ("FamilyModel",)
    width: int = 4
    output_size: int = 3


class FamilyModel(nn.Module):
    """Model composed directly from a family without any source tree."""

    def __init__(self, config: FamilyModelConfig, *, context: Mapping[str, Any] | None = None) -> None:
        """Propagate the same explicit context to the family boundary."""
        super().__init__()
        self.config = config
        self.projection = LinearFamily(
            config=LinearConfig(config.width, config.output_size), context=context,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute model output through normal child-module calls.

        Args:
            inputs: Input activation tensor.
        """
        return self.projection(inputs)


def create_family_model(config: FamilyModelConfig, *, context: Mapping[str, Any]) -> FamilyModel:
    """Stable factory used by real facade and generated-constructor tests.

    Args:
        config: Complete logical model configuration.
    """
    return FamilyModel(config, context=context)


def get_classifier_replacements(*, model: nn.Module, context: Mapping[str, Any]) -> tuple:
    """Stable default source declaration for a small HF model acceptance test."""
    del model, context
    return (ModuleReplacementRequest(("classifier",), LinearFamily),)
