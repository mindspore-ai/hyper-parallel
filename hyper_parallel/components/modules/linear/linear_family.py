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
"""Dense Linear Module Family registration and shared low-precision factories."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from hyper_parallel.components.modules.linear.linear import Linear, LowPrecisionLinear
from hyper_parallel.components.modules.module_family import (
    ModuleFamily,
    SelectorSpec,
    VariantRule,
    register_module_family,
)
from hyper_parallel.components.quantization.config import LowPrecisionConfig, LowPrecisionDtypeScheme
from hyper_parallel.components.quantization.functional.linear_strategy_factory import build_linear_strategy
from hyper_parallel.models.replacement import module_replacement

_MXFP8_DENSE_SCHEME = LowPrecisionDtypeScheme(weight_format="mxfp8", act_format="mxfp8")
_HIF8_DENSE_SCHEME = LowPrecisionDtypeScheme(weight_format="hif8", act_format="hif8")


def replace_linear_with_policy(
    *,
    module: nn.Module,
    module_fqn: str,
    context: Mapping[str, Any],
    scheme: LowPrecisionDtypeScheme | LowPrecisionConfig | None,
) -> LowPrecisionLinear:
    """Validate an exact Dense Linear boundary and bind its resolved compute strategy.

    Args:
        module: Source Linear whose loaded parameters are retained.
        module_fqn: Module path for diagnostics.
        context: Replacement context containing parallel topology.
        scheme: Explicit or policy-resolved low-precision dtype scheme.

    Returns:
        A ``LowPrecisionLinear`` sharing the source parameters without reinitializing them.
    """
    if type(module) is not nn.Linear:  # pylint: disable=unidiomatic-typecheck
        raise TypeError(f"{module_fqn!r} must be exact nn.Linear, got {type(module).__name__}")
    if context.get("pp"):
        raise NotImplementedError(
            "Low-precision Linear training is not yet supported with pipeline parallelism."
        )
    try:
        strategy = build_linear_strategy(
            scheme,
            in_features=module.in_features,
            out_features=module.out_features,
        )
    except ValueError as error:
        raise ValueError(
            f"Low-precision Linear target {module_fqn!r} is invalid: {error}"
        ) from error
    return LowPrecisionLinear.from_linear(module, strategy=strategy)


@module_replacement
def replace_mxfp8_linear(
    *, module: nn.Module, module_fqn: str, context: Mapping[str, Any],
) -> LowPrecisionLinear:
    """Bind the MXFP8 Dense strategy selected by the Linear Family rule.

    Args:
        module: Exact nn.Linear source with feature dimensions divisible by 32.
        module_fqn: Module path for diagnostics.
        context: Replacement context containing parallel topology.

    Returns:
        A low-precision Linear sharing the source parameters and training mode.
    """
    return replace_linear_with_policy(
        module=module,
        module_fqn=module_fqn,
        context=context,
        scheme=_MXFP8_DENSE_SCHEME,
    )


@module_replacement
def replace_hif8_linear(
    *, module: nn.Module, module_fqn: str, context: Mapping[str, Any],
) -> LowPrecisionLinear:
    """Bind the HiFloat8 Dense strategy selected by the Linear Family rule.

    Args:
        module: Exact nn.Linear source; MX block alignment is not required.
        module_fqn: Module path for diagnostics.
        context: Replacement context containing parallel topology.

    Returns:
        A low-precision Linear sharing the source parameters and training mode.
    """
    return replace_linear_with_policy(
        module=module,
        module_fqn=module_fqn,
        context=context,
        scheme=_HIF8_DENSE_SCHEME,
    )


register_module_family(ModuleFamily(
    name="linear",
    baseline=Linear,
    variants={
        "mxfp8": "hyper_parallel.components.modules.linear.linear_family.replace_mxfp8_linear",
        "hif8": "hyper_parallel.components.modules.linear.linear_family.replace_hif8_linear",
    },
    selector_specs={
        "weight_format": SelectorSpec(str, ""),
        "act_format": SelectorSpec(str, ""),
        "is_fake_quantize": SelectorSpec(bool, False),
        "fused": SelectorSpec(bool, False),
    },
    variant_rules=(
        VariantRule(
            {"weight_format": "mxfp8", "act_format": "mxfp8", "is_fake_quantize": False, "fused": False},
            "mxfp8",
        ),
        VariantRule(
            {"weight_format": "hif8", "act_format": "hif8", "is_fake_quantize": False, "fused": False},
            "hif8",
        ),
    ),
))

__all__ = [
    "replace_hif8_linear",
    "replace_linear_with_policy",
    "replace_mxfp8_linear",
]
