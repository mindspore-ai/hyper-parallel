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
"""Dense MLP replacement entry points and family declarations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from hyper_parallel.components.modules.module_family import (
    ModuleFamily,
    SelectorSpec,
    VariantRule,
    build_family,
    register_module_family,
)
from hyper_parallel.models.replacement import module_replacement


@module_replacement
def SwiGLUMLP(  # pylint: disable=invalid-name
    *, config: Any = None, module: nn.Module | None = None,
    module_fqn: str = "", context: Mapping[str, Any] | None = None, mesh: Any = None,
    activation_dtype: str | None = None, swiglu_limit: float | None = None,
    state_layout: str = "linear_fc",
) -> nn.Module:
    """Construct a tracked MLP from complete config or separate source projections."""
    return build_family(SwiGLUMLP, config=config, module=module, module_fqn=module_fqn,
                        context=context, mesh=mesh, activation_dtype=activation_dtype,
                        swiglu_limit=swiglu_limit, state_layout=state_layout)


register_module_family(ModuleFamily(
    name="swiglu_mlp",
    baseline=SwiGLUMLP,
    variants={
        "local": "hyper_parallel.components.modules.mlp.swiglu_mlp.SwiGLUMLPModule",
        "w1_w2": "hyper_parallel.components.modules.mlp.swiglu_mlp.W1W2SwiGLUMLPModule",
    },
    selector_specs={"state_layout": SelectorSpec(str, "linear_fc")},
    variant_rules=(VariantRule({"state_layout": "w1_w2", "parallel": False}, "w1_w2"),),
    default_variant="local",
))

__all__ = ["SwiGLUMLP"]
