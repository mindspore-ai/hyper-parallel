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
"""Linear Family retaining loaded parameters and selecting existing Dense adapters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.modules.family import ModuleFamily, SelectorSpec, VariantRule, register_module_family
from hyper_parallel.models.replacement import module_replacement


@module_replacement
class Linear(nn.Linear):
    """Baseline Dense replacement using ordinary torch linear computation."""

    def __init__(self, *, module: nn.Linear, module_fqn: str = "",
                 context: Mapping[str, Any] | None = None) -> None:
        """Reuse parameters without allocating or initializing a second weight set.

        Args:
            module: Source Linear with loaded weight and optional bias.
            module_fqn: Replacement path; unused by baseline computation.
            context: Replacement context; unused by baseline computation.
        """
        del module_fqn, context
        # Like LowPrecisionLinear.from_linear, reuse parameters without nn.Linear allocating temporary weights.
        nn.Module.__init__(self)  # pylint: disable=non-parent-init-called
        self.in_features, self.out_features = module.in_features, module.out_features
        self.register_parameter("weight", module.weight)
        self.register_parameter("bias", module.bias)
        self.train(module.training)


register_module_family(ModuleFamily(
    name="linear",
    baseline=Linear,
    variants={"mxfp8": "hyper_parallel.components.modules.low_precision_linear.replace_mxfp8_linear"},
    selector_specs={"weight_format": SelectorSpec(str, ""), "act_format": SelectorSpec(str, ""),
                    "is_fake_quantize": SelectorSpec(bool, False), "fused": SelectorSpec(bool, False)},
    variant_rules=(VariantRule(
        {"weight_format": "mxfp8", "act_format": "mxfp8", "is_fake_quantize": False, "fused": False}, "mxfp8"),),
))
