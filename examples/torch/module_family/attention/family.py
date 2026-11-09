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
"""Public attention family; importing this module installs its declaration."""

from __future__ import annotations

from typing import Any

from torch import nn

from examples.torch.module_family.config import DecoderConfig
from hyper_parallel.components.modules import (
    ModuleFamily, VariantRule, build_family, register_module_family,
)
from hyper_parallel.core.dtensor.placement_types import Replicate
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.models.replacement import module_replacement


def _sharding(*, module: nn.Module, module_fqn: str, member: Any, context: Any) -> ModuleShardingSpec:
    del module_fqn, member, context
    return ModuleShardingSpec(params={name: {"cp": Replicate()} for name, _ in module.named_parameters()},
                              in_src={}, in_dst={}, is_boundary=False)


# Public family entries retain constructor-style names for stable target paths.
@module_replacement
def CausalAttention(  # pylint: disable=invalid-name
    *, config: DecoderConfig | None = None, module: nn.Module | None = None,
    module_fqn: str = "", context: Any = None, mesh: Any = None,
) -> nn.Module:
    """Construct the attention family using complete config or a source member."""
    return build_family(CausalAttention, config=config, module=module, module_fqn=module_fqn,
                        context=context, mesh=mesh)


register_module_family(ModuleFamily(
    name="example_causal_attention", baseline=CausalAttention,
    variants={"local": "examples.torch.module_family.attention.attention.CausalAttentionModule",
              "cp": "examples.torch.module_family.attention.cp_attention.CPCausalAttention"},
    default_variant="local", parallel_axis="cp", variant_rules=(VariantRule({"parallel": True}, "cp"),),
    sharding=_sharding,
))
