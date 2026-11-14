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
"""Gated GQA family with a mandatory per-query-head sigmoid gate."""

from __future__ import annotations

from typing import Any

from torch import nn

from hyper_parallel.components.modules.module_family import (
    ModuleFamily, VariantRule, build_family, register_module_family,
)
from hyper_parallel.models.replacement import module_replacement


from ..grouped_query_attention.grouped_query_attention_family import gqa_sharding, validate_gqa_family


@module_replacement
def GatedGroupedQueryAttention(  # pylint: disable=invalid-name
    *, config: Any = None, module: nn.Module | None = None, module_fqn: str = "",
    context: Any = None, mesh: Any = None, overlap: bool = False, ulysses_degree: int | None = None,
) -> nn.Module:
    """Build mandatory gated GQA; CP topology selects the class, overlap only selects scheduling.

    Args:
        config: Reserved for config-based construction; currently a source module is required.
        module: Source attention with the supported projection and checkpoint layout.
        module_fqn: Source module path supplied by replacement.
        context: Logical parallel configuration, independent of concrete process groups.
        mesh: Existing job mesh, borrowed by the shared execution binder.
        overlap: Submit ready Q/K/V transfers before independent projection stages.
        ulysses_degree: Head-parallel CP factor; 1 selects KV AG, None selects pure Ulysses.

    Returns:
        The local or CP member with fixed parameter names and family provenance.
    """
    return build_family(GatedGroupedQueryAttention, config=config, module=module, module_fqn=module_fqn,
                        context=context, mesh=mesh, overlap=overlap, ulysses_degree=ulysses_degree)


register_module_family(ModuleFamily(
    name="gated_grouped_query_attention", baseline=GatedGroupedQueryAttention,
    variants={"local": (
        "hyper_parallel.components.modules.attention.gated_grouped_query_attention."
        "gated_grouped_query_attention.GatedGroupedQueryAttentionModule"
    ),
    "cp": (
        "hyper_parallel.components.modules.attention.gated_grouped_query_attention."
        "cp_gated_grouped_query_attention.CPGatedGroupedQueryAttentionModule"
    )},
    default_variant="local", parallel_axis="cp",
    variant_rules=(VariantRule({"parallel": True}, "cp"),),
    sharding=gqa_sharding, validate=validate_gqa_family,
))
