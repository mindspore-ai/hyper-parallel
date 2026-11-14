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
"""Gated NoPE MLA family with a mandatory output gate."""

from __future__ import annotations

from typing import Any

from torch import nn

from hyper_parallel.components.modules.module_family import (
    ModuleFamily, VariantRule, build_family, register_module_family,
)
from hyper_parallel.components.modules.attention.multi_latent_attention.multi_latent_attention_family import (
    mla_sharding, validate_mla_family,
)
from hyper_parallel.models.replacement import module_replacement


def validate_gated_mla_family(*, config: Any, module: nn.Module | None, context: Any, member: Any) -> None:
    """Keep gated MLA's TP restriction independent of the model API adapter's family name."""
    if member.axis_sizes["tp"] != 1:
        raise ValueError("Gated MLA currently supports TP1 only")
    validate_mla_family(config=config, module=module, context=context, member=member)


@module_replacement
def GatedMultiLatentAttention(  # pylint: disable=invalid-name
    *, config: Any = None, module: nn.Module | None = None,
    module_fqn: str = "", context: Any = None, mesh: Any = None,
    strategy: str = "expanded_ulysses", overlap: bool = False,
    ulysses_degree: int | None = None, kv_exchange: str = "all_gather",
) -> nn.Module:
    """Build the local gated NoPE MLA member.

    Args:
        config: Full model dimensions for direct construction, with a mandatory output gate.
        module: Optional supported NoPE source; exclusive with config.
        module_fqn: Source path supplied by model replacement.
        context: Logical build constraints. Topology selects the local or CP implementation.
        mesh: Optional existing job mesh used by the shared layout planner.
        strategy: Expanded QKV Ulysses or shared latent-KV communication.
        overlap: Enable existing forward and supported backward scheduling.
        ulysses_degree: CP head-parallel factor; the remaining factor partitions queries.
        kv_exchange: Full AllGather or selective causal-document AllToAllV.

    Returns:
        Gated MLA with its own NoPE math, gate Parameters and family provenance.
    """
    return build_family(GatedMultiLatentAttention, config=config, module=module,
                        module_fqn=module_fqn, context=context, mesh=mesh,
                        strategy=strategy, overlap=overlap, ulysses_degree=ulysses_degree, kv_exchange=kv_exchange)


register_module_family(ModuleFamily(
    name="gated_multi_latent_attention", baseline=GatedMultiLatentAttention,
    variants={"local": (
        "hyper_parallel.components.modules.attention.gated_multi_latent_attention."
        "gated_multi_latent_attention.GatedMultiLatentAttentionModule"
    ), "cp": (
        "hyper_parallel.components.modules.attention.gated_multi_latent_attention."
        "cp_gated_multi_latent_attention.CPGatedMultiLatentAttentionModule"
    )},
    default_variant="local", parallel_axis="cp",
    variant_rules=(VariantRule({"parallel": True}, "cp"),),
    sharding=mla_sharding, validate=validate_gated_mla_family,
))
