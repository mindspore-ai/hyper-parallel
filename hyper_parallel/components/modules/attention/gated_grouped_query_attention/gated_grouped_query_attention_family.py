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


from ..grouped_query_attention.grouped_query_attention_family import (
    gqa_local_parallel_spec, gqa_sharding, validate_gqa_family,
)
from .gated_grouped_query_attention import GatedGroupedQueryAttentionModule


def _build_local(
    *, config: Any = None, module: nn.Module | None = None, module_fqn: str = "", context: Any = None,
    ulysses_cp_size: int = 1, ulysses_cp_strategy: str = "alltoall",
    kv_cp_size: int = 1, kv_cp_strategy: str = "allgather", kv_cp_load_balance: bool | str = False,
    ring_cp_size: int = 1, ring_cp_strategy: str = "p2p", ring_cp_load_balance: bool | str = False,
    cp_comm_overlap: bool = False,
) -> nn.Module:
    """Build gated math; shared layout binding handles TP/SP."""
    del (module_fqn, context, ulysses_cp_size, ulysses_cp_strategy, kv_cp_size,
         kv_cp_strategy, kv_cp_load_balance, ring_cp_size, ring_cp_strategy, ring_cp_load_balance,
         cp_comm_overlap)
    return GatedGroupedQueryAttentionModule(config=config, module=module)


def _build_cp(
    *, config: Any = None, module: nn.Module | None = None, module_fqn: str = "", context: Any = None,
    ulysses_cp_size: int = 1, ulysses_cp_strategy: str = "alltoall",
    kv_cp_size: int = 1, kv_cp_strategy: str = "allgather", kv_cp_load_balance: bool | str = False,
    ring_cp_size: int = 1, ring_cp_strategy: str = "p2p", ring_cp_load_balance: bool | str = False,
    cp_comm_overlap: bool = False,
) -> nn.Module:
    """Adapt the public domain configuration to the existing gated-GQA executor."""
    del module_fqn
    from .cp_gated_grouped_query_attention import CPGatedGroupedQueryAttentionModule  # pylint: disable=C0415

    return CPGatedGroupedQueryAttentionModule(
        config=config, module=module, context=context, cp_comm_overlap=cp_comm_overlap,
        ulysses_cp_size=ulysses_cp_size,
    )


@module_replacement
def GatedGroupedQueryAttention(  # pylint: disable=invalid-name
    *, config: Any = None, module: nn.Module | None = None, module_fqn: str = "",
    context: Any = None, mesh: Any = None,
    ulysses_cp_size: int = 1, ulysses_cp_strategy: str = "alltoall",
    kv_cp_size: int = 1, kv_cp_strategy: str = "allgather", kv_cp_load_balance: bool | str = False,
    ring_cp_size: int = 1, ring_cp_strategy: str = "p2p", ring_cp_load_balance: bool | str = False,
    cp_comm_overlap: bool = False,
) -> nn.Module:
    """Build mandatory gated GQA; CP topology selects the class, overlap only selects scheduling.

    Args:
        config: Native dimensions/options; mutually exclusive with module. Q/K RMSNorm is opt-in via qk_norm.
        module: Source attention with the supported projection and checkpoint layout.
        module_fqn: Source module path supplied by replacement.
        context: Logical parallel configuration, independent of concrete process groups.
        mesh: Existing job mesh, borrowed by the shared execution binder.
        ulysses_cp_size: Number of ranks in the Ulysses domain.
        ulysses_cp_strategy: Gated GQA currently supports ``alltoall``.
        kv_cp_size: Number of ranks in the KV AllGather domain.
        kv_cp_strategy: Gated GQA currently supports ``allgather``.
        kv_cp_load_balance: Optional KV-domain zigzag balance; currently unsupported.
        ring_cp_size: Number of ranks in the Ring domain; currently must be one.
        ring_cp_strategy: Ring transport selector, reserved for future support.
        ring_cp_load_balance: Optional Ring-domain zigzag balance; currently unsupported.
        cp_comm_overlap: Submit ready Q/K/V transfers before independent projection stages.

    Returns:
        The local or CP member with fixed parameter names and family provenance.
    """
    return build_family(GatedGroupedQueryAttention, config=config, module=module, module_fqn=module_fqn,
                        context=context, mesh=mesh, ulysses_cp_size=ulysses_cp_size,
                        ulysses_cp_strategy=ulysses_cp_strategy, kv_cp_size=kv_cp_size,
                        kv_cp_strategy=kv_cp_strategy, kv_cp_load_balance=kv_cp_load_balance,
                        ring_cp_size=ring_cp_size, ring_cp_strategy=ring_cp_strategy,
                        ring_cp_load_balance=ring_cp_load_balance, cp_comm_overlap=cp_comm_overlap)


register_module_family(ModuleFamily(
    name="gated_grouped_query_attention", baseline=GatedGroupedQueryAttention,
    variants={"local": (
        "hyper_parallel.components.modules.attention.gated_grouped_query_attention."
        "gated_grouped_query_attention_family._build_local"
    ),
    "cp": "hyper_parallel.components.modules.attention.gated_grouped_query_attention."
          "gated_grouped_query_attention_family._build_cp"},
    default_variant="local", parallel_axis="cp",
    variant_rules=(VariantRule({"parallel": True}, "cp"),),
    sharding=gqa_sharding, validate=validate_gqa_family, parallel_spec=gqa_local_parallel_spec,
))
