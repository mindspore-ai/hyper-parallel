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
"""Module-family factory for Ring Context Parallel GQA."""

from __future__ import annotations

from typing import Any

from torch import nn

from hyper_parallel.components.modules.module_family import (
    ModuleFamily, VariantRule, build_family, register_module_family,
)
from hyper_parallel.models.replacement import module_replacement
from .ring_grouped_query_attention import RingGroupedQueryAttentionModule
from .grouped_query_attention_family import gqa_sharding, validate_gqa_family


def _build_ring(
    *,
    config: Any = None,
    module: nn.Module | None = None,
    module_fqn: str = "",
    context: Any = None,
    fuse_qkv: bool = True,
    overlap: bool = False,
    ulysses_degree: int = 1,
    load_balance: str = "none",
    balance_block_size: int | None = None,
    zigzag_min_length: int | None = None,
) -> nn.Module:
    """Construct the concrete Ring GQA member."""
    return RingGroupedQueryAttentionModule(
        config=config,
        module=module,
        module_fqn=module_fqn,
        context=context,
        fuse_qkv=fuse_qkv,
        overlap=overlap,
        ulysses_degree=ulysses_degree,
        load_balance=load_balance,
        balance_block_size=balance_block_size,
        zigzag_min_length=zigzag_min_length,
    )


@module_replacement
def RingGroupedQueryAttention(
    *,
    config: Any = None,
    module: nn.Module | None = None,
    module_fqn: str = "",
    context: Any = None,
    mesh: Any = None,
    fuse_qkv: bool = True,
    overlap: bool = False,
    ulysses_degree: int = 1,
    load_balance: str = "none",
    balance_block_size: int | None = None,
    zigzag_min_length: int | None = None,
) -> nn.Module:
    """Build GQA with the Ring attention execution member.

    The factory deliberately lives beside ordinary GQA instead of changing
    its default CP algorithm. Existing recipes keep their all-gather/Ulysses
    behavior; Ring is selected explicitly by replacing with this factory.
    """
    return build_family(
        RingGroupedQueryAttention,
        config=config,
        module=module,
        module_fqn=module_fqn,
        context=context,
        mesh=mesh,
        fuse_qkv=fuse_qkv,
        overlap=overlap,
        ulysses_degree=ulysses_degree,
        load_balance=load_balance,
        balance_block_size=balance_block_size,
        zigzag_min_length=zigzag_min_length,
    )


register_module_family(ModuleFamily(
    name="ring_grouped_query_attention",
    baseline=RingGroupedQueryAttention,
    variants={
        "ring": (
            "hyper_parallel.components.modules.attention.grouped_query_attention."
            "ring_grouped_query_attention_family._build_ring"
        ),
    },
    default_variant="ring",
    parallel_axis="cp",
    variant_rules=(VariantRule({"parallel": True}, "ring"),),
    sharding=gqa_sharding,
    validate=validate_gqa_family,
))
