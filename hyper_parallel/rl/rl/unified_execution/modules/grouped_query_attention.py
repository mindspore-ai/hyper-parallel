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
"""Build HP grouped-query attention with a PIK output projection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from torch import nn

from hyper_parallel.components.modules.attention.grouped_query_attention.cp_grouped_query_attention import (
    CPGroupedQueryAttentionModule,
)
from hyper_parallel.components.modules.attention.grouped_query_attention.grouped_query_attention import (
    GroupedQueryAttentionModule,
)
from hyper_parallel.components.modules.attention.grouped_query_attention.grouped_query_attention_family import gqa_sharding
from hyper_parallel.distributed._builder.family_layout import validate_tp_boundary
from hyper_parallel.distributed.module_binding import NativeParallelBindingRequest, ParallelModuleSpec
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.models.replacement import module_replacement
from rl.unified_execution.models import get_model_family, resolve_runtime_symbol
from rl.unified_execution.modules.row_parallel_linear import build_pik_linear


@module_replacement
def build_grouped_query_attention(
    *, module: nn.Module, module_fqn: str, context: Mapping[str, Any],
    num_leaves: int = 8, leaf_layout: str = "strided", fill_uninitialized_memory: bool = False,
) -> GroupedQueryAttentionModule:
    """Build local or CP GQA with a PIK output projection and native HP sharding."""
    # Convert the source model's QKV weights before selecting the HP parallel module.
    family = get_model_family(module.config.model_type)
    gqa_factory = resolve_runtime_symbol(family.gqa_factory)
    local_gqa = gqa_factory(module=module, module_fqn=module_fqn, context={})
    if context.get("cp"):
        gqa = CPGroupedQueryAttentionModule(module=local_gqa, ulysses_degree=1)
        parallel_spec = gqa.parallel_module_spec()
    else:
        gqa = GroupedQueryAttentionModule(module=local_gqa)
        parallel_spec = ParallelModuleSpec("attention.gqa.local")

    gqa.o_proj = build_pik_linear(
        module=gqa.o_proj, module_fqn=module_fqn + ".o_proj", context=context,
        num_leaves=num_leaves, leaf_layout=leaf_layout, fill_uninitialized_memory=fill_uninitialized_memory,
    )
    # The model adapter collects these per-layer declarations before HP builds its sharding plan.
    gqa._rl_native_parallel_request = NativeParallelBindingRequest(  # pylint: disable=protected-access
        (module_fqn,), replace(parallel_spec, validate=_validate_pik_gqa_layout), _pik_gqa_sharding)
    return gqa


def _pik_gqa_sharding(*, module: nn.Module, module_fqn: str = "", member: Any = None,
                      context: Any) -> ModuleShardingSpec:
    """PIK already produces the final TP/SP output; all other ports are standard GQA."""
    layout = gqa_sharding(module=module, module_fqn=module_fqn, member=member, context=context)
    layout.out_src = layout.out_dst.copy()
    return layout


def _validate_pik_gqa_layout(*, module: nn.Module, sharding: ModuleShardingSpec, context: Any) -> None:
    validate_tp_boundary(sharding, _pik_gqa_sharding(module=module, context=context), name="PIK GQA")
    if sharding.out_src != sharding.out_dst:
        raise ValueError("PIK GQA must preserve its already-reduced output")
