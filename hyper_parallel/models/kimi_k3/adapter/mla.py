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
"""Kimi's Tensor-returning API over the shared local and CP gated MLA members."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
from torch import nn

from hyper_parallel.components.modules.module_family import (
    ModuleFamily, VariantRule, build_family, register_module_family,
)
from hyper_parallel.components.modules.attention.gated_multi_latent_attention import GatedMultiLatentAttentionModule
from hyper_parallel.components.modules.attention.gated_multi_latent_attention.cp_gated_multi_latent_attention import (
    CPGatedMultiLatentAttentionModule,
)
from hyper_parallel.components.modules.attention.gated_multi_latent_attention import (
    gated_multi_latent_attention_family as gated_family,
)
from hyper_parallel.components.modules.attention.multi_latent_attention.multi_latent_attention_family import (
    mla_sharding,
)
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.models.replacement import module_replacement


class _KimiMLAForward:
    """Translate the model API without changing NoPE, gating, or CP scheduling."""

    # Preserve Kimi's native (hidden, mask, position_ids, cache) API and Tensor return, not the HF base signature.
    # pylint: disable-next=arguments-differ
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None,
                position_ids: torch.Tensor | None = None, past_key_values: Any | None = None,
                **kwargs: Any) -> torch.Tensor:
        """Run gated MLA with the source tensor-only return contract.

        Args:
            hidden_states: Local input in BSH layout.
            attention_mask: Optional mask forwarded to the attention backend.
            position_ids: Unused by NoPE attention.
            past_key_values: Cache is unsupported by this training replacement.
            **kwargs: Packed lengths and attention options.

        Returns:
            Local hidden states after gated attention and output projection.
        """
        del position_ids
        if past_key_values is not None:
            raise ValueError("Kimi MLA training replacement does not support a KV cache")
        # Both concrete classes place a gated MLA member next in the method
        # resolution order; this mixin owns only the source model's API.
        output, _ = super().forward(  # pylint: disable=no-member
            hidden_states, attention_mask=attention_mask, **kwargs,
        )
        return output


class KimiMultiLatentAttentionModule(_KimiMLAForward, GatedMultiLatentAttentionModule):
    """Local gated MLA with Kimi's forward signature and Tensor output."""


class CPKimiMultiLatentAttentionModule(_KimiMLAForward, CPGatedMultiLatentAttentionModule):
    """CP gated MLA with the same Kimi API and class-owned execution stages."""


def _kimi_mla_sharding(*, module: nn.Module, module_fqn: str, member: Any, context: Any) -> ModuleShardingSpec:
    """Use the shared gated MLA layout with Kimi's single Tensor output port."""
    spec = mla_sharding(module=module, module_fqn=module_fqn, member=member, context=context)
    return replace(spec, out_names=("output",))


def _build_local(
    *, config: Any = None, module: nn.Module | None = None, module_fqn: str = "", context: Any = None,
    kv_exchange: str = "all_gather",
    ulysses_cp_size: int = 1, ulysses_cp_strategy: str = "alltoall",
    kv_cp_size: int = 1, kv_cp_strategy: str = "allgather", kv_cp_load_balance: bool | str = False,
    ring_cp_size: int = 1, ring_cp_strategy: str = "p2p", ring_cp_load_balance: bool | str = False,
    cp_comm_overlap: bool = False,
) -> nn.Module:
    """Retain CP options in the family request without passing them to local math."""
    del (module_fqn, context, kv_exchange, ulysses_cp_size,
         ulysses_cp_strategy, kv_cp_size, kv_cp_strategy, kv_cp_load_balance, ring_cp_size,
         ring_cp_strategy, ring_cp_load_balance, cp_comm_overlap)
    return KimiMultiLatentAttentionModule(config=config, module=module)


def _build_cp(
    *, config: Any = None, module: nn.Module | None = None, module_fqn: str = "", context: Any = None,
    kv_exchange: str = "all_gather", ulysses_cp_size: int = 1, ulysses_cp_strategy: str = "alltoall",
    kv_cp_size: int = 1, kv_cp_strategy: str = "allgather", kv_cp_load_balance: bool | str = False,
    ring_cp_size: int = 1, ring_cp_strategy: str = "p2p", ring_cp_load_balance: bool | str = False,
    cp_comm_overlap: bool = False,
) -> nn.Module:
    """Adapt public CP domains to Kimi's tensor-returning gated MLA member."""
    del module_fqn
    selected_strategy = ulysses_cp_strategy
    if selected_strategy == "alltoall":
        selected_strategy = "expanded_ulysses"
    return CPKimiMultiLatentAttentionModule(
        config=config, module=module, context=context, ulysses_cp_strategy=selected_strategy,
        cp_comm_overlap=cp_comm_overlap, ulysses_cp_size=ulysses_cp_size,
        kv_exchange=kv_exchange,
    )


@module_replacement
def KimiMultiLatentAttention(  # pylint: disable=invalid-name
    *, config: Any = None, module: nn.Module | None = None,
    module_fqn: str = "", context: Any = None, mesh: Any = None,
    ulysses_cp_size: int = 1, ulysses_cp_strategy: str = "alltoall",
    kv_cp_size: int = 1, kv_cp_strategy: str = "allgather", kv_cp_load_balance: bool | str = False,
    ring_cp_size: int = 1, ring_cp_strategy: str = "p2p", ring_cp_load_balance: bool | str = False,
    cp_comm_overlap: bool = False, kv_exchange: str = "all_gather",
) -> nn.Module:
    """Select the shared gated MLA algorithm while retaining the Kimi model API.

    Args:
        config: Full model configuration, exclusive with module.
        module: Optional source NoPE attention with a mandatory output gate.
        module_fqn: Replacement path used by the planner and code generator.
        context: Logical topology selecting the local or CP member.
        mesh: Existing framework mesh, without creating a separate process group.
        ulysses_cp_size: Number of ranks in the Ulysses domain.
        ulysses_cp_strategy: ``alltoall``/``expanded_ulysses`` or ``latent_kv_head``.
        kv_cp_size: Number of ranks in the KV AllGather domain.
        kv_cp_strategy: Kimi MLA currently supports ``allgather``.
        kv_cp_load_balance: KV-domain zigzag balance; currently unsupported.
        ring_cp_size: Ring domain size; currently must be one.
        ring_cp_strategy: Ring transport selector, reserved for future support.
        ring_cp_load_balance: Ring-domain zigzag balance; currently unsupported.
        cp_comm_overlap: Enable existing forward/backward overlap windows.
        kv_exchange: Full AllGather or selective causal-document AllToAllV.

    Returns:
        A family member with unchanged parameter paths and Tensor-only output.
    """
    return build_family(KimiMultiLatentAttention, config=config, module=module,
                        module_fqn=module_fqn, context=context, mesh=mesh,
                        ulysses_cp_size=ulysses_cp_size, ulysses_cp_strategy=ulysses_cp_strategy,
                        kv_cp_size=kv_cp_size, kv_cp_strategy=kv_cp_strategy,
                        kv_cp_load_balance=kv_cp_load_balance, ring_cp_size=ring_cp_size,
                        ring_cp_strategy=ring_cp_strategy, ring_cp_load_balance=ring_cp_load_balance,
                        cp_comm_overlap=cp_comm_overlap, kv_exchange=kv_exchange)


register_module_family(ModuleFamily(
    name="kimi_multi_latent_attention", baseline=KimiMultiLatentAttention,
    variants={
        "local": "hyper_parallel.models.kimi_k3.adapter.mla._build_local",
        "cp": "hyper_parallel.models.kimi_k3.adapter.mla._build_cp",
    },
    default_variant="local", parallel_axis="cp",
    variant_rules=(VariantRule({"parallel": True}, "cp"),),
    sharding=_kimi_mla_sharding, validate=gated_family.validate_gated_mla_family,
))
