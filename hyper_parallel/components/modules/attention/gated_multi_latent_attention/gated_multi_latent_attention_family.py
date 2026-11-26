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

from importlib import import_module
from typing import Any

from torch import nn

from hyper_parallel.components.modules.module_family import (
    ModuleFamily, VariantRule, build_family, register_module_family,
)
from hyper_parallel.components.modules.attention.multi_latent_attention.multi_latent_attention_family import (
    mla_local_parallel_spec, mla_sharding, validate_mla_family,
)
from hyper_parallel.models.replacement import module_replacement


def validate_gated_mla_family(*, config: Any, module: nn.Module | None, context: Any, member: Any) -> None:
    """Keep gated MLA's TP restriction independent of the model API adapter's family name."""
    if member.axis_sizes["tp"] != 1:
        raise ValueError("Gated MLA currently supports TP1 only")
    validate_mla_family(config=config, module=module, context=context, member=member)


def _build_local(
    *, config: Any = None, module: nn.Module | None = None, module_fqn: str = "", context: Any = None,
    kv_exchange: str = "all_gather",
    ulysses_cp_size: int = 1, ulysses_cp_strategy: str = "alltoall",
    kv_cp_size: int = 1, kv_cp_strategy: str = "allgather", kv_cp_load_balance: bool | str = False,
    ring_cp_size: int = 1, ring_cp_strategy: str = "p2p", ring_cp_load_balance: bool | str = False,
    cp_comm_overlap: bool = False,
) -> nn.Module:
    """Consume family-only options; the saved request retains them for CP preparation."""
    # Resolve after family registration; concrete members import this module's layout validators.
    implementation = import_module('.gated_multi_latent_attention', __package__).GatedMultiLatentAttentionModule

    del (module_fqn, context, kv_exchange, ulysses_cp_size,
         ulysses_cp_strategy, kv_cp_size, kv_cp_strategy, kv_cp_load_balance, ring_cp_size,
         ring_cp_strategy, ring_cp_load_balance, cp_comm_overlap)
    return implementation(config=config, module=module)


def _build_cp(
    *, config: Any = None, module: nn.Module | None = None, module_fqn: str = "", context: Any = None,
    kv_exchange: str = "all_gather", ulysses_cp_size: int = 1, ulysses_cp_strategy: str = "alltoall",
    kv_cp_size: int = 1, kv_cp_strategy: str = "allgather", kv_cp_load_balance: bool | str = False,
    ring_cp_size: int = 1, ring_cp_strategy: str = "p2p", ring_cp_load_balance: bool | str = False,
    cp_comm_overlap: bool = False,
) -> nn.Module:
    """Adapt the public domain configuration to the gated MLA CP executor."""
    del module_fqn
    selected_strategy = ulysses_cp_strategy
    if selected_strategy == "alltoall":
        selected_strategy = "expanded_ulysses"
    from .cp_gated_multi_latent_attention import CPGatedMultiLatentAttentionModule  # pylint: disable=C0415

    return CPGatedMultiLatentAttentionModule(
        config=config, module=module, context=context, ulysses_cp_strategy=selected_strategy,
        cp_comm_overlap=cp_comm_overlap, ulysses_cp_size=ulysses_cp_size,
        kv_exchange=kv_exchange,
    )


@module_replacement
def GatedMultiLatentAttention(  # pylint: disable=invalid-name
    *, config: Any = None, module: nn.Module | None = None,
    module_fqn: str = "", context: Any = None, mesh: Any = None,
    ulysses_cp_size: int = 1, ulysses_cp_strategy: str = "alltoall",
    kv_cp_size: int = 1, kv_cp_strategy: str = "allgather", kv_cp_load_balance: bool | str = False,
    ring_cp_size: int = 1, ring_cp_strategy: str = "p2p", ring_cp_load_balance: bool | str = False,
    cp_comm_overlap: bool = False, kv_exchange: str = "all_gather",
) -> nn.Module:
    """Build the local gated NoPE MLA member.

    Args:
        config: Full model dimensions for direct construction, with a mandatory output gate.
        module: Optional supported NoPE source; exclusive with config.
        module_fqn: Source path supplied by model replacement.
        context: Logical build constraints. Topology selects the local or CP implementation.
        mesh: Optional existing job mesh used by the shared layout planner.
        ulysses_cp_size: Number of ranks in the Ulysses domain.
        ulysses_cp_strategy: ``alltoall``/``expanded_ulysses`` or ``latent_kv_head``.
        kv_cp_size: Number of ranks in the KV AllGather domain.
        kv_cp_strategy: MLA currently supports ``allgather``; P2P is not implemented.
        kv_cp_load_balance: KV-domain zigzag balance; currently unsupported by MLA.
        ring_cp_size: Number of ranks in the Ring domain; currently must be one.
        ring_cp_strategy: Ring transport selector, reserved for future MLA support.
        ring_cp_load_balance: Ring-domain zigzag balance; currently unsupported by MLA.
        cp_comm_overlap: Enable existing forward/backward scheduling windows.
        kv_exchange: Full AllGather or selective causal-document AllToAllV.

    Returns:
        Gated MLA with its own NoPE math, gate Parameters and family provenance.
    """
    return build_family(GatedMultiLatentAttention, config=config, module=module,
                        module_fqn=module_fqn, context=context, mesh=mesh,
                        ulysses_cp_size=ulysses_cp_size, ulysses_cp_strategy=ulysses_cp_strategy,
                        kv_cp_size=kv_cp_size, kv_cp_strategy=kv_cp_strategy,
                        kv_cp_load_balance=kv_cp_load_balance, ring_cp_size=ring_cp_size,
                        ring_cp_strategy=ring_cp_strategy, ring_cp_load_balance=ring_cp_load_balance,
                        cp_comm_overlap=cp_comm_overlap, kv_exchange=kv_exchange)


register_module_family(ModuleFamily(
    name="gated_multi_latent_attention", baseline=GatedMultiLatentAttention,
    variants={"local": (
        "hyper_parallel.components.modules.attention.gated_multi_latent_attention."
        "gated_multi_latent_attention_family._build_local"
    ), "cp": "hyper_parallel.components.modules.attention.gated_multi_latent_attention."
          "gated_multi_latent_attention_family._build_cp"},
    default_variant="local", parallel_axis="cp",
    variant_rules=(VariantRule({"parallel": True}, "cp"),),
    sharding=mla_sharding, validate=validate_gated_mla_family, parallel_spec=mla_local_parallel_spec,
))
