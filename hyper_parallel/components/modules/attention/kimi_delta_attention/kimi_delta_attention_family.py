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
"""KimiDeltaAttention family selection; backend and communication are independent options."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from torch import nn

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec

from hyper_parallel.components.modules.module_family import (
    ModuleFamily, VariantRule, build_family, register_module_family,
)
from hyper_parallel.components.modules.attention.cp_config import (
    resolve_attention_cp_config, validate_supported_cp_options,
)
from hyper_parallel.components.modules.attention.linear_attention_utils.parallel_layout import (
    linear_attention_local_spec, linear_attention_sharding,
)
from hyper_parallel.components.modules.attention.linear_attention_utils.layer_adapter import validate_backend_options
from hyper_parallel.models.replacement import module_replacement
from .kimi_delta_attention import KimiDeltaAttentionModule


def _validate_kda_state_protocol(options: Any, cp: int, degree: int, backend: str) -> None:
    """Keep state protocol qualifications separate from layout validation."""
    protocol = options.get("state_cp_method", "p2p")
    if protocol not in ("p2p", "allgather", "grouped_allgather_p2p", "recursive_doubling"):
        raise ValueError("Unknown KDA state_cp_method")
    group = options.get("allgather_group_size", 1)
    if (not isinstance(group, int) or isinstance(group, bool)) or group < 1 or (cp > 1 and (cp // degree) % group):
        raise ValueError("allgather_group_size must divide state CP size")
    if group != 1 and protocol != "grouped_allgather_p2p":
        raise ValueError("allgather_group_size is only valid for grouped_allgather_p2p")
    if cp > degree and backend == "eager":
        raise NotImplementedError("Eager KDA supports local/pure Ulysses only; state CP requires Triton or Ascend C")


def validate_kda_family(*, config: Any, module: nn.Module | None, context: Any, member: Any) -> None:
    """Check this family's supported algorithms before constructing parameters or groups."""
    del context
    backend = validate_backend_options(member.arguments)
    cp = member.axis_sizes["cp"]
    cp_config = resolve_attention_cp_config(member)
    validate_supported_cp_options(
        cp_config, "KDA", ulysses_strategies=("alltoall",),
        ring_strategies=("p2p", "allgather"), allow_ulysses=True, allow_kv=False,
        allow_ring=True,
    )
    degree = cp_config["ulysses_cp_size"]
    if (not isinstance(degree, int) or isinstance(degree, bool)) or degree < 1 or (cp > 1 and cp % degree):
        raise ValueError("ulysses_cp_size must be a positive divisor of cp_size")
    source = config if module is None else module
    kh = getattr(source, "num_heads", None)
    vh = getattr(source, "num_v_heads", None) or kh
    head_degree = member.axis_sizes["tp"] * (degree if cp > 1 else 1)
    if kh is not None and (kh % head_degree or vh % head_degree):
        raise ValueError("TP times ulysses_cp_size must divide both query and value heads")
    state_method = member.arguments.get("state_cp_method") or cp_config["ring_cp_strategy"]
    _validate_kda_state_protocol({**member.arguments, "state_cp_method": state_method}, cp, degree, backend)


def kda_sharding(*, module: nn.Module, module_fqn: str, member: Any, context: Any) -> ModuleShardingSpec:
    """Declare head ownership without splitting the shared low-rank gate inputs.

    Args:
        module: KDA family member with source-owned projection parameters.
        module_fqn: Source path supplied by the common layout planner.
        member: Resolved family selection.
        context: Logical TP/CP sizes and SP mode.
    """
    del module_fqn, member
    columns = ("q_proj", "k_proj", "v_proj", "b_proj", "f_b_proj", "g_b_proj",
               "q_conv1d", "k_conv1d", "v_conv1d")
    roles = {f"{name}.{suffix}": Shard(0) for name in columns for suffix in ("weight", "bias")}
    roles.update({f"{name}.{suffix}": Replicate() for name in ("f_a_proj", "g_a_proj", "o_norm")
                  for suffix in ("weight", "bias")})
    roles.update({"A_log": Shard(0), "dt_bias": Shard(0),
                  "o_proj.weight": Shard(1), "o_proj.bias": Replicate()})
    for name in ("f_proj", "g_proj"):
        projection = getattr(module, name, None)
        if projection is None:
            continue
        linears = [(path, child) for path, child in projection.named_modules() if isinstance(child, nn.Linear)]
        for index, (path, _) in enumerate(linears):
            owner = f"{name}.{path}" if path else name
            placement = Shard(0) if index == len(linears) - 1 else Replicate()
            roles.update({f"{owner}.{suffix}": placement for suffix in ("weight", "bias")})
    return linear_attention_sharding(module=module, context=context, roles=roles,
                                      tp_divide_attrs=("num_k_heads", "num_v_heads", "key_dim", "value_dim"))


def _build_local(
    *, config: Any = None, module: nn.Module | None = None, module_fqn: str = "", context: Any = None,
    backend: str = "eager", conv_backend: str | None = None, chunk_size: int = 64,
    input_layout: str = "BSH",
    state_cp_method: str | None = None, allgather_group_size: int = 1,
    ulysses_cp_size: int = 1, ulysses_cp_strategy: str = "alltoall",
    kv_cp_size: int = 1, kv_cp_strategy: str = "allgather", kv_cp_load_balance: bool | str = False,
    ring_cp_size: int = 1, ring_cp_strategy: str = "p2p", ring_cp_load_balance: bool | str = False,
    cp_comm_overlap: bool = False,
) -> nn.Module:
    """Keep CP selection in the family request, outside the standalone algorithm."""
    del (module_fqn, context, state_cp_method, allgather_group_size,
         ulysses_cp_size, ulysses_cp_strategy, kv_cp_size, kv_cp_strategy, kv_cp_load_balance,
         ring_cp_size, ring_cp_strategy, ring_cp_load_balance, cp_comm_overlap)
    return KimiDeltaAttentionModule(config=config, module=module, backend=backend, conv_backend=conv_backend,
                                     chunk_size=chunk_size, input_layout=input_layout)


def _build_cp(
    *, config: Any = None, module: nn.Module | None = None, module_fqn: str = "", context: Any = None,
    backend: str = "eager", conv_backend: str | None = None, chunk_size: int = 64,
    input_layout: str = "BSH", state_cp_method: str | None = None, allgather_group_size: int = 1,
    ulysses_cp_size: int = 1, ulysses_cp_strategy: str = "alltoall",
    kv_cp_size: int = 1, kv_cp_strategy: str = "allgather", kv_cp_load_balance: bool | str = False,
    ring_cp_size: int = 1, ring_cp_strategy: str = "p2p", ring_cp_load_balance: bool | str = False,
    cp_comm_overlap: bool = False,
) -> nn.Module:
    """Map Ulysses and Ring domains onto KDA's head and recurrent-state groups."""
    del module_fqn
    method = state_cp_method or ring_cp_strategy
    from .cp_kimi_delta_attention import CPKimiDeltaAttentionModule  # pylint: disable=C0415

    return CPKimiDeltaAttentionModule(
        config=config, module=module, context=context, backend=backend, conv_backend=conv_backend,
        chunk_size=chunk_size, input_layout=input_layout, ulysses_cp_size=ulysses_cp_size,
        state_cp_method=method, allgather_group_size=allgather_group_size,
    )


# Public factories follow the ModuleFamily class-style naming convention.
@module_replacement
def KimiDeltaAttention(  # pylint: disable=invalid-name
    *, config: Any = None, module: nn.Module | None = None, module_fqn: str = "",
    context: Any = None, mesh: Any = None, backend: str = "eager", conv_backend: str | None = None,
    chunk_size: int = 64, input_layout: str = "BSH",
    ulysses_cp_size: int = 1, ulysses_cp_strategy: str = "alltoall",
    kv_cp_size: int = 1, kv_cp_strategy: str = "allgather", kv_cp_load_balance: bool | str = False,
    ring_cp_size: int = 1, ring_cp_strategy: str = "p2p", ring_cp_load_balance: bool | str = False,
    cp_comm_overlap: bool = False,
    state_cp_method: str | None = None, allgather_group_size: int = 1,
    hidden_size: int | None = None, num_heads: int = 32, num_v_heads: int | None = None,
    head_k_dim: int = 128, head_v_dim: int = 128, conv_kernel_size: int = 4,
    gate_lower_bound: float = -5.0, use_full_rank_gate: bool = True, rms_norm_eps: float = 1e-6,
    layer_idx: int = 0,
) -> nn.Module:
    """Build the complete local or CP training member with source checkpoint names.

    Args:
        config: Direct model dimensions; mutually exclusive with module.
        module: Existing supported source whose parameters are retained.
        module_fqn: Path supplied by model replacement.
        context: Logical parallel degrees for selection without process groups.
        mesh: Existing runtime mesh for layout and resource binding.
        backend: Explicit eager, triton or ascendc attention implementation.
        conv_backend: Independent convolution implementation; omission preserves scheduling defaults.
        chunk_size: Local chunk length; fused implementations require 64.
        input_layout: BSH or TH, determining the external token shard axis.
        ulysses_cp_size: Number of ranks in the Ulysses head-exchange domain.
        ulysses_cp_strategy: KDA currently supports ``alltoall``.
        kv_cp_size: Must be one; KDA transports recurrent state rather than a KV attention bank.
        kv_cp_strategy: Reserved for KDA; current state exchange is configured by the Ring domain.
        kv_cp_load_balance: KV-domain zigzag balance; unsupported by KDA.
        ring_cp_size: Number of ranks in the recurrent-state exchange domain.
        ring_cp_strategy: KDA currently maps ``p2p`` or ``allgather`` to its state protocol.
        ring_cp_load_balance: Ring zigzag balance; unsupported by KDA.
        cp_comm_overlap: General CP overlap toggle; currently unsupported by KDA.
        state_cp_method: Legacy KDA-specific state-exchange selector.
        allgather_group_size: Legacy grouped-AllGather extension.
        hidden_size: Legacy direct-construction model width; exclusive with config and module.
        num_heads: Query/key head count for direct construction.
        num_v_heads: Value head count; defaults to num_heads.
        head_k_dim: Channels per query/key head.
        head_v_dim: Channels per value head.
        conv_kernel_size: Causal depthwise convolution width.
        gate_lower_bound: Negative lower bound used by the safe decay gate.
        use_full_rank_gate: Whether direct construction uses a full-width output gate.
        rms_norm_eps: Output normalization epsilon.
        layer_idx: Source layer index retained in direct construction.
    """
    if hidden_size is not None:
        if config is not None or module is not None:
            raise ValueError("hidden_size construction cannot be combined with config or module")
        config = SimpleNamespace(hidden_size=hidden_size, num_heads=num_heads, num_v_heads=num_v_heads,
                                 head_k_dim=head_k_dim, head_v_dim=head_v_dim, conv_kernel_size=conv_kernel_size,
                                 gate_lower_bound=gate_lower_bound, use_full_rank_gate=use_full_rank_gate,
                                 rms_norm_eps=rms_norm_eps, layer_idx=layer_idx)
    return build_family(KimiDeltaAttention, config=config, module=module, module_fqn=module_fqn,
                        context=context, mesh=mesh, backend=backend, conv_backend=conv_backend,
                        chunk_size=chunk_size, input_layout=input_layout,
                        ulysses_cp_size=ulysses_cp_size, ulysses_cp_strategy=ulysses_cp_strategy,
                        kv_cp_size=kv_cp_size, kv_cp_strategy=kv_cp_strategy,
                        kv_cp_load_balance=kv_cp_load_balance, ring_cp_size=ring_cp_size,
                        ring_cp_strategy=ring_cp_strategy, ring_cp_load_balance=ring_cp_load_balance,
                        cp_comm_overlap=cp_comm_overlap,
                        state_cp_method=state_cp_method, allgather_group_size=allgather_group_size)


register_module_family(ModuleFamily(
    name="kimi_delta_attention", baseline=KimiDeltaAttention,
    variants={
        "local": ("hyper_parallel.components.modules.attention.kimi_delta_attention."
                  "kimi_delta_attention_family._build_local"),
        "cp": ("hyper_parallel.components.modules.attention.kimi_delta_attention."
               "kimi_delta_attention_family._build_cp"),
    },
    default_variant="local", parallel_axis="cp", variant_rules=(VariantRule({"parallel": True}, "cp"),),
    sharding=kda_sharding, validate=validate_kda_family,
    parallel_spec=linear_attention_local_spec,
))
