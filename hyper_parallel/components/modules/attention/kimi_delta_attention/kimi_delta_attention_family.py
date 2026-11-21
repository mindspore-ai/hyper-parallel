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

from hyper_parallel.components.modules.module_family import (
    ModuleFamily, VariantRule, build_family, register_module_family,
)
from hyper_parallel.components.modules.attention.linear_attention.cp_sharding import (
    linear_attention_local_spec, linear_attention_sharding,
)
from hyper_parallel.components.modules.attention.linear_attention.layer_adapter import validate_backend_options
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
    if member.axis_sizes["tp"] > 1:
        raise NotImplementedError("Linear-attention family TP is not qualified; retain the source TP path")
    backend = validate_backend_options(member.arguments)
    cp = member.axis_sizes["cp"]
    degree = member.arguments.get("ulysses_degree", 1)
    degree = cp if degree is None else degree
    if (not isinstance(degree, int) or isinstance(degree, bool)) or degree < 1 or (cp > 1 and cp % degree):
        raise ValueError("ulysses_degree must be a positive divisor of CP size")
    source = config if module is None else module
    kh = getattr(source, "num_heads", None)
    vh = getattr(source, "num_v_heads", None) or kh
    if kh is not None and cp > 1 and (kh % degree or vh % degree):
        raise ValueError("Ulysses degree must divide both query and value heads")
    _validate_kda_state_protocol(member.arguments, cp, degree, backend)


def _build_local(
    *, config: Any = None, module: nn.Module | None = None, module_fqn: str = "", context: Any = None,
    backend: str = "eager", conv_backend: str | None = None, chunk_size: int = 64,
    input_layout: str = "BSH", ulysses_degree: int | None = 1,
    state_cp_method: str = "p2p", allgather_group_size: int = 1,
) -> nn.Module:
    """Keep CP selection in the family request, outside the standalone algorithm."""
    del module_fqn, context, ulysses_degree, state_cp_method, allgather_group_size
    return KimiDeltaAttentionModule(config=config, module=module, backend=backend, conv_backend=conv_backend,
                                     chunk_size=chunk_size, input_layout=input_layout)


# Public factories follow the ModuleFamily class-style naming convention.
@module_replacement
def KimiDeltaAttention(  # pylint: disable=invalid-name
    *, config: Any = None, module: nn.Module | None = None, module_fqn: str = "",
    context: Any = None, mesh: Any = None, backend: str = "eager", conv_backend: str | None = None,
    chunk_size: int = 64, input_layout: str = "BSH", ulysses_degree: int | None = 1,
    state_cp_method: str = "p2p", allgather_group_size: int = 1,
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
        ulysses_degree: Head-exchange degree; None selects pure Ulysses scheduling.
        state_cp_method: Ordered state exchange: p2p, allgather, grouped_allgather_p2p or recursive_doubling.
        allgather_group_size: Contiguous state-group width for grouped_allgather_p2p.
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
                        chunk_size=chunk_size, input_layout=input_layout, ulysses_degree=ulysses_degree,
                        state_cp_method=state_cp_method, allgather_group_size=allgather_group_size)


register_module_family(ModuleFamily(
    name="kimi_delta_attention", baseline=KimiDeltaAttention,
    variants={
        "local": ("hyper_parallel.components.modules.attention.kimi_delta_attention."
                  "kimi_delta_attention_family._build_local"),
        "cp": ("hyper_parallel.components.modules.attention.kimi_delta_attention."
               "cp_kimi_delta_attention.CPKimiDeltaAttentionModule"),
    },
    default_variant="local", parallel_axis="cp", variant_rules=(VariantRule({"parallel": True}, "cp"),),
    sharding=linear_attention_sharding, validate=validate_kda_family,
    parallel_spec=linear_attention_local_spec,
))
