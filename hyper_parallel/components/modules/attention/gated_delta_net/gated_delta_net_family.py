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
"""GatedDeltaNet family selection; backend and communication are independent options."""
from __future__ import annotations

from typing import Any
from torch import nn

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec

from hyper_parallel.components.modules.module_family import (
    ModuleFamily, VariantRule, build_family, register_module_family,
)
from hyper_parallel.components.modules.attention.linear_attention_utils.parallel_layout import (
    linear_attention_local_spec, linear_attention_sharding, validate_state_packets,
)
from hyper_parallel.components.modules.attention.linear_attention_utils.layer_adapter import validate_backend_options
from hyper_parallel.models.replacement import module_replacement
from .gated_delta_net import GatedDeltaNetModule


def _validate_gdn_protocol(cp, degree, backend, arguments):
    """Reject unimplemented combinations before creating communication groups."""
    if 1 < degree < cp and backend != "ascendc":
        raise NotImplementedError("GDN hybrid Ulysses/state CP currently requires Ascend C")
    method = arguments.get("state_cp_method", "p2p")
    if method not in ("p2p", "allgather"):
        raise ValueError("GDN state_cp_method must be p2p or allgather")
    if cp > degree and backend not in ("triton", "ascendc"):
        raise NotImplementedError("GDN state CP requires Triton or Ascend C")
    if method == "allgather" and backend != "ascendc":
        raise NotImplementedError("GDN AllGather currently requires Ascend C")


def validate_gdn_family(*, config: Any, module: nn.Module | None, context: Any, member: Any) -> None:
    """Check this family's supported algorithms before constructing parameters or groups."""
    del context
    backend = validate_backend_options(member.arguments)
    cp = member.axis_sizes["cp"]
    degree = member.arguments.get("ulysses_degree", 1)
    degree = cp if degree is None else degree
    if (not isinstance(degree, int) or isinstance(degree, bool)) or degree < 1 or (cp > 1 and cp % degree):
        raise ValueError("ulysses_degree must be a positive divisor of CP size")
    source = config if module is None else module
    kh = getattr(source, "num_k_heads", None) if module is not None else source.linear_num_key_heads
    vh = getattr(source, "num_v_heads", None) if module is not None else source.linear_num_value_heads
    head_degree = member.axis_sizes["tp"] * (degree if cp > 1 else 1)
    if kh is not None and (kh % head_degree or vh % head_degree):
        raise ValueError("TP times Ulysses degree must divide both query and value heads")
    _validate_gdn_protocol(cp, degree, backend, member.arguments)
    validate_state_packets(member.arguments, cp, degree, backend, None if vh is None else vh // head_degree)


def gdn_sharding(*, module: nn.Module, module_fqn: str, member: Any, context: Any) -> ModuleShardingSpec:
    """Shard complete Q/K/value groups while retaining the shared per-head norm.

    Args:
        module: GDN family member with grouped QKV and convolution rows.
        module_fqn: Source path supplied by the common layout planner.
        member: Resolved family selection.
        context: Logical TP/CP sizes and SP mode.
    """
    del module_fqn, member
    columns = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "conv1d")
    roles = {f"{name}.{suffix}": Shard(0) for name in columns for suffix in ("weight", "bias")}
    roles.update({"A_log": Shard(0), "dt_bias": Shard(0), "out_proj.weight": Shard(1),
                  "out_proj.bias": Replicate(), "norm.weight": Replicate(), "norm.bias": Replicate()})
    return linear_attention_sharding(
        module=module, context=context, roles=roles,
        tp_divide_attrs=("num_k_heads", "num_v_heads", "key_dim", "value_dim", "conv_dim"))


def _build_local(
    *, config: Any = None, module: nn.Module | None = None, module_fqn: str = "", context: Any = None,
    backend: str = "eager", conv_backend: str | None = None, chunk_size: int = 64,
    input_layout: str = "BSH", ulysses_degree: int | None = 1,
    state_cp_method: str = "p2p", p2p_head_chunks: int = 1, reuse_cp_state_on_recompute: bool = False,
) -> nn.Module:
    """Keep CP selection in the family request, outside the standalone algorithm."""
    del module_fqn, context, ulysses_degree, state_cp_method, p2p_head_chunks, reuse_cp_state_on_recompute
    return GatedDeltaNetModule(config=config, module=module, backend=backend, conv_backend=conv_backend,
                                     chunk_size=chunk_size, input_layout=input_layout)


# Public factories follow the ModuleFamily class-style naming convention.
@module_replacement
def GatedDeltaNet(  # pylint: disable=invalid-name
    *, config: Any = None, module: nn.Module | None = None, module_fqn: str = "",
    context: Any = None, mesh: Any = None, backend: str = "eager", conv_backend: str | None = None,
    chunk_size: int = 64, input_layout: str = "BSH", ulysses_degree: int | None = 1,
    state_cp_method: str = "p2p", p2p_head_chunks: int = 1, reuse_cp_state_on_recompute: bool = False,
) -> nn.Module:
    """Build the complete local or CP training member with source checkpoint names.

    Args:
        config: Direct model dimensions; mutually exclusive with module.
        module: Supported source; QKV and convolution rows receive a reversible grouped conversion.
        module_fqn: Path supplied by model replacement.
        context: Logical parallel degrees for selection without process groups.
        mesh: Existing runtime mesh for layout and resource binding.
        backend: Explicit eager, triton or ascendc attention implementation.
        conv_backend: Independent convolution implementation; omission preserves scheduling defaults.
        chunk_size: Local chunk length; fused implementations require 64.
        input_layout: BSH or TH, determining the external token shard axis.
        ulysses_degree: Head-exchange degree; None selects pure Ulysses scheduling.
        state_cp_method: p2p, or allgather for the native state-CP backend.
        p2p_head_chunks: Number of native state packets, independent of local Prepare.
        reuse_cp_state_on_recompute: Retain the native CP entry across outer checkpoint replay; default False.
    """
    return build_family(GatedDeltaNet, config=config, module=module, module_fqn=module_fqn,
                        context=context, mesh=mesh, backend=backend, conv_backend=conv_backend,
                        chunk_size=chunk_size, input_layout=input_layout, ulysses_degree=ulysses_degree,
                        state_cp_method=state_cp_method,
                        p2p_head_chunks=p2p_head_chunks, reuse_cp_state_on_recompute=reuse_cp_state_on_recompute)


register_module_family(ModuleFamily(
    name="gated_delta_net", baseline=GatedDeltaNet,
    variants={
        "local": ("hyper_parallel.components.modules.attention.gated_delta_net."
                  "gated_delta_net_family._build_local"),
        "cp": ("hyper_parallel.components.modules.attention.gated_delta_net."
               "cp_gated_delta_net.CPGatedDeltaNetModule"),
    },
    default_variant="local", parallel_axis="cp", variant_rules=(VariantRule({"parallel": True}, "cp"),),
    sharding=gdn_sharding, validate=validate_gdn_family,
    parallel_spec=linear_attention_local_spec,
))
