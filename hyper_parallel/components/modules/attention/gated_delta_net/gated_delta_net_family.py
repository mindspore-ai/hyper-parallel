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

from hyper_parallel.components.modules.module_family import (
    ModuleFamily, VariantRule, build_family, register_module_family,
)
from hyper_parallel.components.modules.attention.linear_attention.cp_sharding import (
    linear_attention_local_spec, linear_attention_sharding,
)
from hyper_parallel.components.modules.attention.linear_attention.layer_adapter import validate_backend_options
from hyper_parallel.models.replacement import module_replacement
from .gated_delta_net import GatedDeltaNetModule


def _validate_gdn_protocol(cp, degree, backend):
    """Reject unimplemented combinations before creating communication groups."""
    if cp > 1 and degree not in (1, cp):
        raise NotImplementedError("GDN currently supports pure Ulysses or state P2P, not their hybrid")
    if cp > degree and backend != "triton":
        raise NotImplementedError("GDN state CP requires Triton; eager and Ascend C support local/pure Ulysses")


def validate_gdn_family(*, config: Any, module: nn.Module | None, context: Any, member: Any) -> None:
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
    kh = getattr(source, "num_k_heads", None) if module is not None else source.linear_num_key_heads
    vh = getattr(source, "num_v_heads", None) if module is not None else source.linear_num_value_heads
    if kh is not None and cp > 1 and (kh % degree or vh % degree):
        raise ValueError("Ulysses degree must divide both query and value heads")
    _validate_gdn_protocol(cp, degree, backend)


def _build_local(
    *, config: Any = None, module: nn.Module | None = None, module_fqn: str = "", context: Any = None,
    backend: str = "eager", conv_backend: str | None = None, chunk_size: int = 64,
    input_layout: str = "BSH", ulysses_degree: int | None = 1,
) -> nn.Module:
    """Keep CP selection in the family request, outside the standalone algorithm."""
    del module_fqn, context, ulysses_degree
    return GatedDeltaNetModule(config=config, module=module, backend=backend, conv_backend=conv_backend,
                                     chunk_size=chunk_size, input_layout=input_layout)


# Public factories follow the ModuleFamily class-style naming convention.
@module_replacement
def GatedDeltaNet(  # pylint: disable=invalid-name
    *, config: Any = None, module: nn.Module | None = None, module_fqn: str = "",
    context: Any = None, mesh: Any = None, backend: str = "eager", conv_backend: str | None = None,
    chunk_size: int = 64, input_layout: str = "BSH", ulysses_degree: int | None = 1,
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
    """
    return build_family(GatedDeltaNet, config=config, module=module, module_fqn=module_fqn,
                        context=context, mesh=mesh, backend=backend, conv_backend=conv_backend,
                        chunk_size=chunk_size, input_layout=input_layout, ulysses_degree=ulysses_degree)


register_module_family(ModuleFamily(
    name="gated_delta_net", baseline=GatedDeltaNet,
    variants={
        "local": ("hyper_parallel.components.modules.attention.gated_delta_net."
                  "gated_delta_net_family._build_local"),
        "cp": ("hyper_parallel.components.modules.attention.gated_delta_net."
               "cp_gated_delta_net.CPGatedDeltaNetModule"),
    },
    default_variant="local", parallel_axis="cp", variant_rules=(VariantRule({"parallel": True}, "cp"),),
    sharding=linear_attention_sharding, validate=validate_gdn_family,
    parallel_spec=linear_attention_local_spec,
))
