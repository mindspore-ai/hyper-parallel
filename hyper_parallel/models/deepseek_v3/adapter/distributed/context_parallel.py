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
"""Independent DeepSeek dense MLA CP adapter, without DSA or MoME dependencies."""

from __future__ import annotations

from functools import wraps
from typing import Any, Literal

# Native Torch is required, as in the existing CP adapters; the legacy lint rule predates that migration.
import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import
from torch.nn import functional as F  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.modules.mla_attention import MLAAttention
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.distributed._builder.forward_rewriter import _ForwardRewriteRequest
from hyper_parallel.distributed.context_parallel.mla_context_parallel import (
    MLACPPlan, MLAContextParallel, MLACPStrategy, MLADimensions,
)
from hyper_parallel.distributed.recipe_spec import inner_wrapper


def _head_projection_request(projection: nn.Linear, head_dim: int) -> _ForwardRewriteRequest:
    """Select head rows inside the original Linear call to preserve parameter hooks."""
    original = projection.forward

    # Keep nn.Linear.forward(input=...) compatible with keyword callers.
    @wraps(original)
    def head_projection(  # pylint: disable=redefined-builtin
        input: torch.Tensor, *, mla_head_range: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Project selected contiguous heads inside the original module call."""
        if mla_head_range is None:
            return original(input)
        begin, end = mla_head_range
        if not 0 <= begin < end <= projection.weight.shape[0] // head_dim:
            raise ValueError("MLA projection head range is outside the original parameter")
        rows = slice(begin * head_dim, end * head_dim)
        bias = None if projection.bias is None else projection.bias[rows]
        # Read parameters inside the managed module's forward, after pre-hooks.
        # Slicing preserves the Parameter identity and scatters gradients back.
        return F.linear(input, projection.weight[rows], bias)  # pylint: disable=not-callable

    return _ForwardRewriteRequest(projection, head_projection)


def _validate_up_projections(target_module: MLAAttention, plan: MLACPPlan) -> None:
    """Check TP-local Linear weights without replacing managed parameters."""
    dimensions = plan.dimensions
    projections = (
        (target_module.q_b_proj, dimensions.q_rank, dimensions.qk_dim),
        (target_module.kv_b_proj, dimensions.kv_rank, dimensions.nope_dim + dimensions.value_dim),
    )
    for projection, in_features, head_dim in projections:
        # Quantized/LoRA subclasses do not obey contiguous Linear head slicing.
        if type(projection) is not nn.Linear:  # pylint: disable=unidiomatic-typecheck
            raise ValueError("MLA CP requires plain nn.Linear up projections")
        weight = projection.weight.to_local() if isinstance(projection.weight, DTensor) else projection.weight
        if weight.shape != (plan.local_heads * head_dim, in_features):
            raise ValueError("MLA up-projection shape must match TP-local heads; apply TP parameter sharding first")
        if getattr(projection.forward, "__func__", None) is not nn.Linear.forward:
            raise ValueError("MLA CP must be installed before other up-projection forward adapters")


@inner_wrapper
def mla_cp_wrapper(
    target_module: nn.Module,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
    *,
    strategy: MLACPStrategy = "expanded_ulysses",
    backend: Literal["npu", "sdpa"] = "npu",
) -> list[_ForwardRewriteRequest]:
    """Install dense MLA CP after replacement and TP sharding, before FSDP.

    The generic attention boundary owns the TP output reduction. Parameter
    gradients follow the source-layout reducer: up/Wo shards sum over CP,
    replicated down/Norm parameters additionally sum the TP contributions.
    No TP activation reduction is inserted inside this runtime.

    Args:
        target_module: The MLAAttention replacement with TP-local projections.
        mesh: Full framework mesh, supplied by the recipe rewriter.
        tp_mesh: TP axis or None when degree one is omitted by the builder.
        cp_mesh: CP axis or None when degree one is omitted by the builder.
        ep_mesh: Unused expert axis supplied by the recipe rewriter.
        strategy: Expanded QKV Ulysses or expanded Q with latent KV AllGather.
        backend: Native unequal-dimension NPU FA or CPU SDPA.

    Returns:
        Atomic forward rewrites preserving projection parameters and hooks.
    """
    del mesh, ep_mesh
    if not isinstance(target_module, MLAAttention):
        raise ValueError("mla_cp_wrapper requires the MLAAttention replacement")
    for name, axis in (("TP", tp_mesh), ("CP", cp_mesh)):
        if axis is not None and (axis.size() < 1 or getattr(axis, "ndim", 1) != 1):
            raise ValueError(f"mla_cp_wrapper requires a one-dimensional {name} mesh")
    cp_degree = 1 if cp_mesh is None else cp_mesh.size()
    tp_degree = 1 if tp_mesh is None else tp_mesh.size()
    if target_module.mla_cp_runtime is not None:
        raise ValueError("MLA CP is already configured on this module")
    if target_module.attention_dropout != 0 or target_module.sliding_window is not None:
        raise ValueError("MLA CP requires dropout=0 and no implicit sliding_window; use a supported boolean mask")
    dimensions = MLADimensions(
        heads=target_module.global_num_heads, q_rank=target_module.q_lora_rank,
        kv_rank=target_module.kv_lora_rank, nope_dim=target_module.qk_nope_head_dim,
        rope_dim=target_module.qk_rope_head_dim, value_dim=target_module.v_head_dim,
    )
    plan = MLACPPlan(dimensions, cp_degree, strategy, backend, tp_degree)
    plan.validate_scale(target_module.scaling)
    _validate_up_projections(target_module, plan)
    original = target_module.forward

    @wraps(original)
    def cp_forward(*args: Any, **kwargs: Any) -> Any:
        """Preserve the external attention signature while installing the runtime."""
        return original(*args, **kwargs)

    requests = [_ForwardRewriteRequest(
        target_module, cp_forward,
        companion_attrs={"mla_cp_runtime": MLAContextParallel(plan, cp_mesh)},
    )]
    if strategy == "latent_kv_head":
        requests.append(_head_projection_request(target_module.kv_b_proj, dimensions.nope_dim + dimensions.value_dim))
    return requests
