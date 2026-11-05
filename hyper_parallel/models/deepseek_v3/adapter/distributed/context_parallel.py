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

from collections.abc import Callable, Sequence
from functools import wraps
from typing import Any, Literal

# Native Torch is required, as in the existing CP adapters; the legacy lint rule predates that migration.
import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.modules.mla_attention import MLAAttention
from hyper_parallel.distributed._builder.forward_rewriter import _ForwardRewriteRequest
from hyper_parallel.distributed.context_parallel.mla_backward_overlap import (
    MLAOutputProjectionState, mla_output_projection_request,
)
from hyper_parallel.distributed.context_parallel.mla_context_parallel import (
    MLACPStrategy, prepare_mla_cp,
)
from hyper_parallel.distributed.recipe_spec import inner_wrapper


def mla_cp_forward(
    target_module: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    past_key_values: Any | None = None,
    actual_seq_len: torch.Tensor | Sequence[int] | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the native MLA CP path shared by the wrapper and generated source."""
    if past_key_values is not None:
        raise ValueError("MLA CP training does not support a KV cache")
    batch_size, sequence_length = hidden_states.shape[:-1]
    runtime = target_module.mla_cp_runtime
    supports_output_backward = runtime.plan.overlap and runtime.plan.tp_degree == 1 and runtime.plan.ulysses_degree > 1
    overlap_output_backward = supports_output_backward and torch.is_grad_enabled()
    # Hybrid CP supports output dO overlap, but paired input adjoints are
    # currently implemented only for pure Ulysses.
    overlap_input_backward = overlap_output_backward and runtime.plan.all_gather_degree == 1
    attn_output, attn_weights = runtime(
        target_module, *target_module.project_latent_inputs(hidden_states),
        position_embeddings=position_embeddings, attention_mask=attention_mask,
        actual_seq_len=actual_seq_len, restore_output=not overlap_output_backward,
        overlap_input_backward=overlap_input_backward, **kwargs,
    )
    if overlap_output_backward:
        state = MLAOutputProjectionState(runtime.ulysses_mesh, target_module.v_head_dim)
        return target_module.o_proj(
            attn_output, mla_output_state=state,
        ), attn_weights
    attn_output = attn_output.reshape(batch_size, sequence_length, -1).contiguous()
    return target_module.project_output(attn_output, hidden_states), attn_weights


@inner_wrapper
def mla_cp_wrapper(
    target_module: nn.Module,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
    *,
    strategy: MLACPStrategy = "expanded_ulysses",
    overlap: bool = False,
    ulysses_degree: int | None = None,
    kv_exchange: Literal["all_gather", "all_to_all_v"] = "all_gather",
    rotary_fn: Callable[..., torch.Tensor] | None = None,
    attention_interface: Callable[..., tuple[torch.Tensor, Any]] | None = None,
) -> list[_ForwardRewriteRequest]:
    """Install NPU MLA CP after replacement and TP sharding, before FSDP.

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
        overlap: Enable forward communication overlap and supported backward scheduling.
            Output backward overlap requires TP1 and Ulysses degree > 1, including
            hybrid CP. Paired input backward additionally requires pure Ulysses.
        ulysses_degree: Defaults to CP size; at TP1, one selects KV AllGather,
            and intermediate CP divisors select Ulysses x AllGather.
        kv_exchange: Full KV collection or selective causal-prefix distribution.
        rotary_fn: Optional local RoPE callback forwarded to the shared MLA runtime.
        attention_interface: Optional local attention callback forwarded to the shared MLA runtime.

    Returns:
        Atomic forward rewrites preserving projection parameters and hooks.

    Note:
        Scheduled backward supports complete first-order training backward. Use
        overlap=False for partial autograd.grad calls or higher-order derivatives.
    """
    del mesh, ep_mesh
    if not isinstance(target_module, MLAAttention):
        raise ValueError("mla_cp_wrapper requires the MLAAttention replacement")
    runtime, child_requests = prepare_mla_cp(
        target_module, cp_mesh, tp_mesh, strategy=strategy,
        overlap=overlap, ulysses_degree=ulysses_degree, kv_exchange=kv_exchange,
        rotary_fn=rotary_fn, attention_interface=attention_interface,
    )
    supports_output_backward = overlap and runtime.plan.tp_degree == 1 and runtime.plan.ulysses_degree > 1
    if supports_output_backward:
        child_requests.append(mla_output_projection_request(target_module.o_proj))
    original = target_module.forward

    @wraps(original)
    def cp_forward(
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        actual_seq_len: torch.Tensor | Sequence[int] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Route latent inputs through CP with the model's forward contract.

        Args:
            hidden_states: Local sequence shard in batch, sequence, hidden order.
            position_embeddings: Optional RoPE cosine and sine for local tokens.
            attention_mask: Optional global boolean mask, with True for allowed keys.
            past_key_values: Must be None for CP training.
            actual_seq_len: Optional cumulative global packed document ends.
            **kwargs: Causal, packed and supported model forward options.

        Returns:
            Projected local-token output and None for attention weights.
        """
        return mla_cp_forward(
            target_module,
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            actual_seq_len=actual_seq_len,
            **kwargs,
        )

    requests = [_ForwardRewriteRequest(
        target_module, cp_forward,
        companion_attrs={"mla_cp_runtime": runtime},
    )]
    return requests + child_requests
