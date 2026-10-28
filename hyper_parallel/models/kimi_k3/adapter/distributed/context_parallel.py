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
"""Planner-injected Context Parallel wrappers for Kimi KDA and gated MLA."""
# pylint: disable=forbidden-backend-import
from __future__ import annotations

__all__ = [
    "kimi_delta_attention_cp_wrapper",
    "kimi_mla_cp_wrapper",
    "kimi_delta_attention_p2p_cp_wrapper",
    "kimi_delta_attention_ulysses_cp_wrapper",
]

from functools import wraps
from typing import Any, Type

import torch
from torch import nn

from hyper_parallel.distributed import inner_wrapper
from hyper_parallel.distributed._builder.forward_rewriter import (
    _ForwardRewriteRequest,
)
from hyper_parallel.distributed.context_parallel.collectives import async_ulysses_seq_to_head_launch
from hyper_parallel.distributed.context_parallel.mla_backward_overlap import (
    MLAOutputExchange, mla_output_projection_request, restore_mla_output_gradient,
)
from hyper_parallel.distributed.context_parallel.mla_context_parallel import prepare_mla_cp
from hyper_parallel.models.kimi_k3.adapter.mla import KimiMLAAttention
from hyper_parallel.distributed.context_parallel.kimi_delta_attention import (
    KimiDeltaAttentionLayerCP,
    KimiDeltaAttentionLayerP2PCP,
    KimiDeltaAttentionLayerUlyssesCP,
)


def _build_kda_cp_rewrite(
    target_module: nn.Module,
    cp_mesh: Any,
    tp_mesh: Any,
    *,
    mode: str,
    executor_class: Type[nn.Module],
    backend: str,
    chunk_size: int,
    execution_options: dict[str, Any] | None = None,
) -> _ForwardRewriteRequest:
    """Validate one KDA target and return an atomic forward rewrite request."""
    if cp_mesh is None or cp_mesh.size() <= 1:
        raise ValueError(f"KDA {mode} Context Parallel requires an active CP mesh")
    if tp_mesh is not None and tp_mesh.size() > 1:
        raise NotImplementedError(
            "KDA Context Parallel does not yet support simultaneous TP and CP"
        )
    if getattr(target_module, "_hp_kda_cp_config", None) is not None:
        raise RuntimeError(
            "KDA Context Parallel has already been applied to this module; "
            "applying a CP wrapper twice is not supported"
        )

    execution_options = {} if execution_options is None else execution_options
    executor = executor_class(
        target_module,
        cp_mesh,
        chunk_size=chunk_size,
        backend=backend,
        **execution_options,
    )
    original_forward = target_module.forward

    @wraps(original_forward)
    def cp_forward(*args: Any, **kwargs: Any) -> Any:
        """Execute the selected KDA Context Parallel algorithm.

        Args:
            kwargs: Additional model arguments; packed sequences are unsupported.
        """
        return executor(*args, **kwargs)

    return _ForwardRewriteRequest(
        target_module,
        cp_forward,
        companion_attrs={
            "_hp_kda_cp_config": {
                "mode": mode,
                "backend": backend,
                "chunk_size": chunk_size,
                **execution_options,
            },
        },
    )


@inner_wrapper
def kimi_delta_attention_ulysses_cp_wrapper(
    target_module: nn.Module,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
    backend: str = "eager",
    chunk_size: int = 64,
) -> _ForwardRewriteRequest:
    """Install full-layer Ulysses CP on a Kimi Delta Attention module.

    Args:
        target_module: KDA layer whose parameters remain owned by the model.
        mesh: Planner mesh context.
        tp_mesh: Tensor-parallel mesh; simultaneous TP/CP is unsupported.
        cp_mesh: Chronologically ordered context-parallel mesh.
        ep_mesh: Expert-parallel context, unused by KDA.
        backend: Local KDA implementation: eager or triton.
        chunk_size: Tokens per local recurrence chunk.
    """
    del mesh, ep_mesh
    return _build_kda_cp_rewrite(
        target_module,
        cp_mesh,
        tp_mesh,
        mode="ulysses",
        executor_class=KimiDeltaAttentionLayerUlyssesCP,
        backend=backend,
        chunk_size=chunk_size,
    )


@inner_wrapper
def kimi_delta_attention_p2p_cp_wrapper(
    target_module: nn.Module,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
    backend: str = "eager",
    chunk_size: int = 64,
) -> _ForwardRewriteRequest:
    """Install full-layer recurrent-state P2P CP on a KDA module.

    Args:
        target_module: KDA layer whose parameters remain owned by the model.
        mesh: Planner mesh context.
        tp_mesh: Tensor-parallel mesh; simultaneous TP/CP is unsupported.
        cp_mesh: Chronologically ordered context-parallel mesh.
        ep_mesh: Expert-parallel context, unused by KDA.
        backend: Local KDA implementation: eager or triton.
        chunk_size: Tokens per local recurrence chunk.
    """
    del mesh, ep_mesh
    return _build_kda_cp_rewrite(
        target_module,
        cp_mesh,
        tp_mesh,
        mode="p2p",
        executor_class=KimiDeltaAttentionLayerP2PCP,
        backend=backend,
        chunk_size=chunk_size,
    )


@inner_wrapper
def kimi_delta_attention_cp_wrapper(
    target_module: nn.Module,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
    backend: str = "triton",
    state_cp_method: str = "p2p",
    ulysses_degree: int = 1,
    allgather_group_size: int = 1,
) -> _ForwardRewriteRequest:
    """Install KDA state CP, optionally combined with Ulysses and group-local AG.

    This unified entry uses 64-token local chunks for both backends.

    Args:
        target_module: Training-time KDA layer, retaining its parameter ownership.
        mesh: Planner mesh context.
        tp_mesh: Simultaneous TP and CP is currently unsupported.
        cp_mesh: Chronological CP mesh, retaining its root for hybrid splitting.
        ep_mesh: Unused expert-parallel context.
        backend: Local backend; gathers require triton.
        state_cp_method: State CP method: p2p, allgather, or grouped_allgather_p2p.
        ulysses_degree: Consecutive ranks exchanging token/head shards.
        allgather_group_size: AG width along state CP, only for grouped_allgather_p2p.

    Returns:
        Atomic forward rewrite with the selected configuration recorded.
    """
    del mesh, ep_mesh
    if not isinstance(ulysses_degree, int) or isinstance(ulysses_degree, bool) or ulysses_degree < 1:
        raise ValueError("ulysses_degree must be a positive integer.")
    options = {"state_cp_method": state_cp_method, "allgather_group_size": allgather_group_size,
               "ulysses_degree": ulysses_degree}
    return _build_kda_cp_rewrite(
        target_module, cp_mesh, tp_mesh, mode="hybrid" if ulysses_degree > 1 else "state",
        executor_class=KimiDeltaAttentionLayerCP, backend=backend, chunk_size=64,
        execution_options=options,
    )


@inner_wrapper
def kimi_mla_cp_wrapper(
    target_module: nn.Module, mesh: Any, tp_mesh: Any, cp_mesh: Any, ep_mesh: Any,
    *, strategy: str = "expanded_ulysses", overlap: bool = False,
    ulysses_degree: int | None = None, kv_exchange: str = "all_gather",
    backward_overlap: bool = False,
) -> list[_ForwardRewriteRequest]:
    """Attach shared MLA CP to gated NoPE attention at TP1.

    Args:
        target_module: Kimi MLA replacement retaining its original gate.
        mesh: Full framework mesh, unused.
        tp_mesh: Tensor-parallel mesh; simultaneous TP is unsupported.
        cp_mesh: Chronologically ordered context-parallel mesh.
        ep_mesh: Expert mesh, unused.
        strategy: Expanded QKV or shared latent KV representation.
        overlap: Cover input communication with projections and output communication with the local gate.
        ulysses_degree: Head-parallel degree, defaulting to the complete CP group.
        kv_exchange: Full gather or selective KV exchange.
        backward_overlap: Schedule output adjoints and, with forward overlap, pure-Ulysses input adjoints.

    Returns:
        Atomic model and child-projection rewrites preserving module hooks.
    """
    del mesh, ep_mesh
    if not isinstance(backward_overlap, bool):
        raise ValueError("backward_overlap must be bool")
    if not isinstance(target_module, KimiMLAAttention):
        raise ValueError("Kimi MLA CP requires the KimiMLAAttention replacement")
    if tp_mesh is not None and tp_mesh.size() > 1:
        raise ValueError("Kimi MLA CP currently supports TP1 only")
    runtime, children = prepare_mla_cp(
        target_module, cp_mesh, tp_mesh, strategy=strategy, overlap=overlap,
        ulysses_degree=ulysses_degree, kv_exchange=kv_exchange,
    )
    if backward_overlap:
        if runtime.plan.ulysses_degree <= 1:
            raise ValueError("MLA output backward overlap requires a Ulysses degree greater than one")
        children.append(mla_output_projection_request(target_module.o_proj))
    original = target_module.forward

    @wraps(original)
    def cp_forward(hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None,
                   position_ids: torch.Tensor | None = None, past_key_values: Any | None = None,
                   **kwargs: Any) -> torch.Tensor:
        """Keep NoPE positions and output gating at their model-owned boundary.

        Args:
            hidden_states: Chronological local tokens in BSH layout.
            attention_mask: Optional mask accepted by the selected CP strategy.
            position_ids: Unused by NoPE attention.
            past_key_values: Unsupported training-time KV cache.
            **kwargs: Packed lengths and attention options.

        Returns:
            Projected hidden states in the original local-token order.
        """
        del position_ids
        kwargs.pop("position_embeddings", None)
        if past_key_values is not None:
            raise ValueError("Kimi MLA CP does not support a KV cache")
        return _gated_mla_forward(
            target_module, runtime, hidden_states, attention_mask, kwargs, backward_overlap=backward_overlap,
        )

    return [_ForwardRewriteRequest(
        target_module, cp_forward, companion_attrs={"mla_cp_runtime": runtime},
    )] + children


def _gated_mla_forward(module, runtime, hidden_states, attention_mask, options, *, backward_overlap=False):
    """Prepare the token-local gate while the attention output returns to its owner."""
    output, _ = runtime(
        module, *module.project_latent_inputs(hidden_states), attention_mask=attention_mask,
        restore_output=False, backward_overlap=backward_overlap, **options,
    )
    attention = output
    if backward_overlap:
        output = output.detach()
    pending = None
    if runtime.plan.overlap and runtime.plan.ulysses_degree > 1:
        pending = async_ulysses_seq_to_head_launch(output, 2, 1, runtime.ulysses_mesh)
    else:
        output = runtime.restore_output(output)
    gate = module.g_proj(hidden_states).sigmoid() if module.use_output_gate else None
    if pending is not None:
        output = pending.wait()
    output = output.reshape(*hidden_states.shape[:2], -1)
    if backward_overlap:
        state = MLAOutputExchange(runtime.ulysses_mesh, module.v_head_dim)
        output = restore_mla_output_gradient(attention, output, state)
        return module.o_proj(output, mla_output_exchange=state, mla_output_gate=gate)
    return module.o_proj(output if gate is None else output * gate)
