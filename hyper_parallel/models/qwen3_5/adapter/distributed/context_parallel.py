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
"""Planner-injected Context Parallel wrappers for Qwen3.5 Gated DeltaNet."""
# pylint: disable=forbidden-backend-import

from __future__ import annotations

__all__ = [
    "qwen3_5_gdn_p2p_cp_wrapper",
    "qwen3_5_gdn_ulysses_cp_wrapper",
    "qwen3_5_packed_decoder_wrapper",
    "qwen3_5_gdn_packed_local_wrapper",
]

from functools import wraps
from typing import Any, Type

from torch import Tensor, nn

from hyper_parallel.distributed import inner_wrapper
from hyper_parallel.distributed._builder.forward_rewriter import (
    _ForwardRewriteRequest,
)
from hyper_parallel.distributed.context_parallel.gated_delta_net import (
    GatedDeltaNetP2PCP,
    GatedDeltaNetUlyssesCP,
)


def _build_gdn_cp_rewrite(
    target_module: nn.Module,
    cp_mesh: Any,
    tp_mesh: Any,
    *,
    mode: str,
    executor_class: Type[nn.Module],
    backend: str,
    chunk_size: int,
) -> _ForwardRewriteRequest:
    """Validate one GDN target and return an atomic forward rewrite request."""
    if cp_mesh is None or cp_mesh.size() <= 1:
        raise ValueError(f"GDN {mode} Context Parallel requires an active CP mesh")
    if tp_mesh is not None and tp_mesh.size() > 1:
        raise NotImplementedError(
            "Qwen3.5 GDN Context Parallel does not yet support simultaneous TP and CP"
        )
    if getattr(target_module, "_hp_gdn_cp_config", None) is not None:
        raise RuntimeError(
            "Qwen3.5 GDN Context Parallel has already been applied to this module; "
            "applying a CP wrapper twice is not supported"
        )

    executor = executor_class(
        target_module,
        cp_mesh,
        backend=backend,
        chunk_size=chunk_size,
    )
    original_forward = target_module.forward

    @wraps(original_forward)
    def cp_forward(*args: Any, **kwargs: Any) -> Any:
        """Execute the selected full-layer GDN Context Parallel algorithm."""
        return executor(*args, **kwargs)

    return _ForwardRewriteRequest(
        target_module,
        cp_forward,
        companion_attrs={
            "_hp_gdn_cp_config": {
                "mode": mode,
                "backend": backend,
                "chunk_size": chunk_size,
            },
        },
    )


@inner_wrapper
def qwen3_5_gdn_ulysses_cp_wrapper(
    target_module: nn.Module,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
    backend: str = "eager",
    chunk_size: int = 64,
) -> _ForwardRewriteRequest:
    """Install full-layer Ulysses CP on a Qwen3.5 GDN module."""
    del mesh, ep_mesh
    return _build_gdn_cp_rewrite(
        target_module,
        cp_mesh,
        tp_mesh,
        mode="ulysses",
        executor_class=GatedDeltaNetUlyssesCP,
        backend=backend,
        chunk_size=chunk_size,
    )


@inner_wrapper
def qwen3_5_gdn_p2p_cp_wrapper(
    target_module: nn.Module,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
    backend: str = "eager",
    chunk_size: int = 64,
) -> _ForwardRewriteRequest:
    """Install full-layer recurrent-state P2P CP on a Qwen3.5 GDN module."""
    del mesh, ep_mesh
    return _build_gdn_cp_rewrite(
        target_module,
        cp_mesh,
        tp_mesh,
        mode="p2p",
        executor_class=GatedDeltaNetP2PCP,
        backend=backend,
        chunk_size=chunk_size,
    )


@inner_wrapper
def qwen3_5_packed_decoder_wrapper(
    target_module: nn.Module,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
) -> _ForwardRewriteRequest:
    """Forward explicit per-invocation sample metadata through a linear decoder.

    Install on linear-attention decoder layers in addition to the GDN core rule.
    Full-attention decoders already forward kwargs and need their existing
    packed-capable attention backend. No parameters or module FQNs are changed.
    """
    del mesh, tp_mesh, cp_mesh, ep_mesh
    required = ("linear_attn", "input_layernorm", "post_attention_layernorm", "mlp")
    if getattr(target_module, "layer_type", None) != "linear_attention" or any(
        not hasattr(target_module, name) for name in required
    ):
        raise TypeError("Packed decoder wrapper requires a Qwen3.5 linear-attention decoder.")
    original_forward = target_module.forward

    @wraps(original_forward)
    def packed_forward(
        hidden_states: Tensor,
        position_embeddings: Any = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: Any = None,
        **kwargs: Any,
    ) -> Tensor:
        """Keep metadata local to this invocation, including checkpoint replay."""
        packed = any(
            kwargs.get(key) is not None
            for key in ("packed_sequence_info", "cu_seq_lens", "cu_seqlens", "cu_seq_lens_q", "cu_seq_lens_k")
        )
        if not packed:
            return original_forward(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )
        if past_key_values is not None or kwargs.get("use_cache", False):
            raise NotImplementedError("Packed linear attention supports training without recurrent cache.")
        if getattr(target_module.linear_attn, "_hp_gdn_cp_config", None) is None:
            raise RuntimeError("Install a packed local or CP GDN wrapper before the packed decoder wrapper.")
        residual = hidden_states
        hidden_states = target_module.input_layernorm(hidden_states)
        hidden_states = target_module.linear_attn(
            hidden_states=hidden_states, cache_params=None, attention_mask=attention_mask, **kwargs
        )
        hidden_states = residual + hidden_states
        return hidden_states + target_module.mlp(target_module.post_attention_layernorm(hidden_states))

    return _ForwardRewriteRequest(target_module, packed_forward)


@inner_wrapper
def qwen3_5_gdn_packed_local_wrapper(
    target_module: nn.Module,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
    backend: str = "eager",
    chunk_size: int = 64,
) -> _ForwardRewriteRequest:
    """Use the shared GDN packed path at CP1 while preserving the dense forward."""
    del mesh, ep_mesh
    if cp_mesh is not None and cp_mesh.size() > 1:
        raise ValueError("Use the GDN P2P or Ulysses wrapper when CP is active.")
    if tp_mesh is not None and tp_mesh.size() > 1:
        raise NotImplementedError("Packed local GDN with TP is not qualified.")
    if getattr(target_module, "_hp_gdn_cp_config", None) is not None:
        raise RuntimeError("A GDN execution wrapper is already installed.")
    executor_class = GatedDeltaNetUlyssesCP if backend.lower() == "ascendc" else GatedDeltaNetP2PCP
    executor = executor_class(target_module, None, backend=backend, chunk_size=chunk_size)
    original_forward = target_module.forward

    @wraps(original_forward)
    def packed_forward(*args: Any, **kwargs: Any) -> Any:
        """Dispatch only explicitly packed calls to the local segmented executor."""
        if any(
            kwargs.get(key) is not None
            for key in ("packed_sequence_info", "cu_seq_lens", "cu_seqlens", "cu_seq_lens_q", "cu_seq_lens_k")
        ):
            return executor(*args, **kwargs)
        return original_forward(*args, **kwargs)

    return _ForwardRewriteRequest(
        target_module,
        packed_forward,
        companion_attrs={"_hp_gdn_cp_config": {"mode": "packed_local", "backend": backend, "chunk_size": chunk_size}},
    )
