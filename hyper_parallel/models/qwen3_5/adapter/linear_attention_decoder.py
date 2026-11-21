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
"""Forward document metadata through Qwen linear-attention decoder calls."""
from __future__ import annotations

from functools import wraps
from typing import Any
from torch import Tensor, nn

from hyper_parallel.components.modules.module_family import get_family_build_info
from hyper_parallel.distributed import inner_wrapper
from hyper_parallel.distributed._builder.forward_rewriter import _ForwardRewriteRequest


@inner_wrapper
def qwen3_5_linear_attention_decoder_wrapper(
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

    Args:
        target_module: Source-model module receiving the compatibility adapter.
        mesh: Root device mesh supplied by the legacy adapter interface.
        tp_mesh: Tensor-parallel mesh; linear-attention CP requires compatible ownership.
        cp_mesh: Sequence-ordered CP device mesh.
        ep_mesh: Expert-parallel mesh supplied by the legacy adapter interface.
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
        """Keep metadata local to this invocation, including checkpoint replay.

        Args:
            hidden_states: Local hidden activations in BSH or flat TH layout.
            position_embeddings: Source decoder positional embeddings forwarded unchanged.
            attention_mask: Optional source-model mask; packed boundaries carry document resets.
            position_ids: Source decoder position indices forwarded unchanged.
            past_key_values: Source-model inference cache; training-family execution requires None.
        """
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
        family = get_family_build_info(target_module.linear_attn)
        if family is None or family.resolved.family.name != "gated_delta_net":
            raise RuntimeError(
                "Select the GDN module family before installing packed decoder metadata forwarding.")
        residual = hidden_states
        hidden_states = target_module.input_layernorm(hidden_states)
        hidden_states = target_module.linear_attn(
            hidden_states=hidden_states, cache_params=None, attention_mask=attention_mask, **kwargs
        )
        hidden_states = residual + hidden_states
        return hidden_states + target_module.mlp(target_module.post_attention_layernorm(hidden_states))

    return _ForwardRewriteRequest(target_module, packed_forward)
