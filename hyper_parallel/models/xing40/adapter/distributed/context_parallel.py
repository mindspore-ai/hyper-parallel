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
"""Xing40 context-parallel adapters. Opt-in; no effect on non-CP attention or router."""
from __future__ import annotations

from typing import Any, Literal
# Native Torch is required by this model adapter; the legacy backend-import rule predates that migration.
import torch  # pylint: disable=forbidden-backend-import
from transformers.models.deepseek_v3.modeling_deepseek_v3 import apply_rotary_pos_emb_interleave
from hyper_parallel.components.functional.npu_fusion_attention import (
    _prepare_fusion_attention_context, PACKED_SEQUENCE_ARGUMENTS,
)
from hyper_parallel.distributed.context_parallel.mla_context_parallel import MLACPStrategy
from hyper_parallel.distributed.recipe_spec import inner_wrapper
from hyper_parallel.models.deepseek_v3.adapter.distributed.context_parallel import mla_cp_wrapper
from hyper_parallel.models.xing40.adapter.conversion.module_replacement import Xing40MLAAttention

def xing40_rotary(tensor: torch.Tensor, positions: tuple[torch.Tensor, torch.Tensor] | None, *,
                   interleaved: bool = False) -> torch.Tensor:
    """Apply HF interleaved RoPE in FP32 on BSND and round once.

    Args:
        tensor: Local batch, sequence, head, rotary-width tensor.
        positions: Local cosine and sine embeddings.
        interleaved: Must be true for the Xing40 interleaved layout.

    Returns:
        Rotated tensor in the input dtype.
    """
    if positions is None:
        raise ValueError("Xing40 CP requires RoPE position embeddings")
    if not interleaved:
        raise ValueError('Xing40 CP requires interleaved RoPE')
    cos, sin = positions
    rotated, _ = apply_rotary_pos_emb_interleave(tensor.float(), tensor.float(), cos, sin, unsqueeze_dim=2)
    return rotated.to(tensor.dtype)

def xing40_cp_attention(module: torch.nn.Module, query: torch.Tensor, key: torch.Tensor,
                        value: torch.Tensor, attention_mask: torch.Tensor | None,
                        **kwargs: Any) -> tuple[torch.Tensor, None]:
    """Run implicit-causal TND attention and record per-head softmax maxima.

    Args:
        module: Attention module receiving maxima.
        query: Local query tensor in BNSD layout.
        key: Gathered causal KV prefix in BNSD layout.
        value: Values aligned with key.
        attention_mask: Must be None for this first CP path.
        **kwargs: Explicit query/KV boundaries, causal flag and attention scale; from the MLA CP
            runtime also ``mla_head_offset`` and ``mla_local_heads``, the position of these heads
            among the TP-local heads.

    Returns:
        BSND output and None for attention weights.
    """
    # Rotary and layout contracts also run on CPU; load the optional Ascend kernel only when called.
    import torch_npu  # pylint: disable=import-outside-toplevel
    options = dict(kwargs)
    head_offset = options.pop('mla_head_offset', 0)
    local_heads = options.pop('mla_local_heads', query.shape[1])
    if not any(options.get(k) is not None for k in PACKED_SEQUENCE_ARGUMENTS):
        batch, _, qlen, _ = query.shape
        if qlen != key.shape[2]:
            raise ValueError('unequal CP Q/K lengths require explicit boundaries')
        options['actual_seq_len'] = tuple(range(qlen, batch * qlen + 1, qlen))
    if attention_mask is not None or not options.get('is_causal', getattr(module, 'is_causal', True)):
        raise ValueError('Xing40 CP supports implicit causal attention only')
    context = _prepare_fusion_attention_context(module, query, key, value, attention_mask, options)
    if context.valid_rows is not None:
        raise ValueError('Xing40 CP does not support masked rows')
    if context.input_layout != 'TND':
        raise ValueError('Xing40 CP expects packed TND')
    heads = query.shape[1]
    output, maximum = torch_npu.npu_fusion_attention(
        context.query.contiguous(), context.key.contiguous(), context.value.contiguous(), heads, context.input_layout,
        pse=None, padding_mask=None, atten_mask=context.attention_mask, scale=options['scaling'],
        pre_tockens=context.pre_tokens, next_tockens=context.next_tokens, keep_prob=1.0,
        inner_precise=0, sparse_mode=context.sparse_mode, actual_seq_qlen=context.query_lengths,
        actual_seq_kvlen=context.key_lengths, softmax_layout='TND')[:2]
    if maximum.ndim != 3 or maximum.shape[1] != heads:
        raise ValueError('unexpected Xing40 CP per-head softmax maxima shape')
    with torch.no_grad():
        local_max = maximum.amax(dim=(0, 2))
        if local_heads != heads:
            # Ulysses splits the TP-local heads across CP ranks; QK clipping's MAX reduction fills in the rest.
            placed = local_max.new_full((local_heads,), float('-inf'))
            placed[head_offset:head_offset + heads] = local_max
            local_max = placed
        module.max_logits = local_max if module.max_logits is None else torch.maximum(module.max_logits, local_max)
    return output.reshape(context.batch_size, context.query_length, heads, -1), None

@inner_wrapper
def xing40_mla_cp_wrapper(target_module: torch.nn.Module, mesh: Any, tp_mesh: Any,
                            cp_mesh: Any, ep_mesh: Any, *, strategy: MLACPStrategy = 'expanded_ulysses',
                            overlap: bool = False, ulysses_degree: int | None = 1,
                            kv_exchange: Literal['all_gather', 'all_to_all_v'] = 'all_gather') -> list:
    """Install the framework MLA CP runtime with the Xing40 RoPE and attention.

    The default is the KV AllGather schedule, which the framework plan restricts to TP1; with TP
    above one, set ``ulysses_degree`` to the CP size.

    Args:
        target_module: Xing40 MLA module being wrapped.
        mesh: Full mesh, passed to the generic wrapper.
        tp_mesh: Tensor-parallel mesh or None.
        cp_mesh: Context-parallel mesh.
        ep_mesh: Expert-parallel mesh or None.
        strategy: Expanded QKV Ulysses or expanded Q with latent KV AllGather.
        overlap: Overlap forward input communication with independent MLA projections.
        ulysses_degree: Heads split over this many CP ranks; one selects KV AllGather and None the CP size.
        kv_exchange: Full KV collection or selective causal-prefix distribution.

    Returns:
        Forward rewrite requests from the generic MLA wrapper.
    """
    if not isinstance(target_module, Xing40MLAAttention):
        raise TypeError('Xing40 MLA replacement required')
    if cp_mesh is None:
        raise ValueError('Xing40 CP adapter requires a CP mesh')
    return mla_cp_wrapper(target_module, mesh, tp_mesh, cp_mesh, ep_mesh, strategy=strategy, overlap=overlap,
                          ulysses_degree=ulysses_degree, kv_exchange=kv_exchange,
                          rotary_fn=xing40_rotary, attention_interface=xing40_cp_attention)
