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
"""Xing40 CP2 adapters. Opt-in; no effect on non-CP attention or router."""
from __future__ import annotations

from typing import Any
# Native Torch is required by this model adapter; the legacy backend-import rule predates that migration.
import torch  # pylint: disable=forbidden-backend-import
from transformers.models.deepseek_v3.modeling_deepseek_v3 import apply_rotary_pos_emb_interleave
from hyper_parallel.components.functional.npu_fusion_attention import (
    _prepare_fusion_attention_context, PACKED_SEQUENCE_ARGUMENTS,
)
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
        **kwargs: Explicit query/KV boundaries, causal flag and attention scale.

    Returns:
        BSND output and None for attention weights.
    """
    # Rotary and layout contracts also run on CPU; load the optional Ascend kernel only when called.
    import torch_npu  # pylint: disable=import-outside-toplevel
    options = dict(kwargs)
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
        module.max_logits = local_max if module.max_logits is None else torch.maximum(module.max_logits, local_max)
    return output.reshape(context.batch_size, context.query_length, heads, -1), None

@inner_wrapper
def xing40_mla_cp_wrapper(target_module: torch.nn.Module, mesh: Any, tp_mesh: Any,
                            cp_mesh: Any, ep_mesh: Any) -> list:
    """Install the explicit CP2, TP1, EP1 attention path.

    Args:
        target_module: Xing40 MLA module being wrapped.
        mesh: Full mesh, passed to the generic wrapper.
        tp_mesh: Tensor-parallel mesh or None.
        cp_mesh: Two-rank context-parallel mesh.
        ep_mesh: Expert-parallel mesh or None.

    Returns:
        Forward rewrite requests from the generic MLA wrapper.
    """
    if not isinstance(target_module, Xing40MLAAttention):
        raise TypeError('Xing40 MLA replacement required')
    if (cp_mesh is None or cp_mesh.size() != 2 or (tp_mesh is not None and tp_mesh.size() != 1)
            or (ep_mesh is not None and ep_mesh.size() != 1)):
        raise ValueError('Xing40 initial CP adapter requires CP2, TP1 and EP1')
    return mla_cp_wrapper(target_module, mesh, tp_mesh, cp_mesh, ep_mesh,
                          strategy='expanded_ulysses', ulysses_degree=1, kv_exchange='all_gather',
                          rotary_fn=xing40_rotary, attention_interface=xing40_cp_attention)
