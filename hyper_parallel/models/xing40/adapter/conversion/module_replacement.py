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
"""replacements: Xing4.0 structure replacements onto the generic modules.

:class:`Xing40MLAAttention` extends the generic high-performance ``modules.MLAAttention``
instead of reimplementing kernels; the NPU kernels live in ``functional``.

The module stays importable on CPU-only checkouts: ``modules.rms_norm`` imports the
NPU-only runtime, so it is imported lazily where the latent norms are swapped.
"""

# This adapter uses the Torch runtime, like the framework attention modules.
# pylint: disable=forbidden-backend-import

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from transformers.models.deepseek_v3.modeling_deepseek_v3 import apply_rotary_pos_emb_interleave

from hyper_parallel.components.modules.mla_attention import MLAAttention
from hyper_parallel.models.replacement import module_replacement as register_module_replacement

__all__ = ["Xing40MLAAttention"]


def fusion_attention_with_logit_maxima(module: nn.Module, query: torch.Tensor, key: torch.Tensor,
                                       value: torch.Tensor, attention_mask: torch.Tensor | None,
                                       **kwargs: Any) -> tuple[torch.Tensor, None]:
    """Run causal TND fused attention and fold the per-head logit maxima into ``module.max_logits``.

    This is the ``attention_interface`` that :class:`Xing40MLAAttention` hands to
    ``modules.MLAAttention``.

    Args:
        module: Attention module; its ``max_logits`` holds the maxima since the last QK clipping.
        query: Query states, ``[batch, heads, sequence, head_dim]``.
        key: Key states, ``[batch, heads, sequence, head_dim]``.
        value: Value states, ``[batch, heads, sequence, v_head_dim]``.
        attention_mask: Unused; every sequence attends causally.
        **kwargs: ``scaling``, the attention score scale; the other options of the
            framework attention interface are unused.

    Returns:
        The attention output, ``[batch, sequence, heads, v_head_dim]``, and no attention weights.

    Raises:
        ValueError: If the kernel's ``softmax_max`` is not the expected ``(T, heads, 8)`` shape.
    """
    # torch_npu is optional outside Ascend environments.
    import torch_npu  # pylint: disable=C0415
    from hyper_parallel.components.functional.npu_fusion_attention import (  # pylint: disable=C0415
        _prepare_fusion_attention_context,
    )

    del attention_mask
    batch, heads, sequence = query.shape[:3]
    lengths = tuple(range(sequence, batch * sequence + 1, sequence))
    context = _prepare_fusion_attention_context(module, query, key, value, None, {"actual_seq_len": lengths})
    output, softmax_max = torch_npu.npu_fusion_attention(
        context.query, context.key, context.value, heads, context.input_layout,
        pse=None, padding_mask=None, atten_mask=context.attention_mask, scale=kwargs["scaling"],
        pre_tockens=context.pre_tokens, next_tockens=context.next_tokens, keep_prob=1.0,
        inner_precise=0, sparse_mode=context.sparse_mode, actual_seq_qlen=context.query_lengths,
        actual_seq_kvlen=context.key_lengths, softmax_layout="TND")[:2]
    with torch.no_grad():
        # softmax_max is (T, N, 8) for TND input: core/shard/ops/parallel_npu_flash_attention_score.py's
        # sharding rule hard-codes this ("TND softmax outputs are 3-D (T, N, 8)"), and
        # test_parallel_npu_flash_attention_score.py asserts dim 0 is token, dim 1 is head. Check it
        # instead of silently maxing across the wrong axis if that assumption is ever wrong (MindFormers'
        # own max_logit_tracker.py exists specifically because this failure raises nothing on its own).
        if softmax_max.ndim != 3 or softmax_max.shape[1] != heads:
            raise ValueError(f"Expected (T, {heads}, 8) softmax_max for TND fusion attention, "
                             f"got shape {tuple(softmax_max.shape)}.")
        maximum = softmax_max.amax(dim=(0, 2))
        module.max_logits = maximum if module.max_logits is None else torch.maximum(module.max_logits, maximum)
    return output.view(batch, sequence, heads, -1), None


@register_module_replacement
class Xing40MLAAttention(MLAAttention):
    """Replacement of ``DeepseekV3Attention``: the generic MLA with the MindFormers FP32 RoPE.

    ``modules.MLAAttention`` fuses ``q_a_proj``/``kv_a_proj_with_mqa`` into ``linear_qkv``
    (checkpoint conversion declared through ``make_transforms``) and keeps the latent-norm
    and projection structure. This replacement swaps the latent norms for the NPU ``RMSNorm``
    (a replacement plan cannot also select these children of the replaced attention),
    rotates in FP32 and rounds once as MindFormers, and calls the attention kernel through
    :func:`fusion_attention_with_logit_maxima`, which folds in the per-head logit maxima QK
    clipping consumes.
    """

    def __init__(self, *, module: nn.Module, module_fqn: str = "",
                 context: Mapping[str, Any] | None = None) -> None:
        """Build the replacement from a ``DeepseekV3Attention``.

        Args:
            module: ``DeepseekV3Attention`` instance being replaced.
            module_fqn: Fully qualified name of ``module``.
            context: Replacement context of the build pipeline.
        """
        super().__init__(module=module, module_fqn=module_fqn, context=context,
                         attention_interface=fusion_attention_with_logit_maxima)
        # Lazy: modules.rms_norm imports the NPU-only runtime.
        from hyper_parallel.components.modules.rms_norm import RMSNorm  # pylint: disable=C0415

        self.q_a_layernorm = RMSNorm(module=self.q_a_layernorm)
        self.kv_a_layernorm = RMSNorm(module=self.kv_a_layernorm)
        if not self.linear_qkv.weight.is_meta:
            with torch.no_grad():
                self.linear_qkv.weight.copy_(torch.cat((module.q_a_proj.weight, module.kv_a_proj_with_mqa.weight)))
        # Per-head logit maxima since the last QK clipping, folded in by the attention interface.
        self.max_logits: torch.Tensor | None = None

    def _project_attention_inputs(self, hidden_states: torch.Tensor,
                                  position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
                                  past_key_values: Any | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project hidden states to query, key and value, rotating in FP32 and rounding once.

        Args:
            hidden_states: Normalized layer input, ``[batch, sequence, hidden]``.
            position_embeddings: FP32 RoPE ``(cos, sin)`` tables.
            past_key_values: Must be None; training uses no KV cache.

        Returns:
            Query, key and value states, ``[batch, heads, sequence, head_dim]``.

        Raises:
            ValueError: If the RoPE tables are missing or a KV cache is given.
        """
        if position_embeddings is None or past_key_values is not None:
            raise ValueError("Xing4.0 MLA requires RoPE tables and no KV cache")
        latents = self._project_latents(hidden_states)
        cos, sin = position_embeddings
        query_rope, key_rope = apply_rotary_pos_emb_interleave(
            latents.query_rope.float(), latents.key_rope.float(), cos, sin)
        key_value = self.kv_b_proj(latents.kv_nope).view(
            latents.batch_size, latents.sequence_length, self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim)
        key_nope, value = key_value.transpose(1, 2).split((self.qk_nope_head_dim, self.v_head_dim), dim=-1)
        query_pass = latents.query_pass.transpose(1, 2)
        query = torch.cat((query_pass, query_rope.to(query_pass.dtype)), dim=-1)
        key = torch.cat((key_nope, key_rope.to(key_nope.dtype).expand(-1, self.num_heads, -1, -1)), dim=-1)
        return query, key, value
