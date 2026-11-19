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
"""Experimental DeepSeek-V3 HF/checkpoint contract for staged MLA."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from hyper_parallel.components.modules.head_chunk_mla import HeadChunkMLA
from hyper_parallel.components.modules.mla_attention import MLAAttention
from hyper_parallel.models.replacement import module_replacement


@module_replacement
class DeepseekV3HeadChunkMLA(HeadChunkMLA):
    """Adapt HF latent weights and loss-only kwargs without modifying Trainer.

    Use this opt-in replacement before decoder-level FSDP. It preserves native
    MLA checkpoint conversion and rejects unsupported attention operands in the
    underlying component. It is not a TP or child-projection FSDP adapter.
    """

    def __init__(
        self, *, module: nn.Module, module_fqn: str, context: Mapping[str, Any],
        head_chunk_size: int = 32, offload_saved_stats: bool = False,
        prefetch_slots: int = 2, overlap_backward: bool = False,
        attention_backend: str = "npu",
    ) -> None:
        """Retain native up-projection parameters and the fused-down weight transform."""
        if getattr(getattr(module, "config", None), "model_type", None) != "deepseek_v3":
            raise ValueError("DeepseekV3HeadChunkMLA requires a DeepSeek-V3 source module")
        native = module if isinstance(module, MLAAttention) else MLAAttention(
            module=module, module_fqn=module_fqn, context=context,
        )
        transforms = native.make_transforms()
        if native is not module and not module.q_a_proj.weight.is_meta:
            # Direct construction must preserve already loaded weights too;
            # meta/from-pretrained loading uses the same declared converter.
            with torch.no_grad():
                native.linear_qkv.weight.copy_(torch.cat((module.q_a_proj.weight, module.kv_a_proj_with_mqa.weight)))
        super().__init__(module=native, head_chunk_size=head_chunk_size,
                         offload_saved_stats=offload_saved_stats, prefetch_slots=prefetch_slots,
                         overlap_backward=overlap_backward, attention_backend=attention_backend)
        self._checkpoint_transforms = transforms

    def make_transforms(self) -> list:
        """Return reversible HF q_a/kv_a to native linear_qkv conversion rules."""
        return list(self._checkpoint_transforms)

    def forward(
        self, hidden_states: torch.Tensor, position_embeddings: tuple | None = None,
        attention_mask: torch.Tensor | None = None, past_key_values: Any = None,
        actual_seq_len: Any = None, position_ids: torch.Tensor | None = None,
        use_cache: bool = False, shift_labels: torch.Tensor | None = None,
        num_items_in_batch: Any = None, **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        """Translate only redundant HF metadata; retain all real mask/cache checks.

        Args:
            hidden_states: Local sequence shard in BSH layout.
            position_embeddings: Already prepared local RoPE cosine and sine.
            attention_mask: Must be absent; packed documents use cumulative ends.
            past_key_values: Unsupported training cache.
            actual_seq_len: Global cumulative packed document ends, or None.
            position_ids: Redundant only when position_embeddings is supplied.
            use_cache: Must be false.
            shift_labels: Loss-only targets forwarded by HF, unused by attention.
            num_items_in_batch: Loss denominator metadata, unused by attention.
            **kwargs: Passed to the component for strict validation.

        Returns:
            Local projected output and None for attention weights.
        """
        del shift_labels, num_items_in_batch
        if use_cache:
            raise ValueError("Head-chunk MLA is a full-sequence training path, not a KV cache")
        if position_ids is not None and position_embeddings is None:
            raise ValueError("position_ids requires precomputed position_embeddings")
        return super().forward(hidden_states, position_embeddings=position_embeddings,
                               attention_mask=attention_mask, past_key_values=past_key_values,
                               actual_seq_len=actual_seq_len, **kwargs)
