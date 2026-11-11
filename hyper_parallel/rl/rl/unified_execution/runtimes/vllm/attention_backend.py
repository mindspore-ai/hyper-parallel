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
"""Ascend cache ownership with the shared deterministic V4 numerical kernel.

This module is imported only by workers explicitly selecting unified V4.
Ascend retains cache allocation, writes, transfers and scheduler metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import torch
from vllm.config import CUDAGraphMode
from vllm.v1.attention.backend import AttentionCGSupport, AttentionType
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendMetadata,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import attention_transfer_window

from rl.unified_execution.ops.attention.flash_attention_npu_v4 import (
    prepare_scheduler_metadata, require_attention_provider, varlen_attention,
)


logger = logging.getLogger(__name__)


@dataclass
class UnifiedAttentionMetadata(AscendMetadata):
    """Keep views of runner-owned device buffers alive during graph replay."""

    v4_query_start_loc: torch.Tensor | None = None
    v4_seq_lens: torch.Tensor | None = None
    v4_scheduler_metadata: torch.Tensor | None = None


class UnifiedAttentionMetadataBuilder(AscendAttentionMetadataBuilder):
    """Reuse Ascend scheduling while retaining live query and KV lengths."""

    metadata_cls = UnifiedAttentionMetadata

    def __init__(self, kv_cache_spec: Any, layer_names: list[str], vllm_config: Any, device: torch.device) -> None:
        """Keep one scheduler buffer per native metadata builder for graph replay."""
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.num_query_heads = vllm_config.model_config.get_num_attention_heads(vllm_config.parallel_config)
        self.scheduler_metadata: torch.Tensor | None = None

    @classmethod
    def get_cudagraph_support(cls, vllm_config: Any, kv_cache_spec: Any) -> AttentionCGSupport:
        """Only decode graphs have fixed request slots in the Ascend runner."""
        del vllm_config, kv_cache_spec
        return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def build(
        self, common_prefix_len: int, common_attn_metadata: Any, fast_build: bool = False,
    ) -> UnifiedAttentionMetadata:
        """Bind persistent device buffers instead of copying CPU length lists.

        Prefix-cache hits and chunked prefill use the same full logical KV
        lengths. The runner refreshes these buffers before each graph replay.
        """
        if not common_attn_metadata.causal:
            raise ValueError("Unified V4 attention supports causal decoder attention only")
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        if query_start_loc.shape[0] < num_reqs + 1 or seq_lens.shape[0] < num_reqs:
            raise ValueError("Unified V4 attention requires runner device lengths for every scheduled request")
        metadata.v4_query_start_loc = query_start_loc[:num_reqs + 1]
        metadata.v4_seq_lens = seq_lens[:num_reqs]
        if num_reqs:
            scheduler = prepare_scheduler_metadata(
                max_seqlen_q=max(common_attn_metadata.num_actual_tokens, common_attn_metadata.num_input_tokens),
                max_seqlen_k=self.model_config.max_model_len,
                num_heads=self.num_query_heads, num_kv_heads=self.kv_cache_spec.num_kv_heads,
                head_dim=self.kv_cache_spec.head_size,
                cu_seqlens_q=metadata.v4_query_start_loc, seqused_k=metadata.v4_seq_lens,
            )
            if self.scheduler_metadata is None:
                self.scheduler_metadata = torch.empty_like(scheduler)
            # V4 0.5.0 metadata events cannot be captured. Native runners build
            # metadata each step before replay; refresh a stable device address.
            self.scheduler_metadata.copy_(scheduler)
            metadata.v4_scheduler_metadata = self.scheduler_metadata
        return metadata


class UnifiedAscendAttentionBackend(AscendAttentionBackend):
    """Select V4 per attention layer without replacing Ascend's global backend."""

    @staticmethod
    def get_impl_cls() -> type[UnifiedAscendAttentionImpl]:
        """Return the implementation sharing the training numerical provider."""
        return UnifiedAscendAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[UnifiedAttentionMetadataBuilder]:
        """Return the graph-safe metadata adapter."""
        return UnifiedAttentionMetadataBuilder

    @classmethod
    def supports_pcp(cls) -> bool:
        """Keep context parallelism outside the unified Qwen3 contract."""
        return False


class UnifiedAscendAttentionImpl(AscendAttentionBackendImpl):
    """Preserve native cache writes and replace only the attention computation."""

    def __init__(
        self, num_heads: int, head_size: int, scale: float, num_kv_heads: int,
        alibi_slopes: list[float] | None, sliding_window: int | None,
        kv_cache_dtype: str, logits_soft_cap: float | None, attn_type: str,
        kv_sharing_target_layer_name: str | None, sinks: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        """Reject unsupported numerical variants before invoking the provider."""
        if attn_type != AttentionType.DECODER:
            raise ValueError("Unified V4 attention supports decoder attention only")
        if alibi_slopes is not None or sliding_window is not None or logits_soft_cap not in (None, 0.0):
            raise ValueError("Unified V4 attention does not support ALiBi, sliding windows or logit soft caps")
        if sinks is not None or kv_sharing_target_layer_name is not None:
            raise ValueError("Unified V4 attention does not support attention sinks or shared layer KV")
        if kv_cache_dtype not in ("auto", "bfloat16"):
            raise ValueError("Unified V4 attention requires unquantized BF16 KV cache")
        super().__init__(
            num_heads, head_size, scale, num_kv_heads, alibi_slopes, sliding_window,
            kv_cache_dtype, logits_soft_cap, attn_type, kv_sharing_target_layer_name, sinks, **kwargs,
        )
        parallel = self.vllm_config.parallel_config
        if any(getattr(parallel, name, 1) != 1 for name in (
            "prefill_context_parallel_size", "decode_context_parallel_size",
        )):
            raise ValueError("Unified V4 attention requires context parallel sizes of one")
        if self.vllm_config.speculative_config is not None:
            raise ValueError("Unified V4 attention does not support speculative decoding")
        if self.vllm_config.model_config.runner_type != "generate":
            raise ValueError("Unified V4 attention requires the generation model runner")
        if not self.vllm_config.model_config.enforce_eager:
            if self.vllm_config.compilation_config.cudagraph_mode not in (
                CUDAGraphMode.FULL_DECODE_ONLY, CUDAGraphMode.FULL_AND_PIECEWISE,
            ):
                raise ValueError("Unified V4 attention graph execution requires FULL_DECODE_ONLY or FULL_AND_PIECEWISE")
            logger.info("Unified V4 graph mode: %s", self.vllm_config.compilation_config.cudagraph_mode)
        self.max_model_len = self.vllm_config.model_config.max_model_len
        require_attention_provider()

    @staticmethod
    def update_graph_params(
        update_stream: Any, forward_context: Any, num_tokens: int, vllm_config: Any,
        speculative_config: Any = None, draft_attn_metadatas: Any = None,
    ) -> None:
        """The builder refreshes V4's device metadata; no FIA host-attribute update is needed."""
        del update_stream, forward_context, num_tokens, vllm_config, speculative_config, draft_attn_metadatas

    def forward_impl(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...], attn_metadata: UnifiedAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Attend directly to paged history for prefill, prefix hits and decode.

        Native ``forward`` already wrote current K/V using ``slot_mapping``.
        V4's bottom-right causal mask handles an existing prefix and mixed
        query lengths without splitting requests or unpacking cached history.
        """
        del key, value, kv_cache
        if not attn_metadata.causal:
            raise ValueError("Unified V4 attention requires causal metadata")
        if self.key_cache is None or self.value_cache is None:
            raise ValueError("Unified V4 attention requires vLLM-owned paged KV cache")
        cu_query = attn_metadata.v4_query_start_loc
        seq_lens = attn_metadata.v4_seq_lens
        if cu_query is None or seq_lens is None or attn_metadata.block_tables is None:
            raise ValueError("Unified V4 attention requires device request boundaries and block tables")
        num_reqs = seq_lens.shape[0]
        if attn_metadata.block_tables.shape[0] < num_reqs:
            raise ValueError("Unified V4 attention requires one block-table row per request")
        if query.shape[0] == 0:
            return output.zero_()
        if attn_metadata.v4_scheduler_metadata is None:
            raise ValueError("Unified V4 attention requires scheduler metadata prepared before graph capture")
        key_cache, value_cache = self.key_cache, self.value_cache
        if self.use_bnsd_kv_cache:
            # V4 accepts BNBD physical strides with logical page/token/head axes.
            key_cache = key_cache.transpose(1, 2)
            value_cache = value_cache.transpose(1, 2)
        with attention_transfer_window():
            result = varlen_attention(
                query, key_cache, value_cache,
                cu_seqlens_q=cu_query, seqused_k=seq_lens,
                max_seqlen_q=query.shape[0], max_seqlen_k=self.max_model_len,
                page_table=attn_metadata.block_tables[:num_reqs], scale=self.scale,
                scheduler_metadata=attn_metadata.v4_scheduler_metadata,
            )
            output.copy_(result.reshape_as(output))
        return output
