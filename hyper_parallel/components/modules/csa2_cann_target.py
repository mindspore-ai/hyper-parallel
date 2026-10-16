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
"""Optional CANN CSA2 replacement selected through ``replace_module._target_``."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.functional.aux_loss import aux_loss_auto_scale
from hyper_parallel.components.modules import csa2_cann
from hyper_parallel.components.modules.shared_compressed_dsa_attention import (
    SharedCompressedDSAAttention,
    SharedCompressedDSAIndexer,
)
from hyper_parallel.core.activation_memory.recompute_state import is_recomputing
from hyper_parallel.models.replacement import module_replacement


class _CANNSharedCompressedDSAIndexer(SharedCompressedDSAIndexer):
    """Use CANN Top-K for full-key selection while retaining candidate modes."""

    def _select_tokens(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            merge_weight: torch.Tensor,
            *,
            candidate_blocks: torch.Tensor | None,
            query_offset: int,
            minimum_key_indices: torch.Tensor | None,
            reduce_sum: Callable[[torch.Tensor], torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Select full-key positions with CANN when its mask matches CSA2."""
        if (
                not self.is_candidate_source
                and not self.uses_candidates
                and candidate_blocks is None
                and minimum_key_indices is None
                and reduce_sum is None
                and csa2_cann.supports_cann_indexer_forward(
                    query, key, merge_weight,
                    compress_ratio=self.compress_ratio,
                    sparse_count=self.index_topk,
                    query_offset=query_offset,
                )
        ):
            return csa2_cann.cann_indexer_forward(
                query, key, merge_weight,
                compress_ratio=self.compress_ratio, query_offset=query_offset,
            ), None
        return super()._select_tokens(
            query, key, merge_weight, candidate_blocks=candidate_blocks,
            query_offset=query_offset, minimum_key_indices=minimum_key_indices,
            reduce_sum=reduce_sum,
        )


@module_replacement
class CANNSharedCompressedDSAAttention(SharedCompressedDSAAttention):
    """Use fused attention and Indexer gradients for validated CSA2 shapes.

    The parent owns parameters, CP communication and the fallback. This target
    also uses the CANN full-key Indexer where its compressed mask is supported.
    """

    def __init__(
            self,
            module: torch.nn.Module,
            module_fqn: str = "",
            context: Mapping[str, Any] | None = None,
    ) -> None:
        """Preserve parameter names while selecting the target-only Indexer."""
        super().__init__(module, module_fqn, context)
        source_indexer = getattr(module, "indexer", None)
        if source_indexer is not None and not isinstance(source_indexer, SharedCompressedDSAIndexer):
            self.indexer = _CANNSharedCompressedDSAIndexer(source_indexer)

    def _apply_cann_indexer_loss(
            self,
            indexer_output: Any | None,
            query: torch.Tensor,
            compressed_kv: torch.Tensor | None,
            tp_context: Any | None,
            query_offset: int,
            packed_sequence: Any | None,
    ) -> torch.Tensor:
        """Use the fused KL gradient when its geometry preserves the parent loss."""
        if (
                packed_sequence is None
                and tp_context is None
                and indexer_output is not None
                and self.training
                and self.indexer.loss_coeff
                and compressed_kv is not None
                and csa2_cann.supports_cann_indexer_kl(
                    indexer_output.index_query,
                    indexer_output.index_key,
                    indexer_output.merge_weight,
                    indexer_output.topk_indices,
                    compress_ratio=self.compress_ratio,
                    query_offset=query_offset,
                )
        ):
            indexer_loss = csa2_cann.cann_indexer_kl_loss(
                indexer_output.index_query,
                indexer_output.index_key,
                indexer_output.merge_weight,
                query,
                compressed_kv,
                indexer_output.topk_indices,
                self.sinks,
                attention_scale=self.scaling,
                loss_coeff=self.indexer.loss_coeff,
                query_chunk_size=self.indexer.query_chunk_size,
                compress_ratio=self.compress_ratio,
                query_offset=query_offset,
            )
            return aux_loss_auto_scale(query, indexer_loss)
        return self._apply_indexer_loss(indexer_output, query, compressed_kv, tp_context)

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]],
            position_ids: torch.Tensor,
            attention_mask: torch.Tensor | None,
            past_key_values: Any | None = None,
            **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run local queries against CP-gathered KV through the CANN kernels."""
        (
            shared_state,
            packed_sequence,
            cp_context,
            tp_context,
            batch_size,
            sequence_length,
        ) = self._validate_forward_inputs(hidden_states, attention_mask, past_key_values, kwargs)
        query_offset = 0
        global_sequence_length = sequence_length
        if cp_context is not None:
            query_offset, global_sequence_length = self._validate_cp_inputs(
                cp_context, position_ids, sequence_length, self.compress_ratio, self.is_kv_source,
            )
        segment_starts, minimum_key_indices = self._resolve_packed_geometry(
            packed_sequence, batch_size, sequence_length, query_offset,
            global_sequence_length, hidden_states.device,
        )

        rope_type = "compress" if self.compress_ratio else "main"
        cos, sin = position_embeddings[rope_type]
        key_value, raw_kv_handle = self._project_raw_kv(hidden_states, cos, sin, cp_context)
        query_residual, query = self._project_query(hidden_states, cos, sin)
        latent, compressed_handle = self._compute_compressed_kv(
            hidden_states, position_embeddings, shared_state, cp_context,
        )
        indexer_output = self._run_indexer(
            hidden_states, query_residual, latent, position_embeddings,
            shared_state, cp_context, tp_context, query_offset, minimum_key_indices,
        )

        if raw_kv_handle is not None:
            key_value = raw_kv_handle.wait()
        if compressed_handle is not None:
            compressed = compressed_handle.wait()
            if not is_recomputing():
                shared_state.publish_compressed_kv(self.layer_idx, compressed)
        if key_value is None:
            raise RuntimeError("raw KV all-gather did not produce a tensor")

        compressed_kv = None
        topk_indices = None
        if self.compress_ratio:
            compressed_kv = shared_state.require_compressed_kv(self.kv_source_layer_idx, self.layer_idx)
            topk_indices = shared_state.require_topk_indices(self.index_source_layer_idx, self.layer_idx)
        use_cann_attention = (
            packed_sequence is None
            and tp_context is None
            and compressed_kv is not None
            and topk_indices is not None
            and csa2_cann.supports_cann_attention(
                query, key_value, compressed_kv, topk_indices,
                compress_ratio=self.compress_ratio,
                sliding_window=self.sliding_window,
                query_offset=query_offset,
            )
        )
        if use_cann_attention:
            query = self._apply_cann_indexer_loss(
                indexer_output, query, compressed_kv, tp_context, query_offset, packed_sequence,
            )
            attention_output = csa2_cann.cann_sparse_attention(
                query, key_value, compressed_kv, topk_indices, self.sinks,
                self.scaling, query_offset=query_offset,
            )
        else:
            combined_key_value, sparse_indices, compressed_kv = self._build_sparse_indices(
                shared_state, key_value, batch_size, sequence_length,
                hidden_states.device, query_offset, global_sequence_length, segment_starts,
            )
            query = self._apply_cann_indexer_loss(
                indexer_output, query, compressed_kv, tp_context, query_offset, packed_sequence,
            )
            attention_output = self._run_sparse_attention(
                query, combined_key_value, sparse_indices, hidden_states.device,
            )
        output = self._project_output(attention_output, cos, sin, batch_size, sequence_length)
        return output, None


__all__ = ["CANNSharedCompressedDSAAttention"]
