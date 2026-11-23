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
"""Complete local CSA2 math and shared forward stages."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from hyper_parallel.components.functional.aux_loss import aux_loss_auto_scale
from hyper_parallel.components.functional.sparse_flash_mla import (
    supports_fused_attention_inputs, sparse_flash_mla,
)
from hyper_parallel.components.modules.common.rms_norm import RMSNorm
from hyper_parallel.core.activation_memory.recompute_state import is_recomputing
from .compressor import CSA2Compressor
from .document_layout import SharedCompressedPackedSequence
from .indexer import CSA2Indexer
from .indexer_kl_loss import shared_compressed_indexer_kl_loss


def _rotate_half(tensor: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent real/imaginary pairs for V4 interleaved RoPE."""
    first, second = tensor.unflatten(-1, (-1, 2)).unbind(-1)
    return torch.stack((-second, first), dim=-1).flatten(-2)


def _apply_v41_rope(
        tensor: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        unsqueeze_dim: int = 1,
) -> torch.Tensor:
    """Apply V4 interleaved RoPE to the trailing rotary channels."""
    cos = cos.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    sin = sin.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    rope_dim = cos.shape[-1]
    pass_through, rotary = tensor.split((tensor.shape[-1] - rope_dim, rope_dim), dim=-1)
    rotary = ((rotary.float() * cos) + (_rotate_half(rotary).float() * sin)).to(tensor.dtype)
    return torch.cat((pass_through, rotary), dim=-1) if pass_through.shape[-1] else rotary


@dataclass
class SharedCompressedAttentionState:
    """Autograd-bearing CSA2 state scoped to one training forward.

    State is addressed by the layer that produced it instead of using one
    mutable "latest value" slot. This matters for the released V4.1 topology,
    which has several Full/Reindex groups, and makes a consumer's dependency
    explicit when activation recomputation revisits modules out of forward
    order.

    Publication returns False during replay without replacing an entry.
    Validation observes actual publications through that result.

    Nothing drops an entry early. The state travels into every layer as one of
    the checkpoint region's inputs, so the graph keeps it alive from the forward
    pass to the end of backward, and a layer whose forward is recomputed reads
    its entries a second time there. A published entry therefore has to stay
    readable until backward is done with every consumer, which is also why the
    publishers skip their write while recomputing.
    """

    compressed_kv_by_source: dict[int, torch.Tensor] = field(default_factory=dict)
    index_key_by_source: dict[int, torch.Tensor] = field(default_factory=dict)
    topk_indices_by_source: dict[int, torch.Tensor] = field(default_factory=dict)
    candidate_blocks_by_source: dict[int, torch.Tensor] = field(default_factory=dict)
    compressed_kv_offsets: dict[int, int] = field(default_factory=dict)

    @staticmethod
    def _require(
            values: dict[int, torch.Tensor],
            source_layer: int | None,
            value_name: str,
            consumer_layer: int,
    ) -> torch.Tensor:
        """Return one published tensor or report the broken layer dependency."""
        if source_layer is None or source_layer not in values:
            raise RuntimeError(
                f"layer {consumer_layer} requires {value_name} from source "
                f"layer {source_layer}, but that source has not run in this forward"
            )
        return values[source_layer]

    def publish_compressed_kv(self, source_layer: int, value: torch.Tensor, key_offset: int = 0) -> bool:
        """Publish compressed K=V and its global origin, preserving the autograd graph."""
        if is_recomputing():
            return False
        self.compressed_kv_by_source[source_layer] = value
        self.compressed_kv_offsets[source_layer] = key_offset
        return True

    def require_compressed_kv(self, source_layer: int | None, consumer_layer: int) -> torch.Tensor:
        """Read compressed K=V from the consumer's configured Full layer."""
        return self._require(
            self.compressed_kv_by_source,
            source_layer,
            "compressed K=V",
            consumer_layer,
        )

    def publish_index_key(self, source_layer: int, value: torch.Tensor) -> bool:
        """Publish the Full Indexer's shared key tensor."""
        if is_recomputing():
            return False
        self.index_key_by_source[source_layer] = value
        return True

    def require_index_key(self, source_layer: int | None, consumer_layer: int) -> torch.Tensor:
        """Read the Indexer key associated with a compressed-KV source."""
        return self._require(
            self.index_key_by_source,
            source_layer,
            "Indexer key",
            consumer_layer,
        )

    def publish_topk_indices(self, source_layer: int, value: torch.Tensor) -> bool:
        """Publish one Full/Reindex layer's token selection."""
        if is_recomputing():
            return False
        self.topk_indices_by_source[source_layer] = value
        return True

    def require_topk_indices(self, source_layer: int | None, consumer_layer: int) -> torch.Tensor:
        """Read the latest configured Full/Reindex selection."""
        return self._require(
            self.topk_indices_by_source,
            source_layer,
            "Top-K indices",
            consumer_layer,
        )

    def publish_candidate_blocks(self, source_layer: int, value: torch.Tensor) -> bool:
        """Publish hierarchical candidate blocks from the configured source."""
        if is_recomputing():
            return False
        self.candidate_blocks_by_source[source_layer] = value
        return True

    def require_candidate_blocks(
            self,
            source_layer: int | None,
            consumer_layer: int,
    ) -> torch.Tensor:
        """Read hierarchical candidates for a later Reindex layer."""
        return self._require(
            self.candidate_blocks_by_source,
            source_layer,
            "candidate blocks",
            consumer_layer,
        )


@dataclass(frozen=True)
class CSA2DocumentInputs:
    """Ready document coordinates shared by reference and fused execution."""

    packed: SharedCompressedPackedSequence | None
    query_offset: int
    global_length: int
    segment_starts: torch.Tensor | None
    minimum_key_indices: torch.Tensor | None
    query_segments: tuple | None
    causal_upper: torch.Tensor | None
    raw_document_starts: tuple
    global_key_length: int


@dataclass
class CSA2AttentionInputs:
    """Ready operands at the shared attention consumer boundary."""

    query: torch.Tensor
    key_value: torch.Tensor
    indexer_output: Any = None
    raw_key_offset: int = 0


def build_sliding_window_indices(
        batch_size: int,
        sequence_length: int,
        window_size: int,
        device: torch.device,
        *,
        query_offset: int = 0,
        key_length: int | None = None,
) -> torch.Tensor:
    """Build fixed-width causal sliding-window indices."""
    if key_length is None:
        key_length = sequence_length
    width = min(key_length, window_size)
    query = torch.arange(
        query_offset,
        query_offset + sequence_length,
        device=device,
    ).unsqueeze(-1)
    offsets = torch.arange(width - 1, -1, -1, device=device)
    indices = query - offsets
    indices = indices.masked_fill(indices < 0, -1)
    return indices.unsqueeze(0).expand(batch_size, -1, -1)


def reference_sparse_attention(
        query: torch.Tensor,
        key_value: torch.Tensor,
        sparse_indices: torch.Tensor,
        sinks: torch.Tensor,
        scale: float,
) -> torch.Tensor:
    """Compute sink attention with eager Torch operations on any device.

    Args:
        query: Main queries [batch, heads, local queries, head dimension].
        key_value: Shared K=V bank [batch, 1, keys, head dimension].
        sparse_indices: Visible bank IDs [batch, local queries, selections];
            negative entries are padding and repeated IDs count only once.
        sinks: Learnable scalar sink logits [heads], with zero-valued output.
        scale: Main attention logit scale.

    Returns:
        Attention values [batch, local queries, heads, head dimension].
        Softmax uses FP32; matrix multiplications retain the input precision.

    Note:
        This reference materializes a dense query/key mask and scores. It is
        intended for explicit unfused comparisons, not long-sequence efficiency.
    """
    batch_size, num_heads, sequence_length, _ = query.shape
    key_length = key_value.shape[2]
    valid = sparse_indices >= 0
    safe_indices = sparse_indices.clamp_min(0).long()
    allowed_counts = torch.zeros(
        batch_size,
        sequence_length,
        key_length,
        dtype=torch.int32,
        device=query.device,
    )
    allowed_counts.scatter_add_(2, safe_indices, valid.to(torch.int32))
    allowed = allowed_counts > 0
    logits = torch.matmul(query, key_value.transpose(-1, -2)) * scale
    logits = logits.masked_fill(~allowed.unsqueeze(1), float("-inf"))
    sink_logits = sinks.view(1, num_heads, 1, 1).expand(batch_size, -1, sequence_length, -1)
    probabilities = torch.cat((logits, sink_logits), dim=-1).softmax(dim=-1, dtype=torch.float32)
    return torch.matmul(probabilities[..., :-1].to(key_value.dtype), key_value).transpose(1, 2)


def _build_parameters(config: Any, layer_idx: int) -> nn.Module:
    """Create the canonical V4.1 parameter tree without importing the model."""
    # Transformers provides V4 parameter classes; CSA2 owns the V4.1 math below.
    # Preserve their construction order so seeded initialization stays compatible.
    # pylint: disable-next=import-outside-toplevel
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Attention

    module = DeepseekV4Attention(config, layer_idx)
    module.q_a_norm = RMSNorm(module=module.q_a_norm)
    module.kv_norm = RMSNorm(module=module.kv_norm)
    # DeepSeek-V4 applies this unweighted RMSNorm after q_b_proj. V4.1
    # explicitly removes it: q_norm remains between wq_a and wq_b, while
    # the projected query heads go directly into RoPE.
    del module.q_b_norm
    module.compress_ratio = int(config.v41_compress_ratios[layer_idx])
    module.is_kv_source = layer_idx in config.v41_kv_source_layer_ids
    module.is_index_source = layer_idx in config.v41_index_source_layer_ids
    kv_sources = [source for source in config.v41_kv_source_layer_ids if source <= layer_idx]
    index_sources = [source for source in config.v41_index_source_layer_ids if source <= layer_idx]
    module.kv_source_layer_idx = max(kv_sources, default=None)
    module.index_source_layer_idx = max(index_sources, default=None)
    candidate_source = int(getattr(config, "v41_candidate_source_layer_id", -1))
    module.candidate_source_layer_idx = (
        candidate_source if 0 <= candidate_source <= layer_idx else None
    )
    if module.is_kv_source:
        module.compressor = CSA2Compressor(config, module.compress_ratio, apply_rope=_apply_v41_rope)
    if module.is_index_source:
        module.indexer = CSA2Indexer(
            config=config, compress_ratio=module.compress_ratio, layer_idx=layer_idx, apply_rope=_apply_v41_rope,
        )

    return module


class CompressedSparseAttention2Module(nn.Module):
    """Complete CSA2 algorithm with local preparation and identity exchange stages."""

    def __init__(
            self,
            module: nn.Module | None = None,
            *,
            config: Any = None,
            layer_idx: int = 0,
            replace_indexer: bool = True,
            use_fused_ops: bool = False,
    ) -> None:
        """Create from config or adopt a source module without replacing its parameters."""
        if (module is None) == (config is None):
            raise ValueError("CSA2 requires exactly one of config or module")
        if not isinstance(use_fused_ops, bool):
            raise ValueError("use_fused_ops must be a bool")
        if module is None:
            module = _build_parameters(config, layer_idx)
        super().__init__()
        for name, child in module._modules.items():  # pylint: disable=protected-access
            self.add_module(name, child)
        for name, parameter in module._parameters.items():  # pylint: disable=protected-access
            self.register_parameter(name, parameter)
        for name, buffer in module._buffers.items():  # pylint: disable=protected-access
            self.register_buffer(name, buffer, persistent=name not in module._non_persistent_buffers_set)
        indexer = getattr(self, "indexer", None)
        if (
                replace_indexer
                and indexer is not None
                and not isinstance(indexer, CSA2Indexer)
        ):
            self.indexer = CSA2Indexer(indexer, apply_rope=_apply_v41_rope)
        self.config = module.config
        self.layer_idx = module.layer_idx
        self.num_heads = module.num_heads
        self.num_key_value_groups = module.num_key_value_groups
        self.num_groups = module.config.o_groups
        self.compress_ratio = module.compress_ratio
        self.is_kv_source = module.is_kv_source
        self.is_index_source = module.is_index_source
        self.kv_source_layer_idx = module.kv_source_layer_idx
        self.index_source_layer_idx = module.index_source_layer_idx
        self.candidate_source_layer_idx = module.candidate_source_layer_idx
        self.head_dim = module.head_dim
        self.rope_head_dim = module.config.qk_rope_head_dim
        self.sliding_window = module.sliding_window
        self.scaling = module.scaling
        self.use_fused_ops = use_fused_ops
        if self.is_index_source:
            self.indexer.use_fused = use_fused_ops
        # Children are adopted by identity; preserve their individual training modes.
        self.training = module.training

    def head_reducer(self) -> Callable | None:
        """Keep local head sums; parallel members provide the complete-head reduction."""
        return None

    def query_partition(self, sequence_length: int, position_ids: torch.Tensor) -> tuple[int, int]:
        """Return the complete local sequence coordinates."""
        del position_ids
        return 0, sequence_length

    def _project_raw_kv(self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Project and rotate the local raw K=V rows."""
        key_value = self.kv_norm(self.kv_proj(hidden_states)).unsqueeze(1)
        return _apply_v41_rope(key_value, cos, sin)

    def _project_query(
            self,
            hidden_states: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project and normalize the local query heads."""
        query_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        return query_residual, self._expand_query(query_residual, cos, sin)

    def _expand_query(self, query_residual: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Expand main heads after low-rank query inputs have been dispatched."""
        query = self.q_b_proj(query_residual).view(*query_residual.shape[:2], -1, self.head_dim)
        # Preserve sequence-major storage for sparse attention while keeping the head-major API.
        return _apply_v41_rope(query, cos, sin, unsqueeze_dim=2).transpose(1, 2)

    def _validate_forward_inputs(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None,
            past_key_values: Any | None,
            kwargs: dict[str, Any],
    ) -> tuple[SharedCompressedAttentionState, SharedCompressedPackedSequence | None, int, int]:
        """Pop and validate the shared-attention keyword arguments."""
        if past_key_values is not None:
            raise NotImplementedError("shared compressed DSA training does not support KV cache")
        shared_state = kwargs.pop("shared_attention_state", None)
        if not isinstance(shared_state, SharedCompressedAttentionState):
            raise ValueError("shared_attention_state is required for shared compressed DSA")
        if attention_mask is not None:
            raise NotImplementedError(
                "shared compressed DSA consumes compact packed_seq_params instead of a dense mask"
            )
        batch_size, sequence_length, _ = hidden_states.shape
        packed_sequence = kwargs.pop("packed_seq_params", None)
        if packed_sequence is not None and not isinstance(
                packed_sequence, SharedCompressedPackedSequence
        ):
            raise TypeError(
                "packed_seq_params must be SharedCompressedPackedSequence, "
                f"got {type(packed_sequence).__name__}"
            )
        if "shared_attention_cp_context" in kwargs or "shared_attention_tp_context" in kwargs:
            raise ValueError("CSA2 parallel resources must be bound through its family, not forward kwargs")
        return shared_state, packed_sequence, batch_size, sequence_length

    def _resolve_packed_segments(
            self,
            packed_sequence: "SharedCompressedPackedSequence | None",
            batch_size: int,
            sequence_length: int,
            query_offset: int,
            global_sequence_length: int,
            device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, tuple, torch.Tensor | None]:
        """Validate packed document boundaries and derive local Q/K geometry.

        Both execution modes consume the same per-token tensors: document
        starts, per-query compressed-key minima and the packed causal upper
        bound. Python ``query_segments`` stay available for the eager fused
        kernels only; graph packing inputs never carry them.
        """
        segment_starts = None
        minimum_key_indices = None
        causal_upper = None
        ratio = max(1, self.compress_ratio)
        raw_end = query_offset + sequence_length
        query_segments = ((0, sequence_length, 0, raw_end // ratio, raw_end % ratio),)
        if packed_sequence is not None:
            if batch_size != 1:
                raise ValueError("V4.1 compact packed attention currently requires micro_batch_size=1")
            if (
                    packed_sequence.local_query_start != query_offset
                    or packed_sequence.local_query_length != sequence_length
                    or packed_sequence.global_sequence_length != global_sequence_length
            ):
                packed_shard = (
                    packed_sequence.local_query_start,
                    packed_sequence.local_query_length,
                    packed_sequence.global_sequence_length,
                )
                raise ValueError(
                    "packed sequence geometry does not match the local CP shard: "
                    f"packed={packed_shard}, "
                    f"attention={(query_offset, sequence_length, global_sequence_length)}"
                )
            prepared = packed_sequence.prepare(device, (self.compress_ratio,))
            segment_starts = prepared.local_segment_starts(device)
            if self.compress_ratio:
                minimum_key_indices = prepared.minimum_key_indices(device, self.compress_ratio)
                causal_upper = prepared.causal_key_upper_bounds(device, self.compress_ratio)
            if (self.candidate_source_layer_idx is not None
                    and not prepared.is_compression_aligned(ratio)):
                raise ValueError("candidate-pool packed attention still requires compression-aligned documents")
            query_segments = (
                None if prepared._restored_from_packing_tensors  # pylint: disable=protected-access
                else prepared.indexer_segments(ratio)
            )
        return segment_starts, minimum_key_indices, query_segments, causal_upper

    def prepare_document_inputs(self, hidden_states: torch.Tensor, position_ids: torch.Tensor,
                                packed: SharedCompressedPackedSequence | None) -> CSA2DocumentInputs:
        """Prepare mathematical document bounds without launching communication."""
        batch, length = hidden_states.shape[:2]
        offset, global_length = self.query_partition(length, position_ids)
        if packed is not None:
            packed = packed.prepare(hidden_states.device, (self.compress_ratio,))
            if self.use_fused_ops and packed._restored_from_packing_tensors:  # pylint: disable=protected-access
                raise NotImplementedError(
                    "fused DSA operators require Python document segments, not graph packing inputs"
                )
        starts, minima, segments, upper = self._resolve_packed_segments(
            packed, batch, length, offset, global_length, hidden_states.device,
        )
        keys = (packed.compressed_key_length(self.compress_ratio) if packed is not None
                else global_length // max(1, self.compress_ratio))
        return CSA2DocumentInputs(packed, offset, global_length, starts, minima, segments, upper,
                                  () if packed is None else packed.raw_document_starts(), keys)

    def prepare_compression(self, document: CSA2DocumentInputs, embeddings: dict,
                            device: torch.device) -> tuple:
        """Select complete local compression blocks and their original RoPE positions."""
        if not self.is_kv_source or self.compress_ratio <= 1:
            return None, None
        packed = document.packed
        if packed is None:
            if document.global_length % self.compress_ratio == 0:
                return None, None
            packed = SharedCompressedPackedSequence(torch.tensor((0, document.global_length)),
                                                     0, document.global_length, document.global_length)
        packed = packed.prepare(device, (self.compress_ratio,))
        plan = packed.block_plan(self.compress_ratio)
        positions = None if plan is None else packed.compressed_positions(plan, embeddings["compress"])
        return plan, positions

    def compress_key_value(self, hidden_states: torch.Tensor, embeddings: dict,
                           block_plan: Any, key_positions: tuple | None) -> tuple:
        """Call the compressor through its managed module boundary."""
        if not self.is_kv_source:
            return None, None
        if block_plan is None:
            return self.compressor(hidden_states, embeddings["compress"])
        return self.compressor(hidden_states, key_positions, block_plan)

    def prepare_attention_inputs(self, hidden_states: torch.Tensor, embeddings: dict,
                                 shared_state: SharedCompressedAttentionState,
                                 document: CSA2DocumentInputs) -> CSA2AttentionInputs:
        """Prepare ready local operands for the common attention forward."""
        plan, key_positions = self.prepare_compression(document, embeddings, hidden_states.device)
        cos, sin = embeddings["compress" if self.compress_ratio else "main"]
        key_value = self._project_raw_kv(hidden_states, cos, sin)
        qr, query = self._project_query(hidden_states, cos, sin)
        latent, compressed = self.compress_key_value(hidden_states, embeddings, plan, key_positions)
        if compressed is not None:
            shared_state.publish_compressed_kv(self.layer_idx, compressed)
        indexed = self._run_indexer(hidden_states, qr, latent, embeddings, shared_state, document,
                                    block_plan=plan, key_position_embeddings=key_positions)
        return CSA2AttentionInputs(query, key_value, indexed)

    def exchange_attention(self, prepared: Any, shared_state: SharedCompressedAttentionState) -> CSA2AttentionInputs:
        """Keep local ready operands; CP members wait at this consumer boundary."""
        del shared_state
        return prepared

    def _run_indexer(
            self,
            hidden_states: torch.Tensor,
            query_residual: torch.Tensor,
            latent: torch.Tensor | None,
            position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]],
            shared_state: "SharedCompressedAttentionState",
            document: CSA2DocumentInputs,
            document_lengths: tuple[int, ...] = (),
            block_plan: Any = None,
            key_position_embeddings: tuple | None = None,
            prepare_attention: Callable[[], None] | None = None,
    ) -> Any | None:
        """Run the indexer on index-source layers and publish its forward outputs."""
        if not self.is_index_source:
            return None
        index_key = (
            None
            if self.indexer.owns_key
            else shared_state.require_index_key(self.kv_source_layer_idx, self.layer_idx)
        )
        candidate_blocks = (
            shared_state.require_candidate_blocks(
                self.candidate_source_layer_idx,
                self.layer_idx,
            )
            if self.indexer.uses_candidates else None
        )
        indexer_output = self.indexer(
            hidden_states,
            query_residual,
            latent,
            position_embeddings["compress"],
            prepare_attention=prepare_attention,
            index_key=index_key,
            candidate_blocks=candidate_blocks,
            reduce_sum=self.head_reducer(),
            query_offset=document.query_offset,
            minimum_key_indices=document.minimum_key_indices,
            query_segments=document.query_segments,
            block_plan=block_plan,
            key_position_embeddings=key_position_embeddings,
            document_lengths=document_lengths,
            causal_upper=document.causal_upper,
        )
        if self.indexer.owns_key:
            shared_state.publish_index_key(self.layer_idx, indexer_output.index_key)
        shared_state.publish_topk_indices(self.layer_idx, indexer_output.topk_indices)
        if indexer_output.candidate_blocks is not None:
            shared_state.publish_candidate_blocks(
                self.layer_idx,
                indexer_output.candidate_blocks,
            )
        return indexer_output

    def _build_sparse_indices(
            self,
            key_value: torch.Tensor,
            compressed_kv: torch.Tensor | None,
            topk_indices: torch.Tensor | None,
            sequence_length: int,
            query_offset: int,
            global_sequence_length: int,
            segment_starts: torch.Tensor | None,
            raw_key_offset: int = 0,
            main_key_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assemble dense-reference banks only after reference dispatch is selected."""
        window = build_sliding_window_indices(
            key_value.shape[0], sequence_length, self.sliding_window, key_value.device,
            query_offset=query_offset, key_length=global_sequence_length,
        )
        if segment_starts is not None:
            window.masked_fill_(window < segment_starts.unsqueeze(-1), -1)
        window = torch.where(window >= 0, window - raw_key_offset, -1)
        if compressed_kv is None:
            return key_value, window
        combined_key_value = torch.cat((key_value, compressed_kv.unsqueeze(1)), dim=2)
        compressed_indices = torch.where(
            topk_indices >= 0, topk_indices.long() - main_key_offset + key_value.shape[2], -1,
        )
        return combined_key_value, torch.cat((window, compressed_indices), dim=-1)

    def _apply_indexer_loss(
            self,
            indexer_output: Any | None,
            query: torch.Tensor,
            compressed_kv: torch.Tensor | None,
            query_segments: tuple,
            main_key_offset: int = 0,
    ) -> torch.Tensor:
        """Auto-scale the query by the indexer KL loss when configured."""
        if (
                indexer_output is not None
                and self.training
                and self.indexer.loss_coeff
                and compressed_kv is not None
        ):
            # KL uses the original owner's queries. A view aligns its student K
            # with the compact teacher bank while preserving global Index K gradients.
            index_key = indexer_output.index_key
            if main_key_offset or compressed_kv.shape[1] != index_key.shape[1]:
                index_key = index_key[:, main_key_offset:main_key_offset+compressed_kv.shape[1]]
            selected = indexer_output.topk_indices
            if main_key_offset:
                selected = torch.where(selected >= 0, selected - main_key_offset, -1)
                query_segments = tuple((s, e, ks-main_key_offset, ke-main_key_offset, residual)
                                       for s, e, ks, ke, residual in query_segments)
            indexer_loss = shared_compressed_indexer_kl_loss(
                indexer_output.index_query,
                index_key,
                indexer_output.merge_weight,
                query,
                compressed_kv,
                selected,
                self.sinks,
                attention_scale=self.scaling,
                loss_coeff=self.indexer.loss_coeff,
                query_chunk_size=self.indexer.query_chunk_size,
                reduce_sum=self.head_reducer(),
                compress_ratio=self.compress_ratio,
                query_segments=query_segments,
                use_fused=self.use_fused_ops,
            )
            query = aux_loss_auto_scale(query, indexer_loss)
        return query

    def _run_sparse_attention(
            self, query: torch.Tensor, key_value: torch.Tensor, compressed_kv: torch.Tensor | None,
            topk_indices: torch.Tensor | None, query_offset: int, global_sequence_length: int,
            segment_starts: torch.Tensor | None, query_segments: tuple, raw_key_offset: int = 0,
            main_key_offset: int = 0, raw_document_starts: tuple = (), global_key_length: int | None = None,
    ) -> torch.Tensor:
        """Run the chosen implementation; unsupported explicit fusion always raises."""
        if self.use_fused_ops:
            expected_topk = min(512, global_key_length if global_key_length is not None
                                else global_sequence_length // max(1, self.compress_ratio))
            if not (self.candidate_source_layer_idx is None
                    and (topk_indices is None or topk_indices.shape[-1] == expected_topk)
                    and supports_fused_attention_inputs(
                        query, self.compress_ratio, self.sliding_window,
                        0 if topk_indices is None else topk_indices.shape[-1])):
                raise RuntimeError(
                    "V4.1 fused attention is required but this call is unsupported: "
                    f"layer={self.layer_idx}, device={query.device}, dtype={query.dtype}, Q={tuple(query.shape)}, "
                    f"ratio={self.compress_ratio}, window={self.sliding_window}. "
                    "Check the SMLA training extension and native input contract; no reference fallback was run."
                )
            output = sparse_flash_mla(
                query, key_value, compressed_kv, topk_indices, self.sinks, self.scaling,
                self.compress_ratio, self.sliding_window, query_offset, query_segments, raw_key_offset, main_key_offset,
                raw_document_starts,
            )
            if compressed_kv is not None and all(end == start for _, _, start, end, _ in query_segments):
                # Empty-prefix ranks must still participate in the shared bank's collective backward.
                output = output + compressed_kv[:, :0].sum().to(output.dtype)
            return output
        combined_key_value, sparse_indices = self._build_sparse_indices(
            key_value, compressed_kv, topk_indices, query.shape[2], query_offset,
            global_sequence_length, segment_starts, raw_key_offset, main_key_offset,
        )
        return reference_sparse_attention(query, combined_key_value, sparse_indices, self.sinks, self.scaling)

    def _project_output(
            self,
            attention_output: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            batch_size: int,
            sequence_length: int,
    ) -> torch.Tensor:
        """Undo RoPE and apply the grouped low-rank output projection."""
        attention_output = _apply_v41_rope(attention_output, cos, -sin, unsqueeze_dim=2)
        num_groups = attention_output.shape[2] * self.head_dim // self.o_a_proj.weight.shape[1]
        grouped = attention_output.reshape(batch_size, sequence_length, num_groups, -1)
        hidden_per_group = grouped.shape[-1]
        output_per_group = self.o_a_proj.weight.shape[0] // num_groups
        projection = self.o_a_proj.weight.view(
            num_groups, output_per_group, hidden_per_group
        ).transpose(1, 2)
        flattened = grouped.reshape(-1, num_groups, hidden_per_group).transpose(0, 1)
        projected = torch.bmm(flattened, projection).transpose(0, 1)
        projected = projected.reshape(batch_size, sequence_length, -1)
        return self.o_b_proj(projected)

    def forward(
            self, hidden_states: torch.Tensor,
            position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]],
            position_ids: torch.Tensor, attention_mask: torch.Tensor | None,
            past_key_values: Any | None = None, **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        """Run common math; the family/framework owns distributed port conversion."""
        shared, packed, batch, length = self._validate_forward_inputs(
            hidden_states, attention_mask, past_key_values, kwargs,
        )
        document = self.prepare_document_inputs(hidden_states, position_ids, packed)
        prepared = self.prepare_attention_inputs(hidden_states, position_embeddings, shared, document)
        inputs = self.exchange_attention(prepared, shared)
        del prepared
        compressed, selected, main_offset = None, None, 0
        if self.compress_ratio:
            compressed = shared.require_compressed_kv(self.kv_source_layer_idx, self.layer_idx)
            selected = shared.require_topk_indices(self.index_source_layer_idx, self.layer_idx)
            main_offset = shared.compressed_kv_offsets.get(self.kv_source_layer_idx, 0)
        query = self._apply_indexer_loss(inputs.indexer_output, inputs.query, compressed,
                                        document.query_segments, main_offset)
        output = self._run_sparse_attention(
            query, inputs.key_value, compressed, selected, document.query_offset, document.global_length,
            document.segment_starts, document.query_segments, inputs.raw_key_offset, main_offset,
            document.raw_document_starts, document.global_key_length,
        )
        del inputs
        cos, sin = position_embeddings["compress" if self.compress_ratio else "main"]
        return self._project_output(output, cos, sin, batch, length), None
