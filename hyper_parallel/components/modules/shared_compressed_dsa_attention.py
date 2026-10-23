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
"""High-performance shared compressed DeepSeek Sparse Attention.

This module implements the DeepSeek-V4.1 CSA2 training contract: Full layers
publish compressed K=V and index K, Reindex layers produce fresh Top-K indices,
and Reuse layers consume the latest selection. The decoder's first Full layer
can additionally publish the blockwise candidate pool used by later Reindex
layers. The attention replacement selects eager PyTorch references or fused
Indexer, sparse attention and external-teacher KL operators while preserving
the same sharing and parallel contexts. Explicit fusion rejects unsupported
calls instead of falling back to another implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.functional.aux_loss import aux_loss_auto_scale
from hyper_parallel.components.functional.compressed_document import document_compression_plan
from hyper_parallel.components.functional.lightning_indexer import (
    lightning_indexer, sort_key_indices, validate_indexer_tensors,
)
from hyper_parallel.components.functional.sparse_flash_mla import (
    supports_fused_attention_inputs, sparse_flash_mla,
)
from hyper_parallel.components.functional.sparse_lightning_indexer_kl_loss_grad import (
    gather_selected_keys, sparse_lightning_indexer_kl_loss, validate_fused_kl_inputs,
)
from hyper_parallel.core.activation_memory.recompute_state import is_recomputing
from hyper_parallel.core.activation_memory.wrapper import checkpoint_exclude_wrapper
from hyper_parallel.distributed.context_parallel.index_key_dispatch import IndexKeyPlan
from hyper_parallel.distributed.context_parallel.indexer_balance import (
    resolve_indexer_balance_block_size, validate_indexer_hybrid_thresholds,
)
from hyper_parallel.distributed.context_parallel.sequence_dispatch import (
    SequenceDispatchRoute, compressed_document_offsets, document_prefix_route, sliding_window_route,
)
from hyper_parallel.models.replacement import module_replacement


class SequenceGatherHandle(Protocol):
    """Handle returned by asynchronous differentiable bank communication."""

    def wait(self) -> torch.Tensor:
        """Wait for communication and return the differentiable result."""


@dataclass
class _DeferredSequenceGather:
    """Synchronous fallback with the same interface as an async gather."""

    tensor: torch.Tensor
    sequence_dim: int
    gather_sequence: Callable[[torch.Tensor, int], torch.Tensor]

    def wait(self) -> torch.Tensor:
        """Run the fallback gather at the first consumer boundary."""
        return self.gather_sequence(self.tensor, self.sequence_dim)


@dataclass
class SharedCompressedAttentionState:
    """Autograd-bearing CSA2 state scoped to one training forward.

    State is addressed by the layer that produced it instead of using one
    mutable "latest value" slot. This matters for the released V4.1 topology,
    which has several Full/Reindex groups, and makes a consumer's dependency
    explicit when activation recomputation revisits modules out of forward
    order.

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
    index_key_plans: dict[int, IndexKeyPlan] = field(default_factory=dict)

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

    def publish_compressed_kv(self, source_layer: int, value: torch.Tensor, key_offset: int = 0) -> None:
        """Publish compressed K=V and its global origin, preserving the autograd graph."""
        self.compressed_kv_by_source[source_layer] = value
        self.compressed_kv_offsets[source_layer] = key_offset

    def require_compressed_kv(self, source_layer: int | None, consumer_layer: int) -> torch.Tensor:
        """Read compressed K=V from the consumer's configured Full layer."""
        return self._require(
            self.compressed_kv_by_source,
            source_layer,
            "compressed K=V",
            consumer_layer,
        )

    def publish_index_key(self, source_layer: int, value: torch.Tensor, plan: IndexKeyPlan | None = None) -> None:
        """Publish the shared key tensor together with its independent physical layout."""
        self.index_key_by_source[source_layer] = value
        if plan is None:
            self.index_key_plans.pop(source_layer, None)
        else:
            self.index_key_plans[source_layer] = plan

    def require_index_key(self, source_layer: int | None, consumer_layer: int) -> torch.Tensor:
        """Read the Indexer key associated with a compressed-KV source."""
        return self._require(
            self.index_key_by_source,
            source_layer,
            "Indexer key",
            consumer_layer,
        )

    def publish_topk_indices(self, source_layer: int, value: torch.Tensor) -> None:
        """Publish one Full/Reindex layer's token selection."""
        self.topk_indices_by_source[source_layer] = value

    def require_topk_indices(self, source_layer: int | None, consumer_layer: int) -> torch.Tensor:
        """Read the latest configured Full/Reindex selection."""
        return self._require(
            self.topk_indices_by_source,
            source_layer,
            "Top-K indices",
            consumer_layer,
        )

    def publish_candidate_blocks(self, source_layer: int, value: torch.Tensor) -> None:
        """Publish hierarchical candidate blocks from the configured source."""
        self.candidate_blocks_by_source[source_layer] = value

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


@dataclass
class SharedCompressedIndexerOutput:
    """Tensors produced by one Full or Reindex Indexer invocation."""

    topk_indices: torch.Tensor
    index_key: torch.Tensor
    candidate_blocks: torch.Tensor | None
    index_query: torch.Tensor
    merge_weight: torch.Tensor
    key_plan: IndexKeyPlan | None = None


@dataclass(frozen=True)
class SharedCompressedAttentionCPContext:
    """Communication for local queries and shared compressed banks.

    ``launch_raw`` exchanges the same-document SWA halo when supplied.
    Optional ``launch_main`` exchanges main KV document prefixes. ``launch_index``
    and ``plan_index_key`` select independent compact Index K prefixes; otherwise
    Index K retains differentiable all-gather.
    Contexts without ``launch_raw`` provide the full raw-bank reference.
    """

    size: int
    rank: int
    gather_sequence: Callable[[torch.Tensor, int], torch.Tensor]
    launch_sequence: Callable[[torch.Tensor, int], SequenceGatherHandle] | None = None
    launch_raw: Callable[[torch.Tensor, int, SequenceDispatchRoute], SequenceGatherHandle] | None = None
    balance_indexer: Callable[..., torch.Tensor] | None = None
    launch_main: Callable[[torch.Tensor, int, SequenceDispatchRoute], SequenceGatherHandle] | None = None
    launch_index: Callable[[torch.Tensor, int, IndexKeyPlan], SequenceGatherHandle] | None = None
    plan_index_key: Callable[[tuple, int], IndexKeyPlan] | None = None
    launch_boundary: Callable[[torch.Tensor, int, SequenceDispatchRoute], SequenceGatherHandle] | None = None

    def launch(self, tensor: torch.Tensor, sequence_dim: int) -> SequenceGatherHandle:
        """Launch an async gather or create a deferred synchronous gather."""
        if self.launch_sequence is not None:
            return self.launch_sequence(tensor, sequence_dim)
        return _DeferredSequenceGather(tensor, sequence_dim, self.gather_sequence)


@dataclass(frozen=True)
class SharedCompressedAttentionTPContext:
    """Tensor-parallel score reduction context for a head-sharded Indexer."""

    size: int
    rank: int
    reduce_sum: Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class SharedCompressedPackedSequence:
    """Compact sample boundaries for one contiguous CP query shard."""

    cu_seq_lens: torch.Tensor
    local_query_start: int
    local_query_length: int
    global_sequence_length: int
    _prepared_starts: dict[torch.device, torch.Tensor] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _prepared_minima: dict[tuple[torch.device, int], torch.Tensor] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _compression_rope: dict[int, tuple] = field(default_factory=dict, init=False, repr=False, compare=False)
    _snapshot: dict[str, Any] = field(default_factory=dict, init=False, repr=False, compare=False)
    _prepared_positions: dict[torch.device, torch.Tensor] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _prepared_masks: dict[torch.device, torch.Tensor] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _indexer_segments: dict[int, tuple[tuple[int, int, int, int, int], ...]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def prepare(self, device: torch.device, compress_ratios: tuple[int, ...]) -> SharedCompressedPackedSequence:
        """Prepare an owned boundary snapshot, reused until the input is mutated.

        Normal in-place tensor updates invalidate the cached snapshot. Existing
        consumers retain their owned geometry. Prepared tensors are read-only;
        callers must not mutate boundaries through ``.data`` or external storage.
        Inference tensors without a version counter are always snapshotted anew.

        Args:
            device: Device on which tensors and prepared geometry are consumed.
            compress_ratios: Compression ratios used by the layers in this forward.
        """
        device = torch.device(device)
        # Torch exposes tensor mutation tracking only through its version counter.
        version = None if self.cu_seq_lens.is_inference() else self.cu_seq_lens._version  # pylint: disable=protected-access
        if version is not None and self._snapshot.get("version") == version:
            prepared: SharedCompressedPackedSequence = self if self._snapshot.get("owned") else self._snapshot["value"]
        else:
            prepared = SharedCompressedPackedSequence(
                self.cu_seq_lens.detach().clone(), self.local_query_start,
                self.local_query_length, self.global_sequence_length,
            )
            SharedCompressedPackedSequence._mark_owned(prepared)
            self._snapshot.clear()
            self._snapshot.update(version=version, value=prepared)
        SharedCompressedPackedSequence._prepare_owned_metadata(prepared, device, compress_ratios)
        return prepared

    def _mark_owned(self) -> None:
        """Record the owned tensor's mutation counter for subsequent layer reuse."""
        if not self.cu_seq_lens.is_inference():
            # Torch has no public read API for its tensor mutation counter.
            self._snapshot.update(owned=True, version=self.cu_seq_lens._version)  # pylint: disable=protected-access

    def _prepare_owned_metadata(self, device: torch.device, compress_ratios: tuple[int, ...]) -> None:
        """Populate caches on the owned snapshot, never on caller-owned boundaries."""
        starts = self.local_segment_starts(device)
        self._prepared_starts[device] = starts
        for ratio in set(compress_ratios):
            self.validate_compression_ratio(ratio)
            if ratio > 0 and (device, ratio) not in self._prepared_minima:
                self._prepared_minima[(device, ratio)] = self.minimum_key_indices(device, ratio)

    def local_positions(self, device: torch.device) -> torch.Tensor:
        """Return the immutable local interval's global token positions."""
        device = torch.device(device)
        if device not in self._prepared_positions:
            self._prepared_positions[device] = torch.arange(
                self.local_query_start, self.local_query_start + self.local_query_length, device=device,
            ).unsqueeze(0)
        return self._prepared_positions[device]

    def segment_start_mask(self, device: torch.device) -> torch.Tensor:
        """Reuse the per-batch sample-start mask across model forwards."""
        device = torch.device(device)
        if device not in self._prepared_masks:
            self._prepared_masks[device] = self.local_positions(device) == self.local_segment_starts(device)
        return self._prepared_masks[device]

    def boundaries(self) -> tuple[int, ...]:
        """Reuse the owned batch's host boundaries for routing across layers."""
        if "boundaries" not in self._snapshot:
            self._snapshot["boundaries"] = tuple(self.cu_seq_lens.detach().cpu().tolist())
        return self._snapshot["boundaries"]

    def indexer_segments(self, ratio: int) -> tuple[tuple[int, int, int, int, int], ...]:
        """Describe packed Q intervals and completed K prefixes without per-layer synchronization."""
        if ratio not in self._indexer_segments:
            self.validate_compression_ratio(ratio)
            boundaries = self.boundaries()
            local_start = self.local_query_start
            local_end = local_start + self.local_query_length
            segments = []
            key_base = 0
            for sample_start, sample_end in zip(boundaries, boundaries[1:]):
                start, end = max(local_start, sample_start), min(local_end, sample_end)
                if start < end:
                    segments.append((start-local_start, end-local_start, key_base,
                                     key_base+(end-sample_start)//ratio, (end-sample_start) % ratio))
                key_base += (sample_end-sample_start)//ratio
            self._indexer_segments[ratio] = tuple(segments)
        return self._indexer_segments[ratio]

    def minimum_key_indices(
            self, device: torch.device, compress_ratio: int, segment_starts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the first compressed key belonging to each query's sample.

        Args:
            device: Device on which tensors and prepared geometry are consumed.
            compress_ratio: Compression ratio of the consuming layer.
            segment_starts: Optional precomputed global sample starts for local queries.
        """
        if compress_ratio <= 0:
            raise ValueError("minimum_key_indices requires a positive compression ratio")
        cached = self._prepared_minima.get((torch.device(device), compress_ratio))
        if cached is not None:
            return cached
        self.validate_compression_ratio(compress_ratio)
        del segment_starts
        boundaries = self.cu_seq_lens.to(device=device)
        documents = torch.bucketize(self.local_positions(device).squeeze(0), boundaries[1:], right=True)
        offsets = torch.tensor(compressed_document_offsets(self.boundaries(), compress_ratio),
                               device=device, dtype=torch.long)
        return offsets.index_select(0, documents).unsqueeze(0)

    def local_segment_starts(self, device: torch.device) -> torch.Tensor:
        """Return each local query's global packed-sample start position.

        Args:
            device: Device on which tensors and prepared geometry are consumed.
        """
        cached = self._prepared_starts.get(torch.device(device))
        if cached is not None:
            return cached
        if self.local_query_start < 0 or self.local_query_length < 0 or (
                self.local_query_start + self.local_query_length > self.global_sequence_length
        ):
            raise ValueError("packed local query interval must lie within the global sequence")
        boundaries = self.cu_seq_lens.to(dtype=torch.long)
        if boundaries.ndim != 1 or boundaries.numel() < 2 or int(boundaries[0]) != 0:
            raise ValueError("packed cu_seq_lens must be one-dimensional and start with zero")
        if int(boundaries[-1]) != self.global_sequence_length:
            raise ValueError(
                "packed cu_seq_lens must cover global_sequence_length: "
                f"{int(boundaries[-1])} versus {self.global_sequence_length}"
            )
        if torch.any(boundaries[1:] <= boundaries[:-1]):
            raise ValueError("packed cu_seq_lens must be strictly increasing")
        boundaries = boundaries.to(device=device)
        query_positions = self.local_positions(device).squeeze(0)
        segment_ids = torch.bucketize(query_positions, boundaries[1:], right=True)
        return boundaries[:-1].index_select(0, segment_ids).unsqueeze(0)

    @staticmethod
    def validate_compression_ratio(compress_ratio: int) -> None:
        """Validate the ratio; complete block planning handles natural document lengths."""
        if compress_ratio < 0:
            raise ValueError("compression ratio must be nonnegative")

    def block_plan(self, ratio: int, size: int, rank: int) -> Any:
        """Use the original contiguous compressor fast path when every group is aligned."""
        if ratio <= 1 or (self.local_query_length % ratio == 0
                          and all(boundary % ratio == 0 for boundary in self.boundaries())):
            return None
        return document_compression_plan(self.boundaries(), ratio, self.local_query_length, size, rank)

    def compressed_positions(self, plan: Any, embeddings: tuple, context: Any) -> tuple:
        """Reuse boundary RoPE once per batch/source embedding, retaining its original precision."""
        cos, sin = embeddings
        # Preserve cache correctness when caller-owned RoPE tensors are updated in place.
        versions = tuple(None if item.is_inference() else item._version for item in embeddings)  # pylint: disable=protected-access
        cached = self._compression_rope.get(plan.ratio)
        if (cached is not None and cached[0] is cos and cached[1] is sin and cached[2] == versions
                and None not in versions and not cos.requires_grad and not sin.requires_grad):
            return cached[3]
        result = plan.positions(cos, sin, context)
        if not cos.requires_grad and not sin.requires_grad:
            self._compression_rope[plan.ratio] = (cos, sin, versions, result)
        return result

    def raw_document_starts(self) -> tuple:
        """Keep raw token coordinates separate from compact compressed-key IDs."""
        first, last = self.local_query_start, self.local_query_start + self.local_query_length
        return tuple((max(first, start)-first, start)
                     for start, end in zip(self.boundaries(), self.boundaries()[1:])
                     if max(first, start) < min(last, end))


def _rotate_half(tensor: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent real/imaginary pairs for V4 interleaved RoPE."""
    first = tensor[..., 0::2]
    second = tensor[..., 1::2]
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
    pass_through, rotary = tensor[..., :-rope_dim], tensor[..., -rope_dim:]
    rotary = ((rotary.float() * cos) + (_rotate_half(rotary).float() * sin)).to(tensor.dtype)
    return torch.cat((pass_through, rotary), dim=-1)


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


def _run_lightning_indexer(
        query: torch.Tensor, key: torch.Tensor, merge_weight: torch.Tensor,
        top_k: int, compress_ratio: int, query_offset: int,
        reduce_sum: Callable[[torch.Tensor], torch.Tensor] | None,
        minimum_key_indices: torch.Tensor | None, query_segments: tuple | None,
) -> torch.Tensor:
    """Run LI V2 with explicit adapter errors and native capability validation."""
    if reduce_sum is not None:
        raise NotImplementedError("Fused LightningIndexer does not support TP score reduction before TopK")
    if minimum_key_indices is not None and query_segments is None:
        raise ValueError("Fused LightningIndexer requires query_segments when minimum_key_indices is set")
    validate_indexer_tensors(query, key, merge_weight)
    if query.device.type != "npu":
        raise RuntimeError(f"Fused LightningIndexer requires NPU inputs; got device={query.device}")
    if query_segments is None:
        length = query.shape[1]
        raw_end = query_offset + length
        query_segments = ((0, length, 0, raw_end // compress_ratio, raw_end % compress_ratio),)
    return lightning_indexer(query, key, merge_weight, compress_ratio, top_k, query_segments)


def _causal_key_bounds(
        query: torch.Tensor, ratio: int, query_offset: int,
        minimum_key_indices: torch.Tensor | None, segments: tuple | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Derive visibility from original query positions after task redistribution."""
    batch_size, length = query.shape[:2]
    if segments is None:
        positions = torch.arange(query_offset, query_offset + length, device=query.device)
        return (positions + 1) // ratio, minimum_key_indices
    upper = torch.empty(length, device=query.device, dtype=torch.long)
    lower = torch.empty_like(upper)
    for begin, end, key_start, key_end, residual in segments:
        first = ratio * key_end + residual - (end - begin)
        positions = torch.arange(first, first + end - begin, device=query.device)
        upper[begin:end] = (positions + 1) // ratio
        lower[begin:end] = key_start
    return upper, lower.unsqueeze(0).expand(batch_size, -1)


def _validate_causal_inputs(
        query: torch.Tensor, key: torch.Tensor, merge_weight: torch.Tensor, compress_ratio: int,
        query_chunk_size: int, minimum_key_indices: torch.Tensor | None,
) -> None:
    """Validate selection tensor relationships before scoring or native dispatch."""
    if compress_ratio <= 0:
        raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
    if query_chunk_size <= 0:
        raise ValueError(f"query_chunk_size must be positive, got {query_chunk_size}")
    if query.ndim != 4:
        raise ValueError(f"Indexer query must have shape [B, Q, H, D], got {tuple(query.shape)}")
    batch_size, _, _, head_dim = query.shape
    if key.ndim != 3 or key.shape[::2] != (batch_size, head_dim):
        raise ValueError(
            "compressed index key must have shape [batch, compressed_sequence, head_dim]"
        )
    if merge_weight.shape != query.shape[:3]:
        raise ValueError(
            f"merge_weight must have shape {tuple(query.shape[:3])}, got {tuple(merge_weight.shape)}"
        )
    if minimum_key_indices is not None and minimum_key_indices.shape != query.shape[:2]:
        raise ValueError("minimum_key_indices must have shape [batch, query]")


def _reference_causal_topk(
        scores: torch.Tensor, top_k: int, logical_length: int,
        key_global_positions: torch.Tensor | None,
) -> torch.Tensor:
    """Keep the reference TopK tie convention when remote K rows are omitted.

    Eager TopK can choose different tied IDs when its input width changes.
    Restore logical score slots only; the physical K bank stays compact.
    Native LI already packs the same per-document prefix in both CP modes.
    """
    if key_global_positions is not None:
        full_scores = scores.new_full((*scores.shape[:-1], logical_length), float("-inf"))
        full_scores.index_copy_(-1, key_global_positions, scores)
        scores = full_scores
    length = scores.shape[-1]
    top = scores.topk(min(top_k, length), dim=-1, sorted=False)
    indices = top.indices.masked_fill(~torch.isfinite(top.values), length)
    indices = sort_key_indices(indices, length)
    indices = indices.masked_fill(indices == length, -1)
    if indices.shape[-1] < top_k:
        indices = torch.nn.functional.pad(indices, (0, top_k - indices.shape[-1]), value=-1)
    return indices.to(torch.int32)


@torch.no_grad()
def compressed_causal_topk(
        query: torch.Tensor,
        key: torch.Tensor,
        merge_weight: torch.Tensor,
        *,
        compress_ratio: int,
        sparse_count: int,
        query_offset: int = 0,
        query_chunk_size: int = 256,
        reduce_sum: Callable[[torch.Tensor], torch.Tensor] | None = None,
        minimum_key_indices: torch.Tensor | None = None,
        query_segments: tuple[tuple[int, int, int, int, int], ...] | None = None,
        use_fused: bool = False,
        global_key_length: int | None = None,
        key_global_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select compressed positions with V4.1's ratio-aware causal rule.

    CSA2 keys represent closed groups of ``compress_ratio`` source tokens.
    Both LI V2 and the batched reference preserve the source rule
    ``key < floor((query + 1) / ratio)``. The optional native path requires
    document-local K prefixes; the reference retains TP score reductions.

    Args:
        query_segments: Local Q intersections and physical compressed K prefixes.
        global_key_length: Logical bank length before optional CP compaction, used
            to retain the global output width. Unfilled slots remain -1.
        key_global_positions: Optional compact-to-global row IDs for eager TopK.
            Logical score slots preserve the original reference tie selection.
        use_fused: Select LI V2 and raise if a nonempty call is unsupported.
            False explicitly selects the numerical reference.
    """
    _validate_causal_inputs(query, key, merge_weight, compress_ratio, query_chunk_size, minimum_key_indices)
    batch_size, sequence_length = query.shape[:2]
    compressed_length = key.shape[1]
    logical_length = compressed_length if global_key_length is None else global_key_length
    if logical_length < compressed_length:
        raise ValueError("global_key_length cannot be shorter than the physical Index K bank")
    top_k = min(sparse_count, logical_length)
    if top_k <= 0 or sequence_length == 0 or compressed_length == 0:
        return torch.full(
            (batch_size, sequence_length, max(top_k, 0)), -1, dtype=torch.int32, device=query.device,
        )

    if use_fused:
        return _run_lightning_indexer(
            query, key, merge_weight, top_k, compress_ratio, query_offset,
            reduce_sum, minimum_key_indices, query_segments,
        )

    key_fp32 = key.float().transpose(1, 2).unsqueeze(1)
    key_positions = torch.arange(compressed_length, device=query.device).view(1, 1, -1)
    upper_limits, minimum_key_indices = _causal_key_bounds(
        query, compress_ratio, query_offset, minimum_key_indices, query_segments,
    )
    selected_chunks = []
    for start in range(0, sequence_length, query_chunk_size):
        end = min(start + query_chunk_size, sequence_length)
        # SAC may retain the matmul result; do not mutate its cached tensor.
        scores = torch.matmul(query[:, start:end].float(), key_fp32).relu()
        scores = (scores * merge_weight[:, start:end].float().unsqueeze(-1)).sum(dim=2)
        if reduce_sum is not None:
            scores = reduce_sum(scores)
        visible = upper_limits[start:end]
        scores.masked_fill_(key_positions >= visible.view(1, -1, 1), float("-inf"))
        if minimum_key_indices is not None:
            minimum = minimum_key_indices[:, start:end].unsqueeze(-1)
            scores.masked_fill_(key_positions < minimum, float("-inf"))
        selected_chunks.append(_reference_causal_topk(scores, top_k, logical_length, key_global_positions))
    return torch.cat(selected_chunks, dim=1)


def select_candidate_blocks(
        logits: torch.Tensor,
        compress_lens: torch.Tensor | int,
        topk_blocks: int,
        block_size: int,
) -> torch.Tensor:
    """Select the official block-max hierarchical candidate pool.

    The newest reachable partial block is pinned into the candidate set, then
    the remaining blocks compete by their maximum position score. ``logits``
    must already contain ``-inf`` at causally unreachable positions.
    """
    if topk_blocks <= 0:
        raise ValueError(f"topk_blocks must be positive, got {topk_blocks}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    width = logits.shape[-1]
    if width == 0:
        return torch.zeros_like(logits, dtype=torch.bool)
    padding = -width % block_size
    scores = torch.nn.functional.pad(logits, (0, padding), value=float("-inf"))
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.shape[-1]
    if isinstance(compress_lens, torch.Tensor):
        last = (compress_lens.to(device=logits.device) - 1) // block_size
    else:
        last = (compress_lens - 1) // block_size
    block_positions = torch.arange(num_blocks, device=logits.device)
    scores = scores.masked_fill(block_positions == last, float("inf"))
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool)
    keep.scatter_(-1, top.indices, top.values > float("-inf"))
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def select_candidate_block_indices(
        logits: torch.Tensor,
        compress_lens: torch.Tensor | int,
        topk_blocks: int,
        block_size: int,
) -> torch.Tensor:
    """Return compact block ids for the official hierarchical candidates.

    Storing block ids uses ``block_size`` times less cross-layer state than
    expanded position ids and avoids the full ``[batch, query, key]`` boolean
    mask used by the released inference reference. Invalid slots are ``-1``.
    """
    if topk_blocks <= 0:
        raise ValueError(f"topk_blocks must be positive, got {topk_blocks}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    width = logits.shape[-1]
    if width == 0:
        return torch.empty(*logits.shape[:-1], 0, dtype=torch.int32, device=logits.device)
    padding = -width % block_size
    scores = torch.nn.functional.pad(logits, (0, padding), value=float("-inf"))
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.shape[-1]
    if isinstance(compress_lens, torch.Tensor):
        last = (compress_lens.to(device=logits.device) - 1) // block_size
    else:
        last = (compress_lens - 1) // block_size
    block_positions = torch.arange(num_blocks, device=logits.device)
    scores = scores.masked_fill(block_positions == last, float("inf"))
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1, sorted=False)
    return top.indices.masked_fill(top.values == float("-inf"), -1).to(torch.int32)


def compressed_causal_candidates(
        query: torch.Tensor,
        key: torch.Tensor,
        merge_weight: torch.Tensor,
        *,
        compress_ratio: int,
        topk_blocks: int,
        block_size: int,
        query_offset: int = 0,
        query_chunk_size: int = 256,
        reduce_sum: Callable[[torch.Tensor], torch.Tensor] | None = None,
        minimum_key_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build compact blockwise candidate ids in bounded query chunks."""
    _, sequence_length, _, _ = query.shape
    compressed_length = key.shape[1]
    if minimum_key_indices is not None and minimum_key_indices.shape != query.shape[:2]:
        raise ValueError("minimum_key_indices must have shape [batch, query]")
    key_fp32 = key.float().transpose(1, 2).unsqueeze(1)
    key_positions = torch.arange(compressed_length, device=query.device).view(1, 1, -1)
    candidate_chunks = []
    for start in range(0, sequence_length, query_chunk_size):
        end = min(start + query_chunk_size, sequence_length)
        scores = torch.matmul(query[:, start:end].float(), key_fp32).relu()
        scores = (scores * merge_weight[:, start:end].float().unsqueeze(-1)).sum(dim=2)
        if reduce_sum is not None:
            scores = reduce_sum(scores)
        visible = (
            torch.arange(query_offset + start, query_offset + end, device=query.device) + 1
        ) // compress_ratio
        scores.masked_fill_(key_positions >= visible.view(1, -1, 1), float("-inf"))
        if minimum_key_indices is not None:
            minimum = minimum_key_indices[:, start:end].unsqueeze(-1)
            scores.masked_fill_(key_positions < minimum, float("-inf"))
        candidate_chunks.append(
            select_candidate_block_indices(
                scores,
                visible.view(1, -1, 1),
                topk_blocks,
                block_size,
            )
        )
    return torch.cat(candidate_chunks, dim=1)


def compressed_causal_topk_and_candidates(
        query: torch.Tensor,
        key: torch.Tensor,
        merge_weight: torch.Tensor,
        *,
        compress_ratio: int,
        sparse_count: int,
        topk_blocks: int,
        block_size: int,
        query_offset: int = 0,
        query_chunk_size: int = 256,
        reduce_sum: Callable[[torch.Tensor], torch.Tensor] | None = None,
        minimum_key_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select Full-layer Top-K and candidate blocks from one score pass."""
    if compress_ratio <= 0:
        raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
    if query_chunk_size <= 0:
        raise ValueError(f"query_chunk_size must be positive, got {query_chunk_size}")
    batch_size, sequence_length, _, head_dim = query.shape
    if key.ndim != 3 or key.shape[::2] != (batch_size, head_dim):
        raise ValueError(
            "compressed index key must have shape [batch, compressed_sequence, head_dim]"
        )
    if merge_weight.shape != query.shape[:3]:
        raise ValueError(
            f"merge_weight must have shape {tuple(query.shape[:3])}, got {tuple(merge_weight.shape)}"
        )
    compressed_length = key.shape[1]
    if minimum_key_indices is not None and minimum_key_indices.shape != query.shape[:2]:
        raise ValueError("minimum_key_indices must have shape [batch, query]")
    top_k = min(sparse_count, compressed_length)
    if top_k <= 0:
        empty_topk = torch.empty(
            batch_size,
            sequence_length,
            0,
            dtype=torch.int32,
            device=query.device,
        )
        return empty_topk, empty_topk

    key_fp32 = key.float().transpose(1, 2).unsqueeze(1)
    key_positions = torch.arange(compressed_length, device=query.device).view(1, 1, -1)
    selected_chunks = []
    candidate_chunks = []
    for start in range(0, sequence_length, query_chunk_size):
        end = min(start + query_chunk_size, sequence_length)
        scores = torch.matmul(query[:, start:end].float(), key_fp32).relu()
        scores = (scores * merge_weight[:, start:end].float().unsqueeze(-1)).sum(dim=2)
        if reduce_sum is not None:
            scores = reduce_sum(scores)
        visible = (
            torch.arange(query_offset + start, query_offset + end, device=query.device) + 1
        ) // compress_ratio
        scores.masked_fill_(key_positions >= visible.view(1, -1, 1), float("-inf"))
        if minimum_key_indices is not None:
            minimum = minimum_key_indices[:, start:end].unsqueeze(-1)
            scores.masked_fill_(key_positions < minimum, float("-inf"))
        top = scores.topk(top_k, dim=-1, sorted=False)
        indices = top.indices.masked_fill(~torch.isfinite(top.values), compressed_length)
        indices = indices.sort(dim=-1).values
        selected_chunks.append(indices.masked_fill(indices == compressed_length, -1).to(torch.int32))
        candidate_chunks.append(
            select_candidate_block_indices(
                scores,
                visible.view(1, -1, 1),
                topk_blocks,
                block_size,
            )
        )
    return torch.cat(selected_chunks, dim=1), torch.cat(candidate_chunks, dim=1)


def compressed_candidate_topk(
        query: torch.Tensor,
        key: torch.Tensor,
        merge_weight: torch.Tensor,
        candidate_blocks: torch.Tensor,
        *,
        compress_ratio: int,
        sparse_count: int,
        block_size: int,
        query_offset: int = 0,
        query_chunk_size: int = 256,
        reduce_sum: Callable[[torch.Tensor], torch.Tensor] | None = None,
        minimum_key_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score only hierarchical candidate blocks and return global key ids."""
    if compress_ratio <= 0:
        raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if query_chunk_size <= 0:
        raise ValueError(f"query_chunk_size must be positive, got {query_chunk_size}")
    batch_size, sequence_length, _, head_dim = query.shape
    if key.ndim != 3 or key.shape[::2] != (batch_size, head_dim):
        raise ValueError(
            "compressed index key must have shape [batch, compressed_sequence, head_dim]"
        )
    if merge_weight.shape != query.shape[:3]:
        raise ValueError(
            f"merge_weight must have shape {tuple(query.shape[:3])}, got {tuple(merge_weight.shape)}"
        )
    if candidate_blocks.ndim != 3 or candidate_blocks.shape[:2] != query.shape[:2]:
        raise ValueError(
            "candidate_blocks must have shape [batch, query, candidate_blocks], "
            f"got {tuple(candidate_blocks.shape)}"
        )
    if minimum_key_indices is not None and minimum_key_indices.shape != query.shape[:2]:
        raise ValueError("minimum_key_indices must have shape [batch, query]")

    compressed_length = key.shape[1]
    candidate_width = candidate_blocks.shape[-1] * block_size
    top_k = min(sparse_count, candidate_width)
    if top_k <= 0:
        return torch.empty(
            batch_size,
            sequence_length,
            0,
            dtype=torch.int32,
            device=query.device,
        )

    block_offsets = torch.arange(block_size, device=query.device)
    batch_indices = torch.arange(batch_size, device=query.device).view(-1, 1, 1)
    selected_chunks = []
    for start in range(0, sequence_length, query_chunk_size):
        end = min(start + query_chunk_size, sequence_length)
        blocks = candidate_blocks[:, start:end].long()
        block_positions = blocks.unsqueeze(-1) * block_size + block_offsets
        positions = block_positions.flatten(-2)
        visible = (
            torch.arange(query_offset + start, query_offset + end, device=query.device) + 1
        ) // compress_ratio
        valid = (blocks.unsqueeze(-1) >= 0).expand_as(block_positions).flatten(-2)
        valid = valid & (positions < compressed_length) & (positions < visible.view(1, -1, 1))
        if minimum_key_indices is not None:
            valid = valid & (positions >= minimum_key_indices[:, start:end].unsqueeze(-1))
        safe_positions = positions.clamp(min=0, max=max(compressed_length - 1, 0))
        selected_key = key[batch_indices, safe_positions].float()
        dots = torch.einsum(
            "bchd,bckd->bchk",
            query[:, start:end].float(),
            selected_key,
        ).relu()
        scores = (dots * merge_weight[:, start:end].float().unsqueeze(-1)).sum(dim=2)
        if reduce_sum is not None:
            scores = reduce_sum(scores)
        scores.masked_fill_(~valid, float("-inf"))
        top = scores.topk(top_k, dim=-1, sorted=False)
        selected = positions.gather(-1, top.indices)
        selected.masked_fill_(~torch.isfinite(top.values), compressed_length)
        selected = selected.sort(dim=-1).values
        selected.masked_fill_(selected == compressed_length, -1)
        selected_chunks.append(selected.to(torch.int32))
    return torch.cat(selected_chunks, dim=1)


class _SharedCompressedIndexerKLLoss(torch.autograd.Function):
    """PanGu-style precomputed-gradient KL loss for compressed index keys."""

    @staticmethod
    def forward(
            ctx: Any,
            index_query: torch.Tensor,
            index_key: torch.Tensor,
            merge_weight: torch.Tensor,
            attention_query: torch.Tensor,
            compressed_key: torch.Tensor,
            topk_indices: torch.Tensor,
            sinks: torch.Tensor,
            attention_scale: float,
            loss_coeff: float,
            query_chunk_size: int,
            reduce_sum: Callable[[torch.Tensor], torch.Tensor] | None,
    ) -> torch.Tensor:
        """Compute sparse KL and save only the three Indexer gradients."""
        batch_size, sequence_length, _, _ = index_query.shape
        denominator = batch_size * sequence_length
        total_loss = index_query.new_zeros((), dtype=torch.float32)
        grad_index_query = torch.zeros_like(index_query, dtype=torch.float32)
        grad_index_key = torch.zeros_like(index_key, dtype=torch.float32)
        grad_merge_weight = torch.zeros_like(merge_weight, dtype=torch.float32)

        with torch.no_grad(), torch.autocast(device_type=index_query.device.type, enabled=False):
            for start in range(0, sequence_length, query_chunk_size):
                end = min(start + query_chunk_size, sequence_length)
                selected = topk_indices[:, start:end]
                valid = selected >= 0
                valid_rows = valid.any(dim=-1)
                if not torch.any(valid_rows):
                    continue
                safe_indices = selected.clamp_min(0).long()

                query_chunk = index_query[:, start:end].float()
                weight_chunk = merge_weight[:, start:end].float()
                selected_index_key = gather_selected_keys(index_key, safe_indices).float()
                index_dots = torch.einsum(
                    "bcid,bckd->bcik", query_chunk, selected_index_key
                )
                index_relu = index_dots.relu()
                index_scores = (index_relu * weight_chunk.unsqueeze(-1)).sum(dim=2)
                if reduce_sum is not None:
                    index_scores = reduce_sum(index_scores)
                index_scores.masked_fill_(~valid, -1.0e9)

                selected_attention_key = gather_selected_keys(compressed_key, safe_indices).float()
                attention_scores = torch.einsum(
                    "bhcd,bckd->bhck",
                    attention_query[:, :, start:end].float(),
                    selected_attention_key,
                ) * attention_scale
                attention_scores.masked_fill_(~valid.unsqueeze(1), -1.0e9)
                sink_logits = sinks.float().view(1, -1, 1, 1).expand(
                    batch_size, -1, end - start, -1
                )
                target = torch.cat((attention_scores, sink_logits), dim=-1).softmax(dim=-1)
                target = target[..., :-1].masked_fill(~valid.unsqueeze(1), 0.0).sum(dim=1)
                if reduce_sum is not None:
                    target = reduce_sum(target)
                target = target / target.sum(dim=-1, keepdim=True).clamp_min(
                    torch.finfo(torch.float32).tiny
                )

                log_prediction = index_scores.log_softmax(dim=-1)
                target_log = target.clamp_min(torch.finfo(torch.float32).tiny).log()
                row_loss = (target * (target_log - log_prediction)).sum(dim=-1)
                total_loss.add_(row_loss[valid_rows].sum() * (loss_coeff / denominator))

                grad_scores = log_prediction.exp() * target.sum(-1, keepdim=True) - target
                grad_scores.mul_(loss_coeff / denominator)
                grad_scores.masked_fill_(~valid, 0.0)
                grad_dots = (
                    grad_scores.unsqueeze(2)
                    * weight_chunk.unsqueeze(-1)
                    * (index_dots > 0).to(index_dots.dtype)
                )
                grad_index_query[:, start:end] = torch.einsum(
                    "bcik,bckd->bcid", grad_dots, selected_index_key
                )
                grad_merge_weight[:, start:end] = (
                    grad_scores.unsqueeze(2) * index_relu
                ).sum(dim=-1)
                selected_key_grad = torch.einsum(
                    "bcik,bcid->bckd", grad_dots, query_chunk
                )
                # Preserve the einsum result if SAC cached its storage.
                selected_key_grad = selected_key_grad.masked_fill(~valid.unsqueeze(-1), 0.0)
                scatter_indices = safe_indices.unsqueeze(-1).expand_as(selected_key_grad)
                grad_index_key.scatter_add_(
                    1,
                    scatter_indices.reshape(batch_size, -1, index_key.shape[-1]),
                    selected_key_grad.reshape(batch_size, -1, index_key.shape[-1]),
                )

        ctx.save_for_backward(grad_index_query, grad_index_key, grad_merge_weight)
        return total_loss

    @staticmethod
    def backward(ctx: Any, grad_loss: torch.Tensor) -> tuple:
        """Return the precomputed Indexer gradients and detach the teacher."""
        grad_index_query, grad_index_key, grad_merge_weight = ctx.saved_tensors
        return (
            grad_index_query * grad_loss,
            grad_index_key * grad_loss,
            grad_merge_weight * grad_loss,
            *((None,) * 8),
        )


def shared_compressed_indexer_kl_loss(
        index_query: torch.Tensor,
        index_key: torch.Tensor,
        merge_weight: torch.Tensor,
        attention_query: torch.Tensor,
        compressed_key: torch.Tensor,
        topk_indices: torch.Tensor,
        sinks: torch.Tensor,
        *,
        attention_scale: float,
        loss_coeff: float,
        query_chunk_size: int = 256,
        tp_context: SharedCompressedAttentionTPContext | None = None,
        compress_ratio: int = 1,
        query_segments: tuple[tuple[int, int, int, int, int], ...] | None = None,
        use_fused: bool = False,
) -> torch.Tensor:
    """Compute the sparse-stage DSA KL objective on V4.1 compressed keys.

    The target first softmaxes selected main-attention scores per attention
    head, then averages and L1-normalizes across heads. Indexer inputs are
    detached by the caller, so this objective updates only Indexer parameters.

    ``use_fused`` is valid only for ordinary causal TopK selections, without
    candidate-pool holes. ``query_segments`` supplies each document-local K
    prefix; its ratio and residual encode the CP query offset for the kernel.
    ``use_fused=True`` raises for unsupported calls when ``loss_coeff != 0``;
    False explicitly selects the numerical reference. Native errors propagate.
    """
    if query_chunk_size <= 0:
        raise ValueError(f"query_chunk_size must be positive, got {query_chunk_size}")
    if not loss_coeff:
        return index_query.sum(dtype=torch.float32) * 0.0
    if index_key.shape[1] == 0:
        return (index_query[..., :0].sum(dtype=torch.float32) + index_key.sum(dtype=torch.float32)
                + merge_weight[..., :0].sum(dtype=torch.float32)) * 0.0
    if topk_indices.shape[:2] != index_query.shape[:2]:
        raise ValueError("topk_indices and index_query must share batch/query dimensions")
    if use_fused:
        if tp_context is not None:
            raise NotImplementedError("Fused KL does not support TP score/teacher reduction")
        if query_segments is None:
            raise ValueError("Fused KL requires document-local causal query_segments")
        validate_fused_kl_inputs(index_query, index_key, merge_weight, topk_indices)
        return sparse_lightning_indexer_kl_loss(
            index_query, index_key, merge_weight, attention_query.detach(), compressed_key.detach(),
            topk_indices, sinks.detach(), attention_scale, loss_coeff, query_chunk_size,
            compress_ratio, query_segments,
        )
    return _SharedCompressedIndexerKLLoss.apply(
        index_query,
        index_key,
        merge_weight,
        attention_query.detach(),
        compressed_key.detach(),
        topk_indices,
        sinks.detach(),
        attention_scale,
        loss_coeff,
        query_chunk_size,
        None if tp_context is None else tp_context.reduce_sum,
    )


class SharedCompressedDSAIndexer(nn.Module):
    """Reusable V4.1 compressed Lightning Indexer semantics."""

    # Installed by :meth:`enable_selection_exclude` for a layer that recomputes.
    # It is written through ``object.__setattr__`` so the exclusion region stays
    # out of ``modules()`` and the state dict.
    _selection_region: Callable[..., Any] | None = None

    def __init__(self, module: nn.Module) -> None:
        """Transfer the source projections without changing checkpoint names."""
        super().__init__()
        required = ("q_b_proj", "weights_proj")
        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            raise TypeError(f"shared compressed DSA indexer is missing: {missing}")
        for name, child in module._modules.items():  # pylint: disable=protected-access
            self.add_module(name, child)
        for name, parameter in module._parameters.items():  # pylint: disable=protected-access
            self.register_parameter(name, parameter)
        self.use_fused = False
        self.compress_ratio = module.compress_ratio
        self.global_num_heads = module.num_heads
        self.head_dim = module.head_dim
        self.index_topk = module.index_topk
        self.owns_key = module.owns_key
        self.is_candidate_source = module.is_candidate_source
        self.uses_candidates = module.uses_candidates
        self.candidate_topk_blocks = module.candidate_topk_blocks
        self.candidate_block_size = module.candidate_block_size
        self.loss_coeff = module.loss_coeff
        self.query_chunk_size = int(getattr(module, "query_chunk_size", 256))
        self.train(module.training)

    def _project_index_query(
            self, query_residual: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    ) -> torch.Tensor:
        """Project queries with their original rotary positions, including moved tasks."""
        query = self.q_b_proj(query_residual.detach())
        if query.shape[-1] % self.head_dim:
            raise ValueError(
                f"Indexer query width {query.shape[-1]} is not divisible by head_dim {self.head_dim}"
            )
        query = query.view(*query_residual.shape[:2], -1, self.head_dim)
        return _apply_v41_rope(query.transpose(1, 2), cos, sin).transpose(1, 2)

    def _project_index_key(
            self,
            latent: torch.Tensor | None,
            cos: torch.Tensor,
            sin: torch.Tensor,
            index_key: torch.Tensor | None,
            cp_context: SharedCompressedAttentionCPContext | None,
            key_plan: IndexKeyPlan | None,
            block_plan: Any = None,
            key_position_embeddings: tuple | None = None,
    ) -> tuple[torch.Tensor, "SequenceGatherHandle | None"]:
        """Project or adopt Index K and launch its configured differentiable exchange."""
        if self.owns_key:
            if latent is None:
                raise RuntimeError("a Full Indexer requires the compressor latent")
            key = self.k_norm(self.wk(latent.detach()))
            compressed_length = key.shape[1]
            if key_position_embeddings is None:
                key_cos = cos[:, :compressed_length * self.compress_ratio:self.compress_ratio]
                key_sin = sin[:, :compressed_length * self.compress_ratio:self.compress_ratio]
            else:
                key_cos, key_sin = key_position_embeddings
            key = _apply_v41_rope(key.unsqueeze(1), key_cos, key_sin).squeeze(1)
            if key_plan is not None:
                key_handle = cp_context.launch_index(key, 1, key_plan)
            elif cp_context is not None and block_plan is not None:
                key_handle = block_plan.launch_bank(key, cp_context)
            else:
                key_handle = cp_context.launch(key, 1) if cp_context is not None else None
            return key, key_handle
        if index_key is None:
            raise RuntimeError("a Reindex layer requires index K from the preceding Full layer")
        return index_key, None

    def _resolve_key_plan(
            self, cp_context: SharedCompressedAttentionCPContext | None, lengths: tuple,
            source_plan: IndexKeyPlan | None,
    ) -> IndexKeyPlan | None:
        """Require one communication policy for all consumers of a shared bank."""
        plan = None
        if cp_context is not None and cp_context.plan_index_key is not None:
            if cp_context.launch_index is None:
                raise ValueError("Index K prefix planning requires its differentiable dispatch callback")
            if self.is_candidate_source or self.uses_candidates:
                raise NotImplementedError("Index K prefix dispatch currently requires ordinary causal selection")
            plan = cp_context.plan_index_key(lengths, self.compress_ratio)
        if not self.owns_key and plan != source_plan:
            raise ValueError(
                "Full/Reindex consumers of one Index K bank must share compression and CP dispatch settings"
            )
        return plan

    def forward(
            self,
            hidden_states: torch.Tensor,
            query_residual: torch.Tensor,
            latent: torch.Tensor | None,
            compress_position_embeddings: tuple[torch.Tensor, torch.Tensor],
            *,
            index_key: torch.Tensor | None = None,
            candidate_blocks: torch.Tensor | None = None,
            cp_context: SharedCompressedAttentionCPContext | None = None,
            tp_context: SharedCompressedAttentionTPContext | None = None,
            query_offset: int = 0,
            minimum_key_indices: torch.Tensor | None = None,
            query_segments: tuple[tuple[int, int, int, int, int], ...] | None = None,
            document_lengths: tuple[int, ...] = (),
            index_key_plan: IndexKeyPlan | None = None,
            block_plan: Any = None,
            key_position_embeddings: tuple | None = None,
    ) -> SharedCompressedIndexerOutput:
        """Project local queries and score them against global compressed keys.

        When the enclosing layer is recomputed, :meth:`enable_selection_exclude`
        makes the discrete selection a checkpoint exclusion region: it runs in
        the original forward and its cached result is reused during the
        recomputation, so the chosen key set can never drift and the checkpoint
        keeps saving the same tensors in both passes.
        """
        cos, sin = compress_position_embeddings
        if (self.is_candidate_source or self.uses_candidates) and any(
                length % self.compress_ratio for length in document_lengths):
            raise NotImplementedError("Natural document compression requires ordinary causal Indexer selection")

        key_plan = self._resolve_key_plan(cp_context, document_lengths, index_key_plan)
        key, key_handle = self._project_index_key(
            latent, cos, sin, index_key, cp_context, key_plan, block_plan, key_position_embeddings,
        )

        query = self._project_index_query(query_residual, cos, sin)
        local_num_heads = query.shape[2]
        merge_weight = self.weights_proj(hidden_states.detach())
        if merge_weight.shape[-1] != local_num_heads:
            raise ValueError(
                "Indexer q_b_proj and weights_proj must use the same TP head shard: "
                f"{local_num_heads} query heads versus {merge_weight.shape[-1]} weights"
            )
        merge_weight = merge_weight * (self.head_dim**-0.5 * self.global_num_heads**-0.5)
        reduce_sum = None if tp_context is None else tp_context.reduce_sum
        if key_handle is not None:
            key = key_handle.wait()
        topk_indices, candidate_blocks = self._run_selection(
            query, key, merge_weight, query_residual, (cos, sin), cp_context, tp_context,
            candidate_blocks, query_offset, reduce_sum, minimum_key_indices, query_segments,
            document_lengths, key_plan,
        )
        return SharedCompressedIndexerOutput(
            topk_indices=topk_indices,
            index_key=key,
            candidate_blocks=candidate_blocks,
            index_query=query,
            merge_weight=merge_weight,
            key_plan=key_plan,
        )

    def enable_selection_exclude(self) -> None:
        """Keep discrete selection fixed when the enclosing layer recomputes."""
        object.__setattr__(
            self, "_selection_region", checkpoint_exclude_wrapper(self._select_tokens)
        )

    def _run_selection(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Select directly, or reuse the result retained by the checkpoint region."""
        selection = self._select_tokens if self._selection_region is None else self._selection_region
        return selection(*args, **kwargs)

    def _select_tokens(
            self, query: torch.Tensor, key: torch.Tensor, merge_weight: torch.Tensor,
            query_residual: torch.Tensor, embeddings: tuple,
            cp_context: SharedCompressedAttentionCPContext | None,
            tp_context: SharedCompressedAttentionTPContext | None,
            candidate_blocks: torch.Tensor | None, query_offset: int,
            reduce_sum: Callable[[torch.Tensor], torch.Tensor] | None,
            minimum_key_indices: torch.Tensor | None, query_segments: tuple | None,
            document_lengths: tuple, key_plan: IndexKeyPlan | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Cache dispatch, discrete selection and TopK return as one recompute region."""
        cos, sin = embeddings
        global_keys = sum(length // self.compress_ratio for length in document_lengths)
        if document_lengths and not global_keys:
            topk_indices = torch.empty((*query.shape[:2], 0), device=query.device, dtype=torch.long)
            candidate_blocks = None
        elif cp_context is not None and cp_context.balance_indexer is not None:
            if tp_context is not None or self.is_candidate_source or self.uses_candidates:
                raise NotImplementedError("Indexer load balance currently requires ordinary causal selection and TP=1")

            def select_tasks(task_qr: torch.Tensor, task_weights: torch.Tensor,
                             task_embeddings: tuple, segments: tuple) -> torch.Tensor:
                """Project on the receiver while this layer's parameters are resident."""
                task_query = self._project_index_query(task_qr, *task_embeddings)
                return self._select_indices(
                    task_query, key, task_weights, None, 0, None, None, segments, key_plan,
                )[0]

            topk_indices = cp_context.balance_indexer(
                query_residual, merge_weight, (cos, sin), select_tasks, document_lengths, self.compress_ratio,
            )
            candidate_blocks = None
        else:
            topk_indices, candidate_blocks = self._select_indices(
                query, key, merge_weight, candidate_blocks, query_offset,
                reduce_sum, minimum_key_indices, query_segments, key_plan,
            )
        return topk_indices, candidate_blocks

    def _select_indices(
            self, query: torch.Tensor, key: torch.Tensor, merge_weight: torch.Tensor,
            candidate_blocks: torch.Tensor | None, query_offset: int,
            reduce_sum: Callable[[torch.Tensor], torch.Tensor] | None,
            minimum_key_indices: torch.Tensor | None, query_segments: tuple | None,
            key_plan: IndexKeyPlan | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Select the requested implementation without changing candidate policies."""
        if self.use_fused and (self.is_candidate_source or self.uses_candidates):
            raise NotImplementedError("V4.1 fused Indexer does not support candidate-pool selection")
        if self.is_candidate_source:
            topk_indices, candidate_blocks = compressed_causal_topk_and_candidates(
                query,
                key,
                merge_weight,
                compress_ratio=self.compress_ratio,
                sparse_count=self.index_topk,
                topk_blocks=self.candidate_topk_blocks,
                block_size=self.candidate_block_size,
                query_offset=query_offset,
                query_chunk_size=self.query_chunk_size,
                reduce_sum=reduce_sum,
                minimum_key_indices=minimum_key_indices,
            )
        elif self.uses_candidates:
            if candidate_blocks is None:
                raise RuntimeError("a hierarchical Reindex layer requires candidate block ids")
            topk_indices = compressed_candidate_topk(
                query,
                key,
                merge_weight,
                candidate_blocks,
                compress_ratio=self.compress_ratio,
                sparse_count=self.index_topk,
                block_size=self.candidate_block_size,
                query_offset=query_offset,
                query_chunk_size=self.query_chunk_size,
                reduce_sum=reduce_sum,
                minimum_key_indices=minimum_key_indices,
            )
        else:
            candidate_blocks = None
            topk_indices = compressed_causal_topk(
                query,
                key,
                merge_weight,
                compress_ratio=self.compress_ratio,
                sparse_count=self.index_topk,
                query_offset=query_offset,
                query_chunk_size=self.query_chunk_size,
                reduce_sum=reduce_sum,
                minimum_key_indices=minimum_key_indices,
                query_segments=query_segments if key_plan is None else key_plan.map_segments(query_segments),
                use_fused=self.use_fused,
                global_key_length=None if key_plan is None else key_plan.global_length,
                key_global_positions=(None if key_plan is None or self.use_fused
                                      else key_plan.global_positions(key.device)),
            )
        if key_plan is not None and self.use_fused:
            topk_indices = key_plan.global_indices(topk_indices, query_segments)
        return topk_indices, candidate_blocks


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


class SharedCompressedDSAAttentionBase(nn.Module):
    """Shared V4.1 attention semantics with selectable reference execution."""

    def __init__(
            self,
            module: nn.Module,
            *,
            replace_indexer: bool,
            use_fused_ops: bool,
    ) -> None:
        """Transfer source state and configure the execution implementation."""
        super().__init__()
        for name, child in module._modules.items():  # pylint: disable=protected-access
            self.add_module(name, child)
        for name, parameter in module._parameters.items():  # pylint: disable=protected-access
            self.register_parameter(name, parameter)
        indexer = getattr(self, "indexer", None)
        if (
                replace_indexer
                and indexer is not None
                and not isinstance(indexer, SharedCompressedDSAIndexer)
        ):
            self.indexer = SharedCompressedDSAIndexer(indexer)
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
        self.train(module.training)

    @staticmethod
    def _validate_cp_inputs(
            cp_context: SharedCompressedAttentionCPContext,
            position_ids: torch.Tensor,
            sequence_length: int,
    ) -> tuple[int, int]:
        """Validate contiguous CP shards and return global query geometry."""
        if cp_context.size <= 1 or not 0 <= cp_context.rank < cp_context.size:
            raise ValueError(
                "shared compressed attention CP requires size > 1 and "
                f"0 <= rank < size, got rank={cp_context.rank}, size={cp_context.size}"
            )
        query_offset = cp_context.rank * sequence_length
        if not torch.all(position_ids[..., 0] == query_offset):
            raise ValueError(
                "shared compressed attention CP requires contiguous global position_ids; "
                f"rank {cp_context.rank} expected first position {query_offset}"
            )
        return query_offset, sequence_length * cp_context.size

    def _project_raw_kv(
            self,
            hidden_states: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            cp_context: SharedCompressedAttentionCPContext | None,
            raw_route: SequenceDispatchRoute | None = None,
    ) -> tuple[torch.Tensor | None, SequenceGatherHandle | None]:
        """Project raw K=V and launch CP communication before query GEMMs."""
        key_value = self.kv_norm(self.kv_proj(hidden_states)).unsqueeze(1)
        key_value = _apply_v41_rope(key_value, cos, sin)
        if cp_context is None:
            return key_value, None
        if raw_route is not None:
            return None, cp_context.launch_raw(key_value, 2, raw_route)
        return None, cp_context.launch(key_value, 2)

    def _project_query(
            self,
            hidden_states: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project and normalize the local query heads."""
        batch_size, sequence_length, _ = hidden_states.shape
        query_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        query = self.q_b_proj(query_residual).view(batch_size, sequence_length, -1, self.head_dim)
        query = query.transpose(1, 2)
        return query_residual, _apply_v41_rope(query, cos, sin)

    @staticmethod
    def _validate_forward_inputs(
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None,
            past_key_values: Any | None,
            kwargs: dict[str, Any],
    ) -> tuple[
        "SharedCompressedAttentionState",
        "SharedCompressedPackedSequence | None",
        "SharedCompressedAttentionCPContext | None",
        "SharedCompressedAttentionTPContext | None",
        int,
        int,
    ]:
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
        cp_context = kwargs.pop("shared_attention_cp_context", None)
        if cp_context is not None and not isinstance(cp_context, SharedCompressedAttentionCPContext):
            raise TypeError(
                "shared_attention_cp_context must be SharedCompressedAttentionCPContext, "
                f"got {type(cp_context).__name__}"
            )
        tp_context = kwargs.pop("shared_attention_tp_context", None)
        if tp_context is not None and not isinstance(tp_context, SharedCompressedAttentionTPContext):
            raise TypeError(
                "shared_attention_tp_context must be SharedCompressedAttentionTPContext, "
                f"got {type(tp_context).__name__}"
            )
        return shared_state, packed_sequence, cp_context, tp_context, batch_size, sequence_length

    def _resolve_packed_segments(
            self,
            packed_sequence: "SharedCompressedPackedSequence | None",
            batch_size: int,
            sequence_length: int,
            query_offset: int,
            global_sequence_length: int,
            device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, tuple]:
        """Validate packed document boundaries and derive local Q/K segments."""
        segment_starts = None
        minimum_key_indices = None
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
            packed_sequence = packed_sequence.prepare(device, (self.compress_ratio,))
            segment_starts = packed_sequence.local_segment_starts(device)
            if self.compress_ratio:
                minimum_key_indices = packed_sequence.minimum_key_indices(device, self.compress_ratio)
            query_segments = packed_sequence.indexer_segments(ratio)
        return segment_starts, minimum_key_indices, query_segments

    def _plan_bank_dispatch(
            self, packed_sequence: SharedCompressedPackedSequence | None,
            cp_context: SharedCompressedAttentionCPContext | None,
            sequence_length: int, global_sequence_length: int, device: torch.device,
    ) -> tuple[SequenceDispatchRoute | None, SequenceDispatchRoute | None, tuple[int, ...]]:
        """Reuse document boundaries to route each bank in its own physical coordinates."""
        boundaries = (0, global_sequence_length)
        if packed_sequence is not None:
            boundaries = packed_sequence.prepare(device, (self.compress_ratio,)).boundaries()
        raw_route, main_route = None, None
        if cp_context is not None and cp_context.launch_raw is not None:
            raw_route = sliding_window_route(
                boundaries, sequence_length, cp_context.size, cp_context.rank, self.sliding_window,
            )
        if self.compress_ratio and cp_context is not None and cp_context.launch_main is not None:
            main_route = document_prefix_route(
                boundaries, sequence_length, cp_context.size, cp_context.rank, self.compress_ratio,
            )
        return raw_route, main_route, tuple(end - start for start, end in zip(boundaries, boundaries[1:]))

    def _prepare_compression(
            self, packed: SharedCompressedPackedSequence | None, context: SharedCompressedAttentionCPContext | None,
            embeddings: dict, lengths: tuple, device: torch.device,
    ) -> tuple:
        """Prepare block-start positions only for a source that needs document-local grouping."""
        if not self.is_kv_source or self.compress_ratio <= 1:
            return None, None
        size, rank = (1, 0) if context is None else (context.size, context.rank)
        if packed is None:
            if (sum(lengths)//size) % self.compress_ratio == 0:
                return None, None
            packed = SharedCompressedPackedSequence(torch.tensor((0, sum(lengths))),
                                                     rank*(sum(lengths)//size), sum(lengths)//size, sum(lengths))
        packed = packed.prepare(device, (self.compress_ratio,))
        plan = packed.block_plan(self.compress_ratio, size, rank)
        positions = None if plan is None else packed.compressed_positions(plan, embeddings["compress"], context)
        return plan, positions

    def _compute_compressed_kv(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]],
            shared_state: "SharedCompressedAttentionState",
            cp_context: "SharedCompressedAttentionCPContext | None",
            main_route: SequenceDispatchRoute | None = None,
            block_plan: Any = None,
            key_position_embeddings: tuple | None = None,
    ) -> tuple[torch.Tensor | None, "SequenceGatherHandle | None"]:
        """Compress local hidden states and publish or launch the compressed KV.

        The compressed K=V reaches its consumers - the producing Full layer
        included - through the shared state, so only the publish (or the CP
        gather handle that feeds it) leaves this call.
        """
        latent = None
        compressed_handle = None
        if self.is_kv_source:
            if block_plan is None:
                latent, compressed = self.compressor(hidden_states, position_embeddings["compress"])
            else:
                latent, compressed = self.compressor(hidden_states, key_position_embeddings, block_plan, cp_context)
            if cp_context is None:
                if not is_recomputing():
                    shared_state.publish_compressed_kv(self.layer_idx, compressed)
            elif main_route is not None:
                compressed_handle = cp_context.launch_main(compressed, 1, main_route)
            elif block_plan is not None:
                compressed_handle = block_plan.launch_bank(compressed, cp_context)
            else:
                compressed_handle = cp_context.launch(compressed, 1)
        return latent, compressed_handle

    def _run_indexer(
            self,
            hidden_states: torch.Tensor,
            query_residual: torch.Tensor,
            latent: torch.Tensor | None,
            position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]],
            shared_state: "SharedCompressedAttentionState",
            cp_context: "SharedCompressedAttentionCPContext | None",
            tp_context: "SharedCompressedAttentionTPContext | None",
            query_offset: int,
            minimum_key_indices: torch.Tensor | None,
            query_segments: tuple,
            document_lengths: tuple[int, ...],
            block_plan: Any = None,
            key_position_embeddings: tuple | None = None,
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
            index_key=index_key,
            candidate_blocks=candidate_blocks,
            cp_context=cp_context,
            tp_context=tp_context,
            query_offset=query_offset,
            minimum_key_indices=minimum_key_indices,
            query_segments=query_segments,
            document_lengths=document_lengths,
            block_plan=block_plan,
            key_position_embeddings=key_position_embeddings,
            index_key_plan=shared_state.index_key_plans.get(self.kv_source_layer_idx),
        )
        if is_recomputing():
            # A recompute must not publish again: its consumers read the entries
            # the forward pass wrote, and a second write would break the
            # single-publisher contract.
            return indexer_output
        if self.indexer.owns_key:
            shared_state.publish_index_key(self.layer_idx, indexer_output.index_key, indexer_output.key_plan)
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
            tp_context: "SharedCompressedAttentionTPContext | None",
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
            if indexer_output.key_plan is not None:
                plan = indexer_output.key_plan
                owner_start, owner_end, _ = plan.owner_range
                offset = owner_start - main_key_offset
                if offset < 0 or owner_end - main_key_offset > compressed_kv.shape[1]:
                    raise ValueError("Main KV bank does not cover the original owner's KL prefix")
                compressed_kv = compressed_kv.narrow(1, offset, owner_end - owner_start)
                index_key = plan.owner_view(index_key)
                main_key_offset = owner_start
            elif main_key_offset or compressed_kv.shape[1] != index_key.shape[1]:
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
                tp_context=tp_context,
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
                        query, self.compress_ratio if expected_topk else 0, self.sliding_window,
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
            if compressed_kv is not None and not any(ke > ks for _, _, ks, ke, _ in query_segments):
                # Empty consumers must still return zero gradients through the shared bank collective.
                output = output + compressed_kv[:, :0].sum()
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
        attention_output = _apply_v41_rope(attention_output.transpose(1, 2), cos, -sin).transpose(1, 2)
        grouped = attention_output.reshape(batch_size, sequence_length, self.num_groups, -1)
        hidden_per_group = grouped.shape[-1]
        output_per_group = self.o_a_proj.weight.shape[0] // self.num_groups
        projection = self.o_a_proj.weight.view(
            self.num_groups, output_per_group, hidden_per_group
        ).transpose(1, 2)
        flattened = grouped.reshape(-1, self.num_groups, hidden_per_group).transpose(0, 1)
        projected = torch.bmm(flattened, projection).transpose(0, 1)
        projected = projected.reshape(batch_size, sequence_length, -1)
        return self.o_b_proj(projected)

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]],
            position_ids: torch.Tensor,
            attention_mask: torch.Tensor | None,
            past_key_values: Any | None = None,
            **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Publish or consume compressed KV and execute indexed K=V attention."""
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
                cp_context,
                position_ids,
                sequence_length,
            )
        if packed_sequence is not None:
            packed_sequence = packed_sequence.prepare(hidden_states.device, (self.compress_ratio,))
        segment_starts, minimum_key_indices, query_segments = self._resolve_packed_segments(
            packed_sequence,
            batch_size,
            sequence_length,
            query_offset,
            global_sequence_length,
            hidden_states.device,
        )

        raw_route, main_route, document_lengths = self._plan_bank_dispatch(
            packed_sequence, cp_context, sequence_length, global_sequence_length, hidden_states.device,
        )
        block_plan, key_positions = self._prepare_compression(
            packed_sequence, cp_context, position_embeddings, document_lengths, hidden_states.device,
        )
        raw_document_starts = () if packed_sequence is None else packed_sequence.raw_document_starts()
        raw_key_offset = 0 if raw_route is None else raw_route.global_start
        rope_type = "compress" if self.compress_ratio else "main"
        cos, sin = position_embeddings[rope_type]
        key_value, raw_kv_handle = self._project_raw_kv(hidden_states, cos, sin, cp_context, raw_route)
        query_residual, query = self._project_query(hidden_states, cos, sin)
        latent, compressed_handle = self._compute_compressed_kv(
            hidden_states, position_embeddings, shared_state, cp_context, main_route, block_plan, key_positions,
        )
        indexer_output = self._run_indexer(
            hidden_states,
            query_residual,
            latent,
            position_embeddings,
            shared_state,
            cp_context,
            tp_context,
            query_offset,
            minimum_key_indices,
            query_segments,
            document_lengths,
            block_plan,
            key_positions,
        )

        if raw_kv_handle is not None:
            key_value = raw_kv_handle.wait()
        if compressed_handle is not None:
            compressed = compressed_handle.wait()
            if not is_recomputing():
                shared_state.publish_compressed_kv(
                    self.layer_idx, compressed, 0 if main_route is None else main_route.global_start,
                )
        if key_value is None:
            raise RuntimeError("raw KV communication did not produce a tensor")

        compressed_kv = None
        main_key_offset = 0
        topk_indices = None
        if self.compress_ratio:
            compressed_kv = shared_state.require_compressed_kv(self.kv_source_layer_idx, self.layer_idx)
            topk_indices = shared_state.require_topk_indices(self.index_source_layer_idx, self.layer_idx)
            main_key_offset = shared_state.compressed_kv_offsets.get(self.kv_source_layer_idx, 0)
        query = self._apply_indexer_loss(
            indexer_output, query, compressed_kv, tp_context, query_segments, main_key_offset,
        )
        attention_output = self._run_sparse_attention(
            query, key_value, compressed_kv, topk_indices, query_offset, global_sequence_length,
            segment_starts, query_segments, raw_key_offset, main_key_offset, raw_document_starts,
            sum(length // max(1, self.compress_ratio) for length in document_lengths),
        )
        output = self._project_output(attention_output, cos, sin, batch_size, sequence_length)
        return output, None


@module_replacement
class SharedCompressedDSAAttention(SharedCompressedDSAAttentionBase):
    """High-performance V4.1 shared compressed attention replacement."""

    def __init__(
        self,
        module: nn.Module,
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
        *,
        use_fused_ops: bool = False,
        swa_comm: str = "all_to_all_v",
        indexer_load_balance: str = "none",
        main_kv_comm: str = "all_gather",
        indexer_balance_block_size: int | None = None,
        index_k_comm: str = "all_gather",
        indexer_zigzag_min_length: int | None = None,
        indexer_zigzag_min_chunk_size: int | None = None,
    ) -> None:
        """Transfer source state and select the implementation used for comparisons.

        Args:
            module: Source attention module exposing the shared-compressed DSA
                structural contract.
            module_fqn: Fully qualified source name supplied by replacement.
            context: Read-only replacement context supplied by Trainer.
            use_fused_ops: Use LI V2, SMLA and fused KL when True,
                raising on unsupported calls without automatic fallback.
                False selects eager PyTorch attention, Indexer and KL on every
                device, without loading optional operator extensions.
            swa_comm: Raw KV communication, 'all_to_all_v' exchanges only the SWA
                halo; 'all_gather' keeps the full-bank comparison.
            indexer_load_balance: 'none', 'zigzag', 'greedy' or 'hybrid' query-task allocation.
                Selection exchanges low-rank qr, merge weights and rotary embeddings;
                attention and KL retain the original owner and autograd graph.
            indexer_balance_block_size: Maximum original query rows per greedy task,
                before adjacent tasks on one worker are merged. Greedy/hybrid default to 2048.
                Explicit values require greedy/hybrid and must be positive and no larger
                than local Q. Omit this option for none/zigzag.
            indexer_zigzag_min_length: Explicit hybrid-only minimum document length in raw tokens.
            indexer_zigzag_min_chunk_size: Explicit hybrid-only minimum floor(document length / (2*CP)).
                Both thresholds must pass. Eligible head/tail fragments stay intact;
                leading remainders and other documents share the greedy pool and row budget.
            main_kv_comm: 'all_to_all_v' dispatches document-causal main KV prefixes;
                'all_gather' retains the full-bank comparison.
            index_k_comm: 'all_to_all_v' exchanges the union of worker-LI and owner-KL
                document prefixes. 'all_gather' retains the global bank. All Full/Reindex
                consumers of a shared source must use the same CP/balance settings.

        Note:
            All implementations retain the same parameter and CP/TP contracts.
            Selecting native preserves the existing external teacher and scalar
            KL. Reuse and disabled KL do not introduce extra operator calls.
            Fusion supports ordinary causal r1/r2 training on NPU with BF16
            main attention, D=512, SWA=128, Indexer D=128, TopK=512 and TP=1.
            Active fused KL requires 8, 16, 32 or 64 Indexer heads. Candidate
            pools are unsupported. Packed documents are compressed independently;
            CP still requires equal contiguous raw query shards and batch one.
            Multiple document segments use a native TND call.
            The ops-transformer training extension is loaded on the first
            supported fused call; a missing extension raises ImportError.
        """
        if not isinstance(use_fused_ops, bool):
            raise ValueError(f"use_fused_ops must be a bool, got {use_fused_ops!r}")
        if swa_comm not in ("all_gather", "all_to_all_v"):
            raise ValueError("swa_comm must be 'all_gather' or 'all_to_all_v'")
        if main_kv_comm not in ("all_gather", "all_to_all_v"):
            raise ValueError("main_kv_comm must be 'all_gather' or 'all_to_all_v'")
        if index_k_comm not in ("all_gather", "all_to_all_v"):
            raise ValueError("index_k_comm must be 'all_gather' or 'all_to_all_v'")
        indexer_balance_block_size = resolve_indexer_balance_block_size(
            indexer_load_balance, indexer_balance_block_size,
        )
        validate_indexer_hybrid_thresholds(
            indexer_load_balance, indexer_zigzag_min_length, indexer_zigzag_min_chunk_size,
        )
        del module_fqn, context
        super().__init__(
            module,
            replace_indexer=True,
            use_fused_ops=use_fused_ops,
        )
        self.swa_comm = swa_comm
        self.indexer_load_balance = indexer_load_balance
        self.indexer_balance_block_size = indexer_balance_block_size
        self.indexer_zigzag_min_length = indexer_zigzag_min_length
        self.indexer_zigzag_min_chunk_size = indexer_zigzag_min_chunk_size
        self.main_kv_comm = main_kv_comm
        self.index_k_comm = index_k_comm


__all__ = [
    "SharedCompressedAttentionCPContext",
    "SharedCompressedAttentionState",
    "SharedCompressedPackedSequence",
    "SharedCompressedAttentionTPContext",
    "SharedCompressedDSAAttention",
    "SharedCompressedDSAAttentionBase",
    "SharedCompressedDSAIndexer",
    "build_sliding_window_indices",
    "compressed_candidate_topk",
    "compressed_causal_topk_and_candidates",
    "compressed_causal_topk",
    "select_candidate_block_indices",
    "select_candidate_blocks",
    "shared_compressed_indexer_kl_loss",
    "reference_sparse_attention",
]
