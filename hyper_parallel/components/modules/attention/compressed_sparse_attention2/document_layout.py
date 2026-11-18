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
"""Document boundaries and packed CSA2 coordinates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from functools import lru_cache

import torch
from torch.utils._pytree import register_pytree_node



@dataclass(frozen=True)
class DocumentBlockPlan:
    """Complete compression blocks in one local document bank."""

    ratio: int
    blocks: tuple[tuple[int, int], ...]

    def indices(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return pooling rows and original block-start positions."""
        return _document_block_indices(self, device)


@lru_cache(maxsize=8)
@torch.inference_mode(False)
def _document_block_indices(plan: DocumentBlockPlan, device: torch.device) -> tuple:
    starts = [torch.arange(start, start + count * plan.ratio, plan.ratio, device=device)
              for start, count in plan.blocks]
    starts = torch.cat(starts) if starts else torch.empty(0, device=device, dtype=torch.long)
    rows = (starts.unsqueeze(1) + torch.arange(plan.ratio, device=device)).flatten()
    return rows.long(), starts.long()


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
    _prepared_bounds: dict[tuple[torch.device, int], torch.Tensor] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _packing_alignment: int = field(default=0, init=False, repr=False, compare=False)
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

    _prepared_sequences: dict[torch.device, torch.Tensor] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _use_prepared_packing_inputs: bool = field(default=False, init=False, repr=False, compare=False)
    _restored_from_packing_tensors: bool = field(default=False, init=False, repr=False, compare=False)

    def prepare_packing_geometry(self, device: torch.device) -> None:
        """Prepare CP1 graph metadata independently of compression or indexing.

        The runtime calls this before capture. Eager retains its real boundaries;
        graph inputs carry token-shaped metadata, not a variable document list.
        """
        if self.local_query_start != 0 or self.local_query_length != self.global_sequence_length:
            return
        self.segment_start_mask(device)
        object.__setattr__(self, "_use_prepared_packing_inputs", True)

    def router_metadata(self, device: torch.device) -> tuple[torch.Tensor, int]:
        """Return document IDs and storage capacity; empty rows do not affect the mean."""
        count = self.local_query_length if self._restored_from_packing_tensors else self.cu_seq_lens.numel() - 1
        return self._prepared_sequences[torch.device(device)], count

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
            prepared = type(self)(
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
        if device not in self._prepared_sequences:
            self._prepared_sequences[device] = torch.bucketize(
                self.local_positions(device).squeeze(0), self.cu_seq_lens.to(device=device)[1:], right=True,
            ).unsqueeze(0)
        for ratio in set(compress_ratios):
            self.validate_compression_ratio(ratio)
            if ratio > 0 and (device, ratio) not in self._prepared_minima:
                if self._restored_from_packing_tensors:
                    raise ValueError(
                        "graph packing inputs carry no compressed-key metadata for ratio "
                        f"{ratio}; configure every active model compress ratio on the "
                        "runtime input adapter (compress_ratios) so the graph boundary "
                        "can prepare it"
                    )
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

    def document_boundaries(self) -> tuple[int, ...]:
        """Reuse the owned batch's host boundaries for routing across layers."""
        if "boundaries" not in self._snapshot:
            self._snapshot["boundaries"] = tuple(self.cu_seq_lens.detach().cpu().tolist())
        return self._snapshot["boundaries"]

    def indexer_segments(self, ratio: int) -> tuple[tuple[int, int, int, int, int], ...]:
        """Describe packed Q intervals and completed K prefixes without per-layer synchronization.

        Args:
            ratio: Compression ratio used to express each document's key prefix.
        """
        if self._restored_from_packing_tensors:
            raise NotImplementedError(
                "graph packing inputs do not carry Python document segments; the "
                "reference selection consumes per-token tensor bounds instead"
            )
        if ratio not in self._indexer_segments:
            self.validate_compression_ratio(ratio)
            boundaries = self.document_boundaries()
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
        device = torch.device(device)
        # Pure tensor arithmetic from cu_seq_lens, identical for eager and the
        # graph boundary: count each document's complete blocks, then take the
        # exclusive cumsum as the document's global compressed-key base.
        boundaries = self.cu_seq_lens.to(device=device, dtype=torch.long)
        documents = self._prepared_sequences.get(device)
        if documents is None:
            documents = torch.bucketize(
                self.local_positions(device).squeeze(0), boundaries[1:], right=True,
            ).unsqueeze(0)
        counts = (boundaries[1:] - boundaries[:-1]) // compress_ratio
        offsets = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)[:-1]))
        return offsets.index_select(0, documents.squeeze(0)).unsqueeze(0)

    def causal_key_upper_bounds(
            self, device: torch.device, compress_ratio: int,
    ) -> torch.Tensor:
        """Return the exclusive upper bound of compressed keys visible to each query.

        Both execution modes share one derivation: a query at document-local
        position ``p`` sees compressed keys ``[document base, base + (p+1)//ratio)``.
        The result is a ``[1, local queries]`` tensor, so graph capture replays it
        across batches whose document count and lengths differ.

        Args:
            device: Device on which tensors and prepared geometry are consumed.
            compress_ratio: Compression ratio of the consuming layer.
        """
        if compress_ratio <= 0:
            raise ValueError("causal_key_upper_bounds requires a positive compression ratio")
        key = (torch.device(device), compress_ratio)
        cached = self._prepared_bounds.get(key)
        if cached is None:
            device = torch.device(device)
            minima = self.minimum_key_indices(device, compress_ratio)
            cached = minima + (
                (self.local_positions(device) - self.local_segment_starts(device) + 1) // compress_ratio
            )
            self._prepared_bounds[key] = cached
        return cached

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

    def is_compression_aligned(self, compress_ratio: int) -> bool:
        """Report whether every document boundary lies on a complete-block edge.

        Eager reads the real boundaries; graph packing inputs read the alignment
        the graph boundary validated before capture. Both answer the same
        question for the same batch.

        Args:
            compress_ratio: Compression ratio of the consuming layer.
        """
        if compress_ratio <= 1:
            return True
        if self._restored_from_packing_tensors:
            return 0 < self._packing_alignment and self._packing_alignment % compress_ratio == 0
        return not any(boundary % compress_ratio for boundary in self.document_boundaries())

    def compressed_key_length(self, compress_ratio: int) -> int:
        """Return the total number of complete compressed keys in the batch.

        For alignment-validated graph packing inputs the contiguous count equals
        the per-document sum; eager keeps exact arithmetic for natural lengths.

        Args:
            compress_ratio: Compression ratio of the consuming layer.
        """
        ratio = max(1, compress_ratio)
        if self._restored_from_packing_tensors:
            return self.local_query_length // ratio
        boundaries = self.document_boundaries()
        return sum((end - start) // ratio for start, end in zip(boundaries, boundaries[1:]))

    def block_plan(self, ratio: int) -> Any:
        """Return complete local document blocks; CP ownership is planned by its member."""
        if self._restored_from_packing_tensors or ratio <= 1:
            return None
        boundaries = self.document_boundaries()
        if self.local_query_length % ratio == 0 and all(value % ratio == 0 for value in boundaries):
            return None
        if self.local_query_start or self.local_query_length != self.global_sequence_length:
            raise ValueError("local document block planning requires the complete local sequence")
        blocks = tuple((start, (end - start) // ratio) for start, end in zip(boundaries, boundaries[1:])
                       if (end - start) // ratio)
        return DocumentBlockPlan(ratio, blocks)

    def cached_compressed_positions(self, ratio: int, embeddings: tuple) -> tuple | None:
        """Read a batch-local position cache without launching communication."""
        cos, sin = embeddings
        versions = tuple(None if value.is_inference() else value._version for value in embeddings)  # pylint: disable=protected-access
        cached = self._compression_rope.get(ratio)
        if (cached is not None and cached[0] is cos and cached[1] is sin and cached[2] == versions
                and None not in versions and not cos.requires_grad and not sin.requires_grad):
            return cached[3]
        return None

    def cache_compressed_positions(self, ratio: int, embeddings: tuple, positions: tuple) -> None:
        """Retain only non-differentiable position metadata for the same source tensors."""
        if any(value.requires_grad for value in embeddings):
            return
        versions = tuple(None if value.is_inference() else value._version for value in embeddings)  # pylint: disable=protected-access
        self._compression_rope[ratio] = (*embeddings, versions, positions)

    def compressed_positions(self, plan: Any, embeddings: tuple) -> tuple:
        """Select local block-start positions without any tensor exchange."""
        cached = self.cached_compressed_positions(plan.ratio, embeddings)
        if cached is not None:
            return cached
        rows = plan.indices(embeddings[0].device)[1]
        positions = tuple(value.index_select(1, rows) for value in embeddings)
        self.cache_compressed_positions(plan.ratio, embeddings, positions)
        return positions

    def raw_document_starts(self) -> tuple:
        """Keep raw token coordinates separate from compact compressed-key IDs."""
        if self._restored_from_packing_tensors:
            # Only the fused attention kernel consumes these Python
            # coordinates; fused execution is gated out of graph mode.
            return ()
        first, last = self.local_query_start, self.local_query_start + self.local_query_length
        return tuple((max(first, start)-first, start)
                     for start, end in zip(self.document_boundaries(), self.document_boundaries()[1:])
                     if max(first, start) < min(last, end))


    def pytree_flatten(self) -> tuple[list[Any], tuple]:
        """Expose packing tensors with a schema independent of document count and S."""
        if self._use_prepared_packing_inputs:
            return [self._prepared_starts, self._prepared_positions,
                    self._prepared_masks, self._prepared_sequences,
                    self._prepared_minima], ("packing", self._packing_alignment)
        return [self.cu_seq_lens], (self.local_query_start, self.local_query_length, self.global_sequence_length)

    @classmethod
    def pytree_unflatten(cls, children: Any, context: tuple) -> SharedCompressedPackedSequence:
        """Restore prepared metadata without reading boundary values during capture."""
        if context[0] != "packing":
            return cls(children[0], *context)
        starts, positions, masks, sequences, minima = children
        reference = next(iter(starts.values()))
        length = reference.shape[-1]
        packed = cls(reference.new_empty(0), 0, length, length)
        packed._prepared_starts.update(starts)
        packed._prepared_positions.update(positions)
        packed._prepared_masks.update(masks)
        packed._prepared_sequences.update(sequences)
        packed._prepared_minima.update(minima)
        object.__setattr__(packed, "_packing_alignment", context[1])
        object.__setattr__(packed, "_use_prepared_packing_inputs", True)
        object.__setattr__(packed, "_restored_from_packing_tensors", True)
        packed._mark_owned()
        return packed

class GraphPackedSequence(SharedCompressedPackedSequence):
    """Tensor-only packing inputs created exclusively at the graph boundary.

    The subclass is a registration marker: only graph-boundary inputs are
    flattened as pytree nodes, so eager FSDP keeps treating packed inputs as
    opaque leaves that retain real boundaries and caches. Every semantic lives
    in the shared base, including the compressed-key metadata that lets the
    graph carry indexer and compressor execution with the same code path.
    """


register_pytree_node(GraphPackedSequence, GraphPackedSequence.pytree_flatten,
                    GraphPackedSequence.pytree_unflatten)
