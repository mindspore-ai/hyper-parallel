# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Topology-aware mapper for distributed checkpoint loading across different TP/PP sizes."""
from typing import Optional

from hyper_parallel.core.distributed_checkpoint.metadata import (
    ChunkStorageMetadata,
    MetadataIndex,
    TensorStorageMetadata,
)
from hyper_parallel.core.distributed_checkpoint.planner import (
    LoadItemType,
    ReadItem,
)
from hyper_parallel.core.distributed_checkpoint.reshard import infer_intersection
from hyper_parallel.core.distributed_checkpoint.util import chunk_to_area


class TopologyMapper:
    """Map target FQNs to checkpoint FQNs and compute ReadItems with chunk-overlap planning.

    This class unifies two responsibilities that were previously embedded inside
    ``StandardLoadPlanner.build_local_plan``:

    1. **FQN mapping** -- translate a load-side target FQN to the FQN under which
       the tensor was saved in the checkpoint.  This enables loading a checkpoint
       saved with a different PP topology where the same logical parameter may have
       migrated to a different stage and thus changed its fully-qualified name.
    2. **Chunk-overlap planning** -- compute the intersection between saved chunks
       (from checkpoint metadata) and the target rank's local chunks, producing
       ``ReadItem`` objects whose ``dest_index.fqn`` and ``storage_index.fqn`` are
       correctly separated.

    Unmapped keys are treated as identity (target FQN == checkpoint FQN), so
    existing static-load behaviour is preserved when no ``TopologyMapper`` is
    supplied.

    Args:
        target_to_checkpoint_fqn: Mapping from load-side target FQN to the
            checkpoint FQN.  When ``None`` or empty, all keys are treated as
            identity.  The mapping is deep-copied on construction so that
            later mutations by the caller do not affect the mapper.

    Raises:
        ValueError: If a key or value in *target_to_checkpoint_fqn* is not a
            non-empty string.
    """

    def __init__(
        self,
        target_to_checkpoint_fqn: Optional[dict[str, str]] = None,
    ) -> None:
        raw = target_to_checkpoint_fqn or {}
        self._target_to_checkpoint: dict[str, str] = {}
        for target_fqn, ckpt_fqn in raw.items():
            if not isinstance(target_fqn, str) or not target_fqn:
                raise ValueError(
                    f"target_to_checkpoint_fqn keys must be non-empty strings, "
                    f"got {target_fqn!r}"
                )
            if not isinstance(ckpt_fqn, str) or not ckpt_fqn:
                raise ValueError(
                    f"target_to_checkpoint_fqn values must be non-empty strings, "
                    f"got {ckpt_fqn!r} for target FQN {target_fqn!r}"
                )
            self._target_to_checkpoint[target_fqn] = ckpt_fqn

    def map_fqn(self, target_fqn: str) -> str:
        """Return the checkpoint FQN for *target_fqn*.

        If *target_fqn* is present in the mapping table the corresponding
        checkpoint FQN is returned; otherwise *target_fqn* itself is returned
        (identity mapping).

        Args:
            target_fqn: Fully-qualified name on the load side.

        Returns:
            The checkpoint FQN that should be used to look up metadata.

        Raises:
            ValueError: If *target_fqn* is not a non-empty string.
        """
        if not isinstance(target_fqn, str) or not target_fqn:
            raise ValueError(
                f"target_fqn must be a non-empty string, got {target_fqn!r}"
            )
        return self._target_to_checkpoint.get(target_fqn, target_fqn)

    def compute_required_shards(
        self,
        target_fqn: str,
        checkpoint_md: TensorStorageMetadata,
        local_chunks: list[ChunkStorageMetadata],
    ) -> list[ReadItem]:
        """Compute ReadItems for a single target FQN against checkpoint metadata.

        The returned ``ReadItem`` objects use *target_fqn* as
        ``dest_index.fqn`` and the mapped checkpoint FQN as
        ``storage_index.fqn``, so that the storage reader knows which
        checkpoint shard files to read from while the load planner writes into
        the correct target tensor.

        For each target local chunk the method verifies that the saved chunks
        completely cover it.  If any portion of a local chunk has no
        intersection with the saved chunks a ``ValueError`` is raised, ensuring
        that partially-initialised (zero-filled) tensors never pass silently.

        Args:
            target_fqn: Fully-qualified name on the load side.
            checkpoint_md: Tensor storage metadata from the checkpoint.
            local_chunks: List of local chunks needed by the current rank.

        Returns:
            List of ``ReadItem`` objects whose offsets and lengths describe the
            exact byte ranges to copy from checkpoint chunks into the target
            tensor.

        Raises:
            ValueError: If *target_fqn* is not a non-empty string.
            ValueError: If any target local chunk is not fully covered by the
                saved chunks.
        """
        if not isinstance(target_fqn, str) or not target_fqn:
            raise ValueError(
                f"target_fqn must be a non-empty string, got {target_fqn!r}"
            )

        checkpoint_fqn = self.map_fqn(target_fqn)
        saved_chunks = checkpoint_md.chunks
        if not local_chunks or not saved_chunks:
            return []

        read_items: list[ReadItem] = []
        for local_idx, local_chunk in enumerate(local_chunks):
            items, covered = _compute_local_chunk_reads(
                target_fqn, checkpoint_fqn, local_idx, local_chunk, saved_chunks,
            )
            _validate_full_coverage(local_idx, target_fqn, local_chunk, covered)
            read_items.extend(items)
        return read_items


def _compute_local_chunk_reads(
    target_fqn: str,
    checkpoint_fqn: str,
    local_idx: int,
    local_chunk: ChunkStorageMetadata,
    saved_chunks: list[ChunkStorageMetadata],
) -> tuple[list[ReadItem], list[list[tuple[int, int]]]]:
    """Build ReadItems and per-dimension coverage for one local chunk.

    Args:
        target_fqn: FQN on the load side.
        checkpoint_fqn: FQN in the checkpoint.
        local_idx: Index of the local chunk.
        local_chunk: The target local chunk.
        saved_chunks: All saved chunks from checkpoint metadata.

    Returns:
        A pair ``(read_items, covered_per_dim)`` where *covered_per_dim[dim]*
        is the list of overlapping intervals contributed by each saved chunk.
    """
    local_area = chunk_to_area(local_chunk)
    ndim = len(local_area)
    covered_per_dim: list[list[tuple[int, int]]] = [[] for _ in range(ndim)]
    has_any_overlap = False
    read_items: list[ReadItem] = []

    for storage_idx, storage_chunk in enumerate(saved_chunks):
        saved_area = chunk_to_area(storage_chunk)
        overlap = infer_intersection(local_area, saved_area)
        if overlap is None:
            continue
        has_any_overlap = True
        for dim in range(ndim):
            covered_per_dim[dim].append(overlap[dim])

        dest_offsets = tuple(
            overlap[i][0] - local_chunk.offsets[i] for i in range(len(overlap))
        )
        storage_offsets = tuple(
            overlap[i][0] - storage_chunk.offsets[i] for i in range(len(overlap))
        )
        lengths = tuple(
            overlap[i][1] - overlap[i][0] for i in range(len(overlap))
        )
        read_items.append(
            ReadItem(
                type=LoadItemType.TENSOR,
                dest_index=MetadataIndex(
                    fqn=target_fqn, offset=local_chunk.offsets, index=local_idx,
                ),
                dest_offsets=dest_offsets,
                storage_index=MetadataIndex(
                    fqn=checkpoint_fqn, offset=storage_chunk.offsets, index=storage_idx,
                ),
                storage_offsets=storage_offsets,
                lengths=lengths,
            )
        )

    if not has_any_overlap:
        raise ValueError(
            f"Target local chunk {local_idx} of {target_fqn!r} "
            f"(offsets={local_chunk.offsets}, sizes={local_chunk.sizes}) "
            f"has no intersection with any saved chunk"
        )
    return read_items, covered_per_dim


def _validate_full_coverage(
    local_idx: int,
    target_fqn: str,
    local_chunk: ChunkStorageMetadata,
    covered_per_dim: list[list[tuple[int, int]]],
) -> None:
    """Raise ``ValueError`` if *covered_per_dim* does not fully cover *local_chunk*.

    Args:
        local_idx: Index of the local chunk.
        target_fqn: FQN on the load side.
        local_chunk: The target local chunk.
        covered_per_dim: Per-dimension overlapping intervals from saved chunks.

    Raises:
        ValueError: If any dimension is not fully covered.
    """
    local_area = chunk_to_area(local_chunk)
    uncovered_dims: list[int] = []
    for dim, intervals in enumerate(covered_per_dim):
        merged = _merge_intervals(intervals)
        local_start, local_end = local_area[dim]
        if not merged or merged[0][0] > local_start or merged[-1][1] < local_end or len(merged) != 1:
            uncovered_dims.append(dim)
    if uncovered_dims:
        raise ValueError(
            f"Target local chunk {local_idx} of {target_fqn!r} "
            f"(offsets={local_chunk.offsets}, sizes={local_chunk.sizes}) "
            f"is not fully covered by saved chunks; uncovered dimensions: "
            f"{uncovered_dims}"
        )


def _merge_intervals(
    intervals: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Merge overlapping or adjacent intervals into sorted, non-overlapping ranges.

    Args:
        intervals: List of ``(start, end)`` intervals (start < end).

    Returns:
        Sorted list of merged ``(start, end)`` intervals.
    """
    if not intervals:
        return []
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged: list[tuple[int, int]] = [sorted_intervals[0]]
    for start, end in sorted_intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged
