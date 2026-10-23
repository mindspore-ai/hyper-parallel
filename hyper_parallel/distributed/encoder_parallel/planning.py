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
"""Pure topology, token ownership, and image work planning."""

from __future__ import annotations

from dataclasses import dataclass
from math import lcm
from typing import Any, Mapping, Sequence

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed.encoder_parallel.contracts import EncoderSegment, EncoderWorkItem


def required_omni_global_alignment(mesh_context: Any) -> int:
    """Return the shared compact-attention and decoder-SP alignment.

    Args:
        mesh_context: Decoder CP, TP, and sequence-parallel topology.
    """
    cp_size = int(mesh_context.cp_size)
    tp_parts = int(mesh_context.tp_size) if mesh_context.sequence_parallel else 1
    return lcm(2 * cp_size, cp_size * tp_parts)


@dataclass(frozen=True)
class DecoderSequenceLayout:
    """The exact decoder hidden-token interval on one CP/TP rank."""

    sequence_parallel: bool
    cp_rank: int
    cp_size: int
    tp_rank: int
    tp_size: int
    cp_local_length: int
    global_sequence_length: int
    cp_local_start: int
    cp_local_stop: int
    global_start: int
    global_stop: int

    @classmethod
    def build(cls, *, mesh_context: Any, cp_local_length: int,
              cp_rank: int | None = None, tp_rank: int | None = None) -> "DecoderSequenceLayout":
        """Derive the token interval after embedding's TP reduce-scatter."""
        cp_size = int(mesh_context.cp_size)
        tp_size = int(mesh_context.tp_size)
        cp_rank = int(mesh_context.cp_rank if cp_rank is None else cp_rank)
        tp_rank = int(mesh_context.tp_rank if tp_rank is None else tp_rank)
        sp = bool(mesh_context.sequence_parallel)
        if not 0 <= cp_rank < cp_size or not 0 <= tp_rank < tp_size:
            raise ValueError(
                "[HP-ENCODER-004] decoder CP/TP coordinate is outside topology; "
                f"cp_rank={cp_rank}; cp_size={cp_size}; tp_rank={tp_rank}; tp_size={tp_size}"
            )
        if cp_local_length < 1 or (sp and cp_local_length % tp_size):
            raise ValueError(
                "[HP-ENCODER-004] invalid decoder sequence layout; "
                f"sequence_parallel={sp}; cp_local_length={cp_local_length}; tp_size={tp_size}; "
                "constraint=cp_local_length % tp_size == 0 when SP is enabled"
            )
        local_length = cp_local_length // (tp_size if sp else 1)
        local_start = (tp_rank if sp else 0) * local_length
        global_start = cp_rank * cp_local_length + local_start
        return cls(sp, cp_rank, cp_size, tp_rank, tp_size, cp_local_length,
                   cp_size * cp_local_length, local_start, local_start + local_length,
                   global_start, global_start + local_length)

    def localize(self, value: torch.Tensor, *, sequence_dim: int = 1) -> torch.Tensor:
        """Slice a CP-local auxiliary tensor to this hidden-token interval.

        Args:
            value: Auxiliary tensor replicated across decoder TP ranks.
            sequence_dim: Dimension containing CP-local token rows.
        """
        if value.shape[sequence_dim] != self.cp_local_length:
            raise ValueError(
                "[HP-ENCODER-004] CP-local token tensor length mismatch; "
                f"actual={value.shape[sequence_dim]}; expected={self.cp_local_length}; "
                f"cp_rank={self.cp_rank}; tp_rank={self.tp_rank}"
            )
        return value.narrow(sequence_dim, self.cp_local_start,
                            self.cp_local_stop - self.cp_local_start).contiguous()

    def validate_hidden(self, hidden: torch.Tensor, *, sequence_dim: int = 1) -> None:
        """Fail before merging against an unexpected decoder hidden layout.

        Args:
            hidden: Decoder hidden tensor after embedding redistribution.
            sequence_dim: Dimension containing decoder token rows.
        """
        expected = self.global_stop - self.global_start
        if hidden.shape[sequence_dim] != expected:
            raise ValueError(
                "[HP-ENCODER-004] decoder hidden does not match planned CP/SP shard; "
                f"actual={hidden.shape[sequence_dim]}; expected={expected}; "
                f"global_interval=({self.global_start},{self.global_stop}); "
                f"cp_rank={self.cp_rank}; tp_rank={self.tp_rank}"
            )


def assign_lpt(items: Sequence[EncoderWorkItem], encoder_dp_size: int,
               encoder_cp_size: int) -> tuple[tuple[EncoderWorkItem, ...], ...]:
    """Assign image work by stable largest-processing-time-first scheduling.

    Args:
        items: Images with valid patch counts and stable IDs.
        encoder_dp_size: Number of visual work buckets.
        encoder_cp_size: Sequence-sharding degree of each bucket.
    """
    if encoder_dp_size < 1 or encoder_cp_size < 1:
        raise ValueError("[HP-ENCODER-002] encoder DP and CP sizes must be positive")
    buckets: list[list[EncoderWorkItem]] = [[] for _ in range(encoder_dp_size)]
    loads = [0] * encoder_dp_size
    for item in sorted(items, key=lambda value: (-value.sequence_length ** 2, value.batch_index, value.item_id)):
        bucket = min(range(encoder_dp_size), key=lambda rank: (loads[rank], rank))
        buckets[bucket].append(item)
        loads[bucket] += item.sequence_length ** 2 // encoder_cp_size
    return tuple(tuple(bucket) for bucket in buckets)


def build_segments(items: Sequence[EncoderWorkItem], encoder_cp_rank: int,
                   encoder_cp_size: int) -> tuple[EncoderSegment, ...]:
    """Build equal-size padded sequence shards for one Encoder-CP rank.

    Args:
        items: Images assigned to this Encoder-DP bucket.
        encoder_cp_rank: Sequence-shard coordinate within the bucket.
        encoder_cp_size: Number of shards per image.
    """
    if not 0 <= encoder_cp_rank < encoder_cp_size:
        raise ValueError("[HP-ENCODER-002] invalid encoder CP rank")
    segments = []
    offset = 0
    for item in items:
        if item.sequence_length < 1:
            raise ValueError("[HP-ENCODER-013] image patch count must be positive")
        local_length = (item.sequence_length + encoder_cp_size - 1) // encoder_cp_size
        start = encoder_cp_rank * local_length
        valid = max(0, min(local_length, item.sequence_length - start))
        segments.append(EncoderSegment(item, start, valid, local_length, offset))
        offset += local_length
    if not segments:
        segments.append(EncoderSegment(None, 0, 1, 1, 0))
    return tuple(segments)


def build_destinations(items: Sequence[EncoderWorkItem], layouts: Mapping[int, DecoderSequenceLayout]
                       ) -> dict[int, tuple[int, ...]]:
    """Find every CP/TP rank whose decoder hidden intersects each image span.

    Args:
        items: Images carrying their global decoder token spans.
        layouts: Decoder token interval for every rank in the parent group.
    """
    lengths = {layout.global_sequence_length for layout in layouts.values()}
    if len(lengths) != 1:
        raise ValueError("[HP-ENCODER-004] inconsistent decoder global lengths")
    global_length = next(iter(lengths))
    result = {}
    for item in items:
        start = item.decoder_token_start
        stop = start + item.decoder_token_length
        if item.decoder_token_length <= 0 or start < 0 or stop > global_length:
            raise ValueError(
                "[HP-ENCODER-016] image span is outside the decoder domain; "
                f"item_id={item.item_id}; span=({start},{stop}); global_sequence_length={global_length}"
            )
        owners = tuple(rank for rank, layout in layouts.items()
                       if max(start, layout.global_start) < min(stop, layout.global_stop))
        if not owners:
            raise ValueError(f"[HP-ENCODER-016] image span has no decoder token owner; item_id={item.item_id}")
        result[item.item_id] = owners
    return result
