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
"""Document intervals, local packed execution and chunk plans for linear attention.

This dependency leaf is shared with functional adapters and model runtimes.
Keep it independent of module-family registration, backends and collectives.
"""
from __future__ import annotations
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Mapping, Sequence

import torch


# Bound retained interval views and their materialized chunk indices. Evicted
# views can be rebuilt; this cache budget does not limit CP degree or documents.
_MAX_CACHED_INTERVALS = 32


@dataclass(frozen=True)
class PackedChunkPlan:
    """Read-only kernel indices for one local packed interval and chunk size."""

    cu_seqlens: torch.Tensor
    cu_seqlens_cpu: torch.Tensor
    chunk_indices: torch.Tensor
    chunk_offsets: torch.Tensor

    def kwargs(self) -> dict[str, torch.Tensor]:
        """Return the common FLA varlen arguments without copying device tensors."""
        return {"cu_seqlens": self.cu_seqlens, "chunk_indices": self.chunk_indices}


@dataclass(frozen=True, eq=False)
class PackedDocumentInfo:
    """Validated immutable document snapshot shared by a batch's interval views.

    Identity hashing lets bounded CPU-plan caches reuse this snapshot without
    hashing every document offset at every layer. Cache keys retain the object,
    so distinct batches cannot collide through reuse of a Python object ID.
    """

    offsets: tuple[int, ...]

    def __post_init__(self) -> None:
        offsets = tuple(self.offsets)
        if (
            len(offsets) < 2
            or offsets[0] != 0
            or any(not isinstance(value, int) or isinstance(value, bool) for value in offsets)
            or any(left >= right for left, right in zip(offsets, offsets[1:]))
        ):
            raise ValueError("Packed sample offsets must be strictly increasing integers starting at zero.")
        object.__setattr__(self, "offsets", offsets)


@dataclass(frozen=True)
class PackedSequenceInfo:
    """Own immutable sample offsets and a contiguous physical token interval.

    Offsets are global within one CP group, including any explicitly declared
    padding sample. Local CP cuts are fragments, never new sample resets.
    Device plans contain read-only tensors; each invocation saves the tensors
    it consumes so that ordinary autograd version checks remain effective.
    """

    global_offsets: tuple[int, ...] | PackedDocumentInfo
    start: int
    end: int
    local_offsets: tuple[int, ...] = field(init=False)
    documents: PackedDocumentInfo = field(init=False, repr=False, compare=False, hash=False)
    _head_index: int = field(init=False, repr=False, compare=False, hash=False)
    _tail_index: int = field(init=False, repr=False, compare=False, hash=False)
    _plans: dict[tuple[torch.device, int], PackedChunkPlan] = field(
        default_factory=dict, init=False, repr=False, compare=False, hash=False
    )
    _intervals: dict[tuple[int, int], PackedSequenceInfo] = field(
        default_factory=dict, init=False, repr=False, compare=False, hash=False
    )

    def __post_init__(self) -> None:
        documents = (self.global_offsets if isinstance(self.global_offsets, PackedDocumentInfo)
                     else PackedDocumentInfo(self.global_offsets))
        offsets = documents.offsets
        if (
            not isinstance(self.start, int) or isinstance(self.start, bool)
            or not isinstance(self.end, int) or isinstance(self.end, bool)
            or not 0 <= self.start < self.end <= offsets[-1]
        ):
            raise ValueError("Packed token interval must be nonempty and covered by global sample offsets.")
        head = bisect_right(offsets, self.start) - 1
        tail = bisect_left(offsets, self.end) - 1
        local = (0, *(value - self.start for value in offsets[head + 1:tail + 1]), self.end - self.start)
        object.__setattr__(self, "global_offsets", offsets)
        object.__setattr__(self, "local_offsets", local)
        object.__setattr__(self, "documents", documents)
        object.__setattr__(self, "_head_index", head)
        object.__setattr__(self, "_tail_index", tail)

    @property
    def length(self) -> int:
        """Return this interval's physical token count."""
        return self.end - self.start

    @property
    def num_sequences(self) -> int:
        """Return the number of nonempty local sample fragments."""
        return len(self.local_offsets) - 1

    @property
    def continues_previous(self) -> bool:
        """Whether the first fragment consumes the preceding interval's state."""
        return self.start != self.global_offsets[self._head_index]

    @property
    def continues_next(self) -> bool:
        """Whether the last fragment feeds the following interval's state."""
        return self.end != self.global_offsets[self._tail_index + 1]

    @property
    def transfer_active(self) -> bool:
        """Whether the preceding interval can affect this interval's exit."""
        return self.continues_previous and self.num_sequences == 1

    @property
    def head(self) -> slice:
        """Return the first fragment's local token slice."""
        return slice(0, self.local_offsets[1])

    @property
    def tail(self) -> slice:
        """Return the last fragment's local token slice."""
        return slice(self.local_offsets[-2], self.length)

    @property
    def preceding_sample_tokens(self) -> int:
        """Return how many earlier tokens belong to the first local sample."""
        return self.start - self.global_offsets[self._head_index]

    def interval(self, start: int, end: int) -> PackedSequenceInfo:
        """Describe another contiguous interval without retaining artificial cuts.

        Args:
            start: Inclusive start of the token interval.
            end: Exclusive end of the token interval.
        """
        key = (start, end)
        if key == (self.start, self.end):
            return self
        if key not in self._intervals:
            if len(self._intervals) == _MAX_CACHED_INTERVALS:
                self._intervals.pop(next(iter(self._intervals)))
            self._intervals[key] = PackedSequenceInfo(self.documents, start, end)
        return self._intervals[key]

    def ulysses(self, degree: int, local_rank: int) -> PackedSequenceInfo:
        """Describe the token interval after consecutive-rank AllToAll.

        Args:
            degree: Number of ranks participating in the head/sequence exchange.
            local_rank: Rank index in the Ulysses group.
        """
        if degree < 1 or not 0 <= local_rank < degree:
            raise ValueError("Invalid Ulysses degree or local rank.")
        start = self.start - local_rank * self.length
        return self.interval(start, start + degree * self.length)

    def materialize(self, device: torch.device, chunk_size: int) -> PackedChunkPlan:
        """Prepare immutable indices once per batch, layout, device and chunk size.

        Args:
            device: Device on which to materialize sequence metadata.
            chunk_size: Number of tokens in a recurrence chunk.
        """
        device = torch.device(device)
        cache_key = (device, chunk_size)
        if cache_key in self._plans:
            return self._plans[cache_key]
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive.")
        counts = [
            (end - start + chunk_size - 1) // chunk_size
            for start, end in zip(self.local_offsets, self.local_offsets[1:])
        ]
        indices = [(sequence, chunk) for sequence, count in enumerate(counts) for chunk in range(count)]
        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1] + count)
        cpu = torch.tensor(self.local_offsets, dtype=torch.int64)
        plan = PackedChunkPlan(
            cpu.to(device),
            cpu,
            torch.tensor(indices, device=device, dtype=torch.int64),
            torch.tensor(offsets, device=device, dtype=torch.int64),
        )
        self._plans[cache_key] = plan
        return plan

    def first_state(self, state: torch.Tensor | None) -> torch.Tensor | None:
        """Expand only a continued first fragment's state for a varlen kernel.

        Args:
            state: State tensor at the invocation boundary, or None for zero input.
        """
        if state is None or not self.continues_previous:
            return None
        if self.num_sequences == 1:
            return state
        return torch.cat((state, state.new_zeros((self.num_sequences - 1, *state.shape[1:]))))

    def last_state(self, state: torch.Tensor | None) -> torch.Tensor | None:
        """Expand a terminal state gradient into the last fragment slot.

        Args:
            state: State tensor at the invocation boundary, or None for zero input.
        """
        if state is None or not self.continues_next:
            return None
        if self.num_sequences == 1:
            return state
        return torch.cat((state.new_zeros((self.num_sequences - 1, *state.shape[1:])), state))


@dataclass(frozen=True)
class PackedSummaryExchangePlan:
    """Host-only forward edges; reverse communication uses their transpose."""

    send_counts: tuple[int, ...]
    recv_counts: tuple[int, ...]
    active: bool


@lru_cache(maxsize=32)
def _packed_summary_exchange_plan(documents: PackedDocumentInfo, length: int,
                                  rank: int, size: int) -> PackedSummaryExchangePlan:
    """Build summary counts for this rank's tail consumers and head producers."""
    offsets = documents.offsets
    if offsets[-1] != length * size:
        raise ValueError("Sparse summaries require equal contiguous token shards covering the document offsets.")
    start, end = rank * length, (rank + 1) * length
    first = offsets[bisect_right(offsets, start) - 1] // length
    last = (offsets[bisect_left(offsets, end)] - 1) // length
    sends = tuple(int(rank < peer <= last) for peer in range(size))
    receives = tuple(int(first <= peer < rank) for peer in range(size))
    # A single global decision lets all ranks skip an empty collective together.
    # Inspect rank cuts rather than walking a potentially much longer document list.
    active = any(offsets[bisect_left(offsets, cut)] != cut for cut in range(length, length * size, length))
    return PackedSummaryExchangePlan(sends, receives, active)


def plan_packed_summary_exchange(info: PackedSequenceInfo, rank: int, size: int) -> PackedSummaryExchangePlan:
    """Cache exact head/tail document dependencies without device metadata reads.

    Args:
        info: Host document metadata after any Ulysses token redistribution.
        rank: Chronological rank within the state group.
        size: Number of equal contiguous token shards in that group.
    """
    if not 0 <= rank < size or info.start != rank * info.length:
        raise ValueError("Sparse summary metadata must match the chronological state-group rank.")
    return _packed_summary_exchange_plan(info.documents, info.length, rank, size)


def _offset_tuple(value: torch.Tensor | Sequence[int]) -> tuple[int, ...]:
    """Snapshot user metadata so a later batch cannot mutate saved semantics."""
    if isinstance(value, torch.Tensor):
        if value.ndim != 1 or value.dtype not in (torch.int32, torch.int64):
            raise ValueError("Cumulative sample offsets must be a one-dimensional integer tensor.")
        return tuple(value.detach().cpu().tolist())
    return tuple(value)


def _global_offsets(info: PackedSequenceInfo | None, aliases: list) -> tuple[int, ...]:
    """Check alias agreement before constructing or reusing local metadata."""
    if info is not None and not isinstance(info, PackedSequenceInfo):
        raise TypeError("packed_sequence_info must be PackedSequenceInfo.")
    offsets = info.global_offsets if info is not None else _offset_tuple(aliases[0])
    # Runtime q/k aliases share this immutable tuple; comparing the same tuple
    # still walks its elements in Python, so skip only exact object identity.
    if any(_offset_tuple(value) != offsets for value in aliases if value is not offsets):
        raise ValueError("Packed sequence aliases disagree; provide one consistent set of global sample offsets.")
    return offsets


def resolve_packed_sequence(
    arguments: Mapping[str, Any],
    local_length: int,
    *,
    cp_rank: int = 0,
    cp_size: int = 1,
) -> PackedSequenceInfo | None:
    """Normalize model-level aliases, all of which describe GLOBAL sample offsets.

    Kernel-level cu_seqlens is derived from the returned local interval. Prefer
    a prebuilt packed_sequence_info from the runtime adapter to avoid host reads
    or per-layer validation of device metadata.

    Args:
        arguments: Per-forward keyword arguments carrying document boundaries.
        local_length: Number of tokens owned by this CP rank.
        cp_rank: Rank index within the CP mesh.
        cp_size: Number of ranks in the CP mesh.
    """
    info = arguments.get("packed_sequence_info")
    aliases = [
        arguments[name]
        for name in ("cu_seq_lens", "cu_seqlens", "cu_seq_lens_q", "cu_seq_lens_k")
        if arguments.get(name) is not None
    ]
    if arguments.get("packed_seq_params") is not None or arguments.get("seq_idx") is not None:
        raise ValueError(
            "Use global cu_seq_lens or PackedSequenceInfo; implicit packed/seq_idx conversion is unsupported."
        )
    if info is None and not aliases:
        return None
    if not 0 <= cp_rank < cp_size:
        raise ValueError("Invalid CP rank or size for packed sequence metadata.")
    offsets = _global_offsets(info, aliases)
    if info is None:
        info = PackedSequenceInfo(offsets, cp_rank * local_length, (cp_rank + 1) * local_length)
    if (
        info.length != local_length
        or info.start != cp_rank * local_length
        or info.global_offsets[-1] != cp_size * local_length
    ):
        raise ValueError("Packed metadata must cover the chronological, equal-length CP token intervals.")
    return info


def local_sequence_offsets(
    cu_seqlens: torch.Tensor | Sequence[int], tokens: int, *, require_cpu: bool = False,
) -> tuple[int, ...]:
    """Validate local document offsets without constructing a CP interval.

    Args:
        cu_seqlens: Local cumulative document boundaries.
        tokens: Local packed token count.
        require_cpu: Reject device metadata when the native ABI requires host offsets.
    """
    if isinstance(cu_seqlens, torch.Tensor):
        if require_cpu and cu_seqlens.device.type != "cpu":
            raise ValueError("Native cu_seqlens must be a CPU integer tensor or sequence.")
        if cu_seqlens.ndim != 1 or cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise ValueError("cu_seqlens must be a one-dimensional integer tensor.")
        cu_seqlens = cu_seqlens.detach().cpu().tolist()
    offsets = tuple(cu_seqlens)
    if (len(offsets) < 2 or offsets[0] != 0
            or any(not isinstance(value, int) or isinstance(value, bool) for value in offsets)
            or any(left >= right for left, right in zip(offsets, offsets[1:]))):
        raise ValueError("Packed sample offsets must be strictly increasing integers starting at zero.")
    if offsets[-1] != tokens:
        raise ValueError("Local cumulative offsets must end at the input token count.")
    return offsets


def packed_eager(
    operator: Callable[..., tuple[torch.Tensor, torch.Tensor | None]],
    inputs: tuple[torch.Tensor, ...],
    cu_seqlens: torch.Tensor | Sequence[int],
    *,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Evaluate the local eager recurrence independently per sample fragment.

    Args:
        operator: Local recurrence callable to evaluate for each document.
        inputs: Token-aligned recurrence operands.
        cu_seqlens: Cumulative document boundaries for packed tokens.
        initial_state: Optional incoming-state slot for each local sample fragment.
        output_final_state: Whether to return one outgoing state per local sample fragment.
    """
    if inputs[0].shape[0] != 1:
        raise ValueError("Packed eager requires batch dimension one.")
    offsets = local_sequence_offsets(cu_seqlens, inputs[0].shape[1])
    if initial_state is not None and initial_state.shape[0] != len(offsets) - 1:
        raise ValueError("Packed initial_state requires one slot per local sample fragment.")
    outputs, finals = [], []
    for index, (start, end) in enumerate(zip(offsets, offsets[1:])):
        state = None if initial_state is None else initial_state[index : index + 1]
        output, final = operator(
            *(tensor[:, start:end] for tensor in inputs),
            initial_state=state,
            output_final_state=output_final_state,
            **kwargs,
        )
        outputs.append(output)
        if output_final_state:
            finals.append(final)
    return torch.cat(outputs, dim=1), torch.cat(finals) if output_final_state else None


def local_chunk_metadata(
    cu_seqlens: torch.Tensor | Sequence[int] | None, tokens: int, chunk_size: int = 64,
) -> dict[str, tuple[int, ...]]:
    """Build host offsets and flattened document/chunk pairs for local operators.

    Args:
        cu_seqlens: Host document offsets, or None for dense input.
        tokens: Number of tokens in this invocation.
        chunk_size: Number of tokens in a kernel chunk.

    Returns:
        Local cu_seqlens and flattened chunk_indices, or an empty dense descriptor.
    """
    if cu_seqlens is None:
        return {}
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    offsets = local_sequence_offsets(cu_seqlens, tokens, require_cpu=True)
    indices = tuple(value for seq, (begin, end) in enumerate(zip(offsets, offsets[1:]))
                    for chunk in range((end - begin + chunk_size - 1) // chunk_size) for value in (seq, chunk))
    return {"cu_seqlens": offsets, "chunk_indices": indices}
