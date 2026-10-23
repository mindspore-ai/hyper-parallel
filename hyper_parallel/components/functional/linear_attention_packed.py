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
"""Packed token metadata and causal convolution shared by linear attention."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


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


@dataclass(frozen=True)
class PackedSequenceInfo:
    """Own immutable sample offsets and a contiguous physical token interval.

    Offsets are global within one CP group, including any explicitly declared
    padding sample. Local CP cuts are fragments, never new sample resets.
    Device plans contain read-only tensors; each invocation saves the tensors
    it consumes so that ordinary autograd version checks remain effective.
    """

    global_offsets: tuple[int, ...]
    start: int
    end: int
    local_offsets: tuple[int, ...] = field(init=False)
    _plans: dict[tuple[torch.device, int], PackedChunkPlan] = field(
        default_factory=dict, init=False, repr=False, compare=False, hash=False
    )
    _intervals: dict[tuple[int, int], PackedSequenceInfo] = field(
        default_factory=dict, init=False, repr=False, compare=False, hash=False
    )

    def __post_init__(self) -> None:
        offsets = tuple(self.global_offsets)
        if (
            len(offsets) < 2
            or offsets[0] != 0
            or any(not isinstance(x, int) or isinstance(x, bool) for x in offsets)
            or any(left >= right for left, right in zip(offsets, offsets[1:]))
        ):
            raise ValueError("Packed sample offsets must be strictly increasing integers starting at zero.")
        if (
            not isinstance(self.start, int) or isinstance(self.start, bool)
            or not isinstance(self.end, int) or isinstance(self.end, bool)
            or not 0 <= self.start < self.end <= offsets[-1]
        ):
            raise ValueError("Packed token interval must be nonempty and covered by global sample offsets.")
        local = (0, *(x - self.start for x in offsets if self.start < x < self.end), self.end - self.start)
        object.__setattr__(self, "global_offsets", offsets)
        object.__setattr__(self, "local_offsets", local)

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
        return self.start not in self.global_offsets

    @property
    def continues_next(self) -> bool:
        """Whether the last fragment feeds the following interval's state."""
        return self.end not in self.global_offsets

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
        index = bisect_right(self.global_offsets, self.start) - 1
        return self.start - self.global_offsets[index]

    def interval(self, start: int, end: int) -> PackedSequenceInfo:
        """Describe another contiguous interval without retaining artificial cuts."""
        key = (start, end)
        if key == (self.start, self.end):
            return self
        if key not in self._intervals:
            self._intervals[key] = PackedSequenceInfo(self.global_offsets, start, end)
        return self._intervals[key]

    def ulysses(self, degree: int, local_rank: int) -> PackedSequenceInfo:
        """Describe the token interval after consecutive-rank AllToAll."""
        if degree < 1 or not 0 <= local_rank < degree:
            raise ValueError("Invalid Ulysses degree or local rank.")
        start = self.start - local_rank * self.length
        return self.interval(start, start + degree * self.length)

    def materialize(self, device: torch.device, chunk_size: int) -> PackedChunkPlan:
        """Prepare immutable indices once per batch, layout, device and chunk size."""
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
        """Expand only a continued first fragment's state for a varlen kernel."""
        if state is None or not self.continues_previous:
            return None
        if self.num_sequences == 1:
            return state
        return torch.cat((state, state.new_zeros((self.num_sequences - 1, *state.shape[1:]))))

    def last_state(self, state: torch.Tensor | None) -> torch.Tensor | None:
        """Expand a terminal state gradient into the last fragment slot."""
        if state is None or not self.continues_next:
            return None
        if self.num_sequences == 1:
            return state
        return torch.cat((state.new_zeros((self.num_sequences - 1, *state.shape[1:])), state))


def _offset_tuple(value: torch.Tensor | Sequence[int]) -> tuple[int, ...]:
    """Snapshot user metadata so a later batch cannot mutate saved semantics."""
    if isinstance(value, torch.Tensor):
        if value.ndim != 1 or value.dtype not in (torch.int32, torch.int64):
            raise ValueError("Cumulative sample offsets must be a one-dimensional integer tensor.")
        return tuple(value.detach().cpu().tolist())
    return tuple(value)


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
    if info is not None and not isinstance(info, PackedSequenceInfo):
        raise TypeError("packed_sequence_info must be PackedSequenceInfo.")
    offsets = info.global_offsets if info is not None else _offset_tuple(aliases[0])
    if any(_offset_tuple(value) != offsets for value in aliases):
        raise ValueError("Packed sequence aliases disagree; provide one consistent set of global sample offsets.")
    if info is None:
        info = PackedSequenceInfo(offsets, cp_rank * local_length, (cp_rank + 1) * local_length)
    if (
        info.length != local_length
        or info.start != cp_rank * local_length
        or info.global_offsets[-1] != cp_size * local_length
    ):
        raise ValueError("Packed metadata must cover the chronological, equal-length CP token intervals.")
    return info


def packed_causal_conv1d(
    inputs: torch.Tensor,
    convolution: nn.Conv1d,
    info: PackedSequenceInfo,
    halo: torch.Tensor | None = None,
    *,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply depthwise causal SiLU convolution with exact sample isolation.

    Vectorize over tokens rather than loop over samples. Accumulate low-precision
    products in FP32; keep FP32/FP64 reference inputs at their original precision.
    Masking taps also masks their autograd contribution to an exchanged halo.
    """
    if inputs.ndim != 3 or inputs.shape[:2] != (1, info.length):
        raise ValueError("Packed convolution expects [1,T,C] matching its token interval.")
    channels = inputs.shape[-1]
    if (
        convolution.groups != convolution.in_channels
        or convolution.out_channels != convolution.in_channels
        or convolution.stride != (1,)
    ):
        raise ValueError("Packed convolution requires a stride-one depthwise Conv1d.")
    if weight is None:
        weight, bias = convolution.weight, convolution.bias
    width, dilation = convolution.kernel_size[0], convolution.dilation[0]
    if weight.shape != (channels, 1, width):
        raise ValueError("Packed convolution weight must match the local head channels.")
    halo_width = (width - 1) * dilation
    if halo is None:
        halo = inputs.new_zeros((1, halo_width, channels))
    if halo.shape != (1, halo_width, channels):
        raise ValueError("Packed convolution halo has an incompatible shape.")
    extended = torch.cat((halo, inputs), dim=1)
    positions = torch.arange(info.start, info.end, device=inputs.device)
    boundaries = torch.tensor(info.global_offsets, device=inputs.device)
    sample_indices = torch.bucketize(positions, boundaries[1:], right=True)
    starts = boundaries[sample_indices]
    dtype = torch.float32 if inputs.dtype in (torch.bfloat16, torch.float16) else inputs.dtype
    output = torch.zeros_like(inputs, dtype=dtype)
    weights = weight[:, 0].to(dtype)
    for tap in range(width):
        valid = positions - (width - 1 - tap) * dilation >= starts
        values = extended[:, tap * dilation : tap * dilation + info.length].to(dtype)
        values = torch.where(valid[None, :, None], values, 0.0)
        output = output + values * weights[None, None, :, tap]
    if bias is not None:
        output = output + bias.to(dtype)
    return F.silu(output.to(inputs.dtype))


def packed_reference(
    operator: Callable[..., tuple[torch.Tensor, torch.Tensor | None]],
    inputs: tuple[torch.Tensor, ...],
    cu_seqlens: torch.Tensor | Sequence[int],
    *,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Evaluate the portable reference independently per sample fragment."""
    if inputs[0].shape[0] != 1:
        raise ValueError("Packed reference requires batch dimension one.")
    info = PackedSequenceInfo(_offset_tuple(cu_seqlens), 0, inputs[0].shape[1])
    if info.global_offsets[-1] != info.length:
        raise ValueError("Local cumulative offsets must end at the input token count.")
    if initial_state is not None and initial_state.shape[0] != info.num_sequences:
        raise ValueError("Packed initial_state requires one slot per local sample fragment.")
    outputs, finals = [], []
    for index, (start, end) in enumerate(zip(info.local_offsets, info.local_offsets[1:])):
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
