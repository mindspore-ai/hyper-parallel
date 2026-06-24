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
"""KV cache container for generation."""
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple

import torch

PastKeyValues = List[Tuple[torch.Tensor, torch.Tensor]]


@dataclass(frozen=True)
class SequenceShardInfo:
    """Sequence range held by one context-parallel rank."""

    rank: int
    world_size: int
    start: int
    end: int
    global_seq_len: int

    @property
    def local_seq_len(self) -> int:
        return self.end - self.start


def get_sequence_shard_info(
    global_seq_len: int,
    rank: int,
    world_size: int,
) -> SequenceShardInfo:
    """Return the contiguous sequence shard range for a CP rank."""
    if global_seq_len < 0:
        raise ValueError("global_seq_len must be >= 0")
    if world_size <= 0:
        raise ValueError("world_size must be > 0")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must be in [0, world_size)")
    base = global_seq_len // world_size
    remainder = global_seq_len % world_size
    start = rank * base + min(rank, remainder)
    end = start + base + (1 if rank < remainder else 0)
    return SequenceShardInfo(
        rank=rank,
        world_size=world_size,
        start=start,
        end=end,
        global_seq_len=global_seq_len,
    )


def shard_past_key_values(
    past_key_values: Iterable,
    rank: int,
    world_size: int,
    global_seq_len: Optional[int] = None,
) -> Tuple[PastKeyValues, SequenceShardInfo]:
    """Shard full past key values on the sequence dimension for CP cache."""
    values = KVCache._detach_and_validate(past_key_values)
    if not values:
        seq_len = 0 if global_seq_len is None else global_seq_len
    else:
        seq_len = values[0][0].shape[-2]
    if global_seq_len is None:
        global_seq_len = seq_len
    if seq_len != global_seq_len:
        raise ValueError("global_seq_len must match full cache sequence length")
    shard_info = get_sequence_shard_info(global_seq_len, rank, world_size)
    sharded = [
        (
            key.narrow(-2, shard_info.start, shard_info.local_seq_len).contiguous(),
            value.narrow(-2, shard_info.start, shard_info.local_seq_len).contiguous(),
        )
        for key, value in values
    ]
    return sharded, shard_info


class KVCache:
    """Stores per-layer key/value tensors."""

    def __init__(self):
        self.past_key_values: Optional[Any] = None

    @property
    def is_empty(self) -> bool:
        return self.past_key_values is None or (
            isinstance(self.past_key_values, list) and len(self.past_key_values) == 0
        )

    def clear(self) -> None:
        """Drop all cached tensors."""
        self.past_key_values = None

    def update(self, past_key_values: Optional[Iterable]) -> None:
        """Replace the cache with detached past key values."""
        if past_key_values is None:
            return
        if self._is_opaque_cache(past_key_values):
            self.past_key_values = past_key_values
            return
        values = self._detach_and_validate(past_key_values)
        self.past_key_values = None if not values else values

    def merge(self, past_key_values: Optional[Iterable]) -> None:
        """Append incremental key/value tensors on the sequence dimension."""
        if past_key_values is None:
            return
        if self._is_opaque_cache(past_key_values):
            self.past_key_values = past_key_values
            return
        new_values = self._detach_and_validate(past_key_values)
        if not new_values:
            return
        if self.past_key_values is None:
            self.past_key_values = new_values
            return
        if len(self.past_key_values) != len(new_values):
            raise ValueError("past_key_values layer count mismatch")
        merged = []
        for (old_k, old_v), (new_k, new_v) in zip(self.past_key_values, new_values):
            self._validate_pair_shapes(old_k, old_v)
            self._validate_pair_shapes(new_k, new_v)
            if old_k.shape[:-2] != new_k.shape[:-2] or old_k.shape[-1] != new_k.shape[-1]:
                raise ValueError("key cache shape mismatch")
            if old_v.shape[:-2] != new_v.shape[:-2] or old_v.shape[-1] != new_v.shape[-1]:
                raise ValueError("value cache shape mismatch")
            merged.append((
                torch.cat([old_k, new_k], dim=-2),
                torch.cat([old_v, new_v], dim=-2),
            ))
        self.past_key_values = merged

    @classmethod
    def _detach_and_validate(cls, past_key_values: Iterable) -> PastKeyValues:
        values = []
        for item in past_key_values:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("each cache entry must be a (key, value) pair")
            key, value = item
            cls._validate_pair_shapes(key, value)
            values.append((key.detach(), value.detach()))
        return values

    @staticmethod
    def _validate_pair_shapes(key: torch.Tensor, value: torch.Tensor) -> None:
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise ValueError("key and value must be tensors")
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError("key and value must have shape (batch, heads, seq, dim)")
        if key.shape != value.shape:
            raise ValueError("key and value batch/heads/seq/dim dimensions must match")

    @staticmethod
    def _is_opaque_cache(past_key_values) -> bool:
        return hasattr(past_key_values, "get_seq_length") and not isinstance(
            past_key_values, (list, tuple),
        )


class ContextParallelKVCache(KVCache):
    """Stores a local sequence shard of generation KV cache."""

    def __init__(self, rank: int, world_size: int):
        super().__init__()
        if world_size <= 0:
            raise ValueError("world_size must be > 0")
        if rank < 0 or rank >= world_size:
            raise ValueError("rank must be in [0, world_size)")
        self.rank = rank
        self.world_size = world_size
        self.shard_info = get_sequence_shard_info(0, rank, world_size)

    def update_full(self, past_key_values: Optional[Iterable]) -> None:
        """Shard full prefill K/V cache and store only this rank's sequence slice."""
        if past_key_values is None:
            return
        sharded, shard_info = shard_past_key_values(
            past_key_values,
            rank=self.rank,
            world_size=self.world_size,
        )
        self.past_key_values = sharded
        self.shard_info = shard_info

    def update_local(
        self,
        past_key_values: Optional[Iterable],
        shard_info: SequenceShardInfo,
    ) -> None:
        """Store K/V tensors that are already local to this CP rank."""
        if past_key_values is None:
            return
        self._validate_shard_info(shard_info)
        values = self._detach_and_validate(past_key_values)
        self._validate_local_seq_len(values, shard_info.local_seq_len)
        self.past_key_values = values
        self.shard_info = shard_info

    def merge_local(
        self,
        past_key_values: Optional[Iterable],
        global_seq_len: Optional[int] = None,
    ) -> None:
        """Append local incremental K/V tensors and advance global sequence metadata."""
        if past_key_values is None:
            return
        new_values = self._detach_and_validate(past_key_values)
        if self.past_key_values is None:
            if global_seq_len is None and self.world_size > 1:
                raise ValueError("global_seq_len is required for initial CP local cache")
            inferred_global = (
                self.shard_info.global_seq_len + new_values[0][0].shape[-2]
                if global_seq_len is None and new_values
                else global_seq_len
            )
            shard_info = get_sequence_shard_info(
                0 if inferred_global is None else inferred_global,
                self.rank,
                self.world_size,
            )
            self._validate_local_seq_len(new_values, shard_info.local_seq_len)
            self.past_key_values = new_values
            self.shard_info = shard_info
            return
        old_local_seq_len = self.shard_info.local_seq_len
        next_global_seq_len = (
            self.shard_info.global_seq_len + new_values[0][0].shape[-2]
            if global_seq_len is None and new_values
            else global_seq_len
        )
        if next_global_seq_len is None:
            raise ValueError("global_seq_len is required for empty incremental cache")
        shard_info = get_sequence_shard_info(
            next_global_seq_len,
            self.rank,
            self.world_size,
        )
        expected_growth = shard_info.local_seq_len - old_local_seq_len
        actual_growth = new_values[0][0].shape[-2] if new_values else 0
        if expected_growth != actual_growth:
            raise ValueError("local cache growth does not match CP shard metadata")
        super().merge(new_values)
        self.shard_info = shard_info

    def clear(self) -> None:
        """Drop all cached tensors and reset CP sequence metadata."""
        super().clear()
        self.shard_info = get_sequence_shard_info(0, self.rank, self.world_size)

    def _validate_shard_info(self, shard_info: SequenceShardInfo) -> None:
        if shard_info.rank != self.rank or shard_info.world_size != self.world_size:
            raise ValueError("shard_info does not match this CP cache")

    @staticmethod
    def _validate_local_seq_len(values: PastKeyValues, local_seq_len: int) -> None:
        for key, value in values:
            if key.shape[-2] != local_seq_len or value.shape[-2] != local_seq_len:
                raise ValueError("local cache sequence length does not match shard_info")
