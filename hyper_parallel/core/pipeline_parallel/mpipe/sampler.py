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
"""Per-PP-rank disk-sharded sampler for MPipe + ``data.load: single``.

Framework-agnostic: only manipulates integer indices, never touches sample
data.
"""
from typing import Any, Dict, Iterable, Iterator


class PPRankOwnedSampler:
    """Filter a DP-sharded base sampler to indices for ``owned_micros`` only.

    Used with MPipe + ``data.load: "single"`` to shard disk reads across PP
    ranks. The base sampler emits a deterministic stream of global indices
    (same seed on every rank); within each pipeline step the next
    ``pp_micro_batch_num * micro_batch_size`` indices are laid out
    micro-major, and this wrapper yields only the slots in ``owned_micros``.
    """

    def __init__(self, base_sampler: Iterable[int], owned_micros: Iterable[int],
                 micro_batch_size: int, pp_micro_batch_num: int) -> None:
        """Wrap ``base_sampler``, keeping only this rank's micro-batch slots.

        Args:
            base_sampler (Iterable[int]): Deterministic global-index stream.
            owned_micros (Iterable[int]): Micro indices this rank loads.
            micro_batch_size (int): Samples per micro-batch.
            pp_micro_batch_num (int): Micro-batches per pipeline step.
        """
        self.base = base_sampler
        self.owned_micros = sorted(owned_micros)
        self.micro_bs = int(micro_batch_size)
        self.step_size = int(pp_micro_batch_num) * self.micro_bs

    def __iter__(self) -> Iterator[int]:
        """Yield only the owned slots of each micro-major step chunk."""
        chunk = []
        for idx in self.base:
            chunk.append(idx)
            if len(chunk) == self.step_size:
                for m in self.owned_micros:
                    yield from chunk[m * self.micro_bs : (m + 1) * self.micro_bs]
                chunk = []

    def __len__(self) -> int:
        """Number of indices yielded per epoch (trailing partial step dropped)."""
        full_chunks = len(self.base) // self.step_size
        return full_chunks * len(self.owned_micros) * self.micro_bs

    def set_epoch(self, epoch: int) -> None:
        """Forward the epoch to the base sampler when it supports reseeding."""
        if hasattr(self.base, "set_epoch"):
            self.base.set_epoch(epoch)

    def state_dict(self) -> Dict[str, Any]:
        """Return the base sampler's state, or an empty dict if it has none."""
        if hasattr(self.base, "state_dict"):
            return {"base": self.base.state_dict()}
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore the base sampler's state from :meth:`state_dict` output."""
        if "base" in state and hasattr(self.base, "load_state_dict"):
            self.base.load_state_dict(state["base"])


def mpipe_owned_micros(pp_size: int, micro_batch_num: int, pp_rank: int,
                       mode: str = "full") -> frozenset:
    """Micro indices PP rank ``pp_rank`` loads under MPipe (``NT = min(PP, M)``).

    ``mode="full"`` (default): round-robin — rank ``i < NT`` owns
    ``{m | m % NT == i}``. ``mode="min"``: rank 0 owns ``{0, NT..M-1}``,
    ranks ``1..NT-1`` own ``{rank}``. Ranks ``i >= NT`` own nothing; for
    ``M <= NT`` both modes collapse to ``{rank}`` per rank.
    """
    nt = min(pp_size, micro_batch_num)
    if pp_rank >= nt:
        return frozenset()
    if mode == "full":
        return frozenset(m for m in range(micro_batch_num) if m % nt == pp_rank)
    if mode == "min":
        if pp_rank == 0:
            return frozenset({0, *range(nt, micro_batch_num)})
        return frozenset({pp_rank})
    raise ValueError(f"mpipe transpose overflow mode must be 'full' or 'min', got {mode!r}")
