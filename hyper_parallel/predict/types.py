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
"""Task-independent contracts for offline prediction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Protocol, TypeVar

import torch.distributed as dist

_T = TypeVar("_T")


class PredictionSettings(Protocol):
    """Structural config interface; prediction never imports the task host."""

    input: str | None
    output_file: str | None
    batch_size: int
    max_batch_tokens: int | None
    bucket_by_length: bool
    overwrite: bool


@dataclass(frozen=True)
class PredictionRequest:
    """One uniquely identified request with stable input ordering."""

    input_id: str | int
    ordinal: int
    prompt: str | None = None
    messages: list[dict[str, Any]] | None = None


@dataclass
class PredictionSummary:
    """Result of a committed prediction run, identical on all lanes."""

    num_requests: int
    num_samples: int
    output_file: str
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable report."""
        return asdict(self)


@dataclass
class PredictionContext:
    """Explicit DP ownership; the default is a single independent process."""

    rank: int = 0
    world_size: int = 1
    group: Any = None

    def __post_init__(self) -> None:
        """Reject invalid ownership before reading inputs."""
        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise ValueError("prediction rank must be in [0, world_size)")
        if self.world_size > 1 and (
                not dist.is_initialized()
                or dist.get_world_size(self.group) != self.world_size
                or dist.get_rank(self.group) != self.rank):
            raise ValueError("PredictionContext must match its initialized process group")

    def exchange(self, value: Any) -> list[Any]:
        """Exchange small control records after each independent phase.

        Args:
            value: Small picklable control record contributed by this prediction lane.

        Returns:
            One record per lane in group-rank order, or ``[value]`` for a single lane.
        """
        if self.world_size == 1:
            return [value]
        values = [None] * self.world_size
        dist.all_gather_object(values, value, group=self.group)
        return values

    def phase(self, operation: Callable[[], _T]) -> _T:
        """Propagate recoverable lane failures before any rank enters the next phase.

        A crashed process is handled by the launcher's process-group timeout.

        Args:
            operation: Zero-argument callable performing this lane's independent work.

        Returns:
            The local operation's result after every lane reports success.
        """
        result = None
        error = None
        try:
            result = operation()
        except Exception as exc:
            if self.world_size == 1:
                raise
            error = f"rank {self.rank}: {type(exc).__name__}: {exc}"
        errors = [value for value in self.exchange(error) if value is not None]
        if errors:
            raise RuntimeError("Prediction phase failed: " + "; ".join(errors))
        return result
