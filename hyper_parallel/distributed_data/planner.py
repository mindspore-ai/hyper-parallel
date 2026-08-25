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
"""Whole-step and online-microbatch planning from lightweight metadata."""

from __future__ import annotations

import hashlib
import json
from typing import Sequence

from hyper_parallel.distributed_data.balance import BalanceItem, BatchBalancer, GreedyBatchBalancer
from hyper_parallel.distributed_data.cost_model import CostModel, LinearMultimodalCostModel
from hyper_parallel.distributed_data.schema import BatchPlan, PlannedSample, SampleMeta, TensorShardSpec


class DistributedBatchPlanner:
    """Plan DP-rank slots across a whole step or one online microbatch."""

    def __init__(
        self,
        data_parallel_size: int,
        micro_batch_size: int,
        micro_batch_num: int,
        *,
        cost_model: CostModel | None = None,
        balancer: BatchBalancer | None = None,
        cp_shards: tuple[TensorShardSpec, ...] = (),
    ) -> None:
        """Initialize fixed optimizer-step dimensions and planning policies."""
        for name, value in (
            ("data_parallel_size", data_parallel_size),
            ("micro_batch_size", micro_batch_size),
            ("micro_batch_num", micro_batch_num),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer, but got {value!r}.")
        self.data_parallel_size = data_parallel_size
        self.micro_batch_size = micro_batch_size
        self.micro_batch_num = micro_batch_num
        self.cost_model = cost_model or LinearMultimodalCostModel()
        self.balancer = balancer or GreedyBatchBalancer()
        self.cp_shards = cp_shards

    @property
    def local_samples_per_step(self) -> int:
        """Return candidate samples contributed by each data owner per step."""
        return self.micro_batch_size * self.micro_batch_num

    @property
    def global_samples_per_step(self) -> int:
        """Return total candidates required to plan one optimizer step."""
        return self.local_samples_per_step * self.data_parallel_size

    def plan(
        self,
        candidates: Sequence[SampleMeta],
        *,
        step: int,
        cursor_start: int,
    ) -> BatchPlan:
        """Build a deterministic plan without fetching sample data.

        Args:
            candidates: Metadata for the complete global optimizer-step batch.
            step: Logical optimizer-step index.
            cursor_start: Per-owner candidate cursor before this plan.

        Returns:
            A deterministic :class:`BatchPlan`.
        """
        return self._plan(
            candidates,
            step=step,
            cursor_start=cursor_start,
            micro_batch_start=0,
            micro_batch_num=self.micro_batch_num,
        )

    def plan_microbatch(
        self,
        candidates: Sequence[SampleMeta],
        *,
        step: int,
        cursor_start: int,
        micro_batch_index: int,
    ) -> BatchPlan:
        """Build one online microbatch plan when later metadata is unavailable.

        Args:
            candidates: Metadata for one complete global microbatch.
            step: Logical optimizer-step index.
            cursor_start: Per-owner candidate cursor before this microbatch.
            micro_batch_index: Microbatch position within the optimizer step.

        Returns:
            A deterministic one-microbatch :class:`BatchPlan`.
        """
        if (
            not isinstance(micro_batch_index, int)
            or isinstance(micro_batch_index, bool)
            or micro_batch_index < 0
            or micro_batch_index >= self.micro_batch_num
        ):
            raise ValueError(
                f"micro_batch_index must be in [0, {self.micro_batch_num}), but got {micro_batch_index!r}."
            )
        return self._plan(
            candidates,
            step=step,
            cursor_start=cursor_start,
            micro_batch_start=micro_batch_index,
            micro_batch_num=1,
        )

    def _plan(
        self,
        candidates: Sequence[SampleMeta],
        *,
        step: int,
        cursor_start: int,
        micro_batch_start: int,
        micro_batch_num: int,
    ) -> BatchPlan:
        if step < 0 or cursor_start < 0:
            raise ValueError(f"step and cursor_start must be non-negative, but got {step} and {cursor_start}.")
        expected_candidates = self.data_parallel_size * self.micro_batch_size * micro_batch_num
        if len(candidates) != expected_candidates:
            raise ValueError(
                f"Planner requires {expected_candidates} candidates for this planning window, "
                f"but got {len(candidates)}."
            )
        sample_ids = [metadata.sample_id for metadata in candidates]
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("Planner candidates must have unique sample_id values.")

        items = tuple(
            BalanceItem(metadata=metadata, source_position=position, cost=self.cost_model.estimate(metadata))
            for position, metadata in enumerate(candidates)
        )
        slot_count = self.data_parallel_size * micro_batch_num
        slots = self.balancer.balance(items, slot_count, self.micro_batch_size)

        planned_samples = []
        for slot_index, slot in enumerate(slots):
            micro_batch_index = micro_batch_start + slot_index // self.data_parallel_size
            target_data_rank = slot_index % self.data_parallel_size
            for position_in_micro_batch, item in enumerate(slot):
                planned_samples.append(
                    PlannedSample(
                        meta=item.metadata,
                        source_position=item.source_position,
                        target_data_rank=target_data_rank,
                        micro_batch_index=micro_batch_index,
                        position_in_micro_batch=position_in_micro_batch,
                        cost=item.cost,
                    )
                )

        cursor_end = cursor_start + self.micro_batch_size * micro_batch_num
        replay_id = self._replay_id(
            step,
            cursor_start,
            micro_batch_start,
            micro_batch_num,
            tuple(planned_samples),
        )
        return BatchPlan(
            replay_id=replay_id,
            step=step,
            cursor_start=cursor_start,
            cursor_end=cursor_end,
            data_parallel_size=self.data_parallel_size,
            micro_batch_size=self.micro_batch_size,
            micro_batch_num=micro_batch_num,
            samples=tuple(planned_samples),
            cp_shards=self.cp_shards,
            micro_batch_start=micro_batch_start,
        )

    def _replay_id(
        self,
        step: int,
        cursor_start: int,
        micro_batch_start: int,
        micro_batch_num: int,
        samples: tuple[PlannedSample, ...],
    ) -> str:
        stable_plan = {
            "step": step,
            "cursor_start": cursor_start,
            "data_parallel_size": self.data_parallel_size,
            "micro_batch_size": self.micro_batch_size,
            "micro_batch_start": micro_batch_start,
            "micro_batch_num": micro_batch_num,
            "samples": [
                {
                    "sample_id": sample.meta.sample_id,
                    "target_data_rank": sample.target_data_rank,
                    "micro_batch_index": sample.micro_batch_index,
                    "position_in_micro_batch": sample.position_in_micro_batch,
                }
                for sample in samples
            ],
            "cp_shards": [
                {"path": list(spec.path), "dim": spec.dim}
                for spec in self.cp_shards
            ],
        }
        encoded = json.dumps(stable_plan, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:24]
