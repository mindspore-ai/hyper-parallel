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
"""Whole-step planning from lightweight local-batch metadata."""

from __future__ import annotations

import hashlib
import json
from typing import Sequence

from hyper_parallel.distributed_data.balance import BalanceItem, BatchBalancer, GreedyBatchBalancer
from hyper_parallel.distributed_data.cost_model import CostModel, LinearMultimodalCostModel
from hyper_parallel.distributed_data.schema import BatchPlan, LocalBatchMeta, PlannedLocalBatch, TensorShardSpec


class DistributedBatchPlanner:
    """Assign one complete local batch to every DP-rank execution slot."""

    def __init__(
        self,
        data_parallel_size: int,
        micro_batch_num: int,
        *,
        cost_model: CostModel | None = None,
        balancer: BatchBalancer | None = None,
        cp_shards: tuple[TensorShardSpec, ...] = (),
    ) -> None:
        """Initialize fixed execution dimensions and planning policies."""
        for name, value in (
            ("data_parallel_size", data_parallel_size),
            ("micro_batch_num", micro_batch_num),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer, but got {value!r}.")
        self.data_parallel_size = data_parallel_size
        self.micro_batch_num = micro_batch_num
        self.cost_model = cost_model or LinearMultimodalCostModel()
        self.balancer = balancer or GreedyBatchBalancer()
        self.cp_shards = cp_shards

    @property
    def local_batches_per_step(self) -> int:
        """Return local batches contributed by each data owner per step."""
        return self.micro_batch_num

    @property
    def global_local_batches_per_step(self) -> int:
        """Return total local batches required to plan one optimizer step."""
        return self.local_batches_per_step * self.data_parallel_size

    def plan(
        self,
        candidates: Sequence[LocalBatchMeta],
        *,
        step: int,
        local_batch_offset_start: int,
    ) -> BatchPlan:
        """Build a deterministic plan without inspecting local-batch payloads.

        Args:
            candidates: Metadata for all rank-local batches in one optimizer step.
            step: Logical optimizer-step index.
            local_batch_offset_start: Inclusive per-owner local-batch offset.

        Returns:
            A deterministic :class:`BatchPlan`.
        """
        return self._plan(candidates, step=step, local_batch_offset_start=local_batch_offset_start)

    def _plan(
        self,
        candidates: Sequence[LocalBatchMeta],
        *,
        step: int,
        local_batch_offset_start: int,
    ) -> BatchPlan:
        if step < 0 or local_batch_offset_start < 0:
            raise ValueError(
                "step and local_batch_offset_start must be non-negative, "
                f"but got {step} and {local_batch_offset_start}."
            )
        expected_candidates = self.global_local_batches_per_step
        if len(candidates) != expected_candidates:
            raise ValueError(
                f"Planner requires {expected_candidates} candidates for this planning window, "
                f"but got {len(candidates)}."
            )
        local_batch_ids = [metadata.local_batch_id for metadata in candidates]
        if len(set(local_batch_ids)) != len(local_batch_ids):
            raise ValueError("Planner candidates must have unique local_batch_id values.")

        items = tuple(
            BalanceItem(metadata=metadata, source_position=position, cost=self.cost_model.estimate(metadata))
            for position, metadata in enumerate(candidates)
        )
        slot_count = self.global_local_batches_per_step
        slots = self.balancer.balance(items, slot_count, 1)

        planned_local_batches = []
        for slot_index, slot in enumerate(slots):
            micro_batch_index = slot_index // self.data_parallel_size
            target_data_rank = slot_index % self.data_parallel_size
            item = slot[0]
            planned_local_batches.append(
                PlannedLocalBatch(
                    meta=item.metadata,
                    source_position=item.source_position,
                    target_data_rank=target_data_rank,
                    micro_batch_index=micro_batch_index,
                    cost=item.cost,
                )
            )

        local_batch_offset_end = local_batch_offset_start + self.micro_batch_num
        plan_id = self._plan_id(
            step,
            local_batch_offset_start,
            tuple(planned_local_batches),
        )
        return BatchPlan(
            plan_id=plan_id,
            step=step,
            local_batch_offset_start=local_batch_offset_start,
            local_batch_offset_end=local_batch_offset_end,
            data_parallel_size=self.data_parallel_size,
            micro_batch_num=self.micro_batch_num,
            local_batches=tuple(planned_local_batches),
            cp_shards=self.cp_shards,
        )

    def _plan_id(
        self,
        step: int,
        local_batch_offset_start: int,
        local_batches: tuple[PlannedLocalBatch, ...],
    ) -> str:
        stable_plan = {
            "step": step,
            "local_batch_offset_start": local_batch_offset_start,
            "data_parallel_size": self.data_parallel_size,
            "micro_batch_num": self.micro_batch_num,
            "local_batches": [
                {
                    "local_batch_id": local_batch.meta.local_batch_id,
                    "target_data_rank": local_batch.target_data_rank,
                    "micro_batch_index": local_batch.micro_batch_index,
                }
                for local_batch in local_batches
            ],
            "cp_shards": [
                {"path": list(spec.path), "dim": spec.dim}
                for spec in self.cp_shards
            ],
        }
        encoded = json.dumps(stable_plan, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:24]
