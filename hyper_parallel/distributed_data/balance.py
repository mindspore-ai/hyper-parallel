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
"""Deterministic inter-microbatch load balancing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from hyper_parallel.distributed_data.schema import LocalBatchMeta, WorkloadCost


@dataclass(frozen=True)
class BalanceItem:
    """One metadata item and its planner cost."""

    metadata: LocalBatchMeta
    source_position: int
    cost: WorkloadCost


@dataclass
class _BalanceSlot:
    items: list[BalanceItem] = field(default_factory=list)
    cost: WorkloadCost = field(default_factory=WorkloadCost)


class BatchBalancer(Protocol):
    """Assign a fixed candidate set to fixed-capacity execution slots."""

    def balance(
        self,
        items: tuple[BalanceItem, ...],
        slot_count: int,
        slot_capacity: int,
    ) -> tuple[tuple[BalanceItem, ...], ...]:
        """Return one ordered item tuple per execution slot."""


class GreedyBatchBalancer:
    """Deterministic longest-processing-time greedy balancer."""

    def balance(
        self,
        items: tuple[BalanceItem, ...],
        slot_count: int,
        slot_capacity: int,
    ) -> tuple[tuple[BalanceItem, ...], ...]:
        """Place expensive local batches into the cheapest non-full slot.

        Args:
            items: Candidate local batches and their normalized costs.
            slot_count: Number of ``(microbatch, data_rank)`` slots.
            slot_capacity: Local batches in every slot.

        Returns:
            Balanced slots, with local batches in stable source order per slot.
        """
        if slot_count < 1 or slot_capacity < 1:
            raise ValueError(
                f"slot_count and slot_capacity must be positive, but got {slot_count} and {slot_capacity}."
            )
        expected = slot_count * slot_capacity
        if len(items) != expected:
            raise ValueError(f"Balancer expected {expected} items, but got {len(items)}.")

        ordered_items = sorted(
            items,
            key=lambda item: (
                -item.cost.dominant,
                -item.cost.total,
                0 if isinstance(item.metadata.local_batch_id, int) else 1,
                str(item.metadata.local_batch_id),
                item.source_position,
            ),
        )
        slots = [_BalanceSlot() for _ in range(slot_count)]
        for item in ordered_items:
            available = (
                (slot_index, slot)
                for slot_index, slot in enumerate(slots)
                if len(slot.items) < slot_capacity
            )
            slot_index, selected = min(
                available,
                key=lambda indexed_slot: self._placement_score(indexed_slot[1], item, indexed_slot[0]),
            )
            del slot_index
            selected.items.append(item)
            selected.cost = selected.cost + item.cost

        return tuple(
            tuple(sorted(slot.items, key=lambda item: item.source_position))
            for slot in slots
        )

    @staticmethod
    def _placement_score(slot: _BalanceSlot, item: BalanceItem, slot_index: int) -> tuple[float, ...]:
        projected = slot.cost + item.cost
        return (
            projected.dominant,
            projected.total,
            slot.cost.dominant,
            slot.cost.total,
            len(slot.items),
            float(slot_index),
        )
