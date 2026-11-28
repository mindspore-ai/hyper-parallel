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
"""CPU checks for prepared CP batches and consumed-cursor checkpointing."""
from dataclasses import FrozenInstanceError
from threading import Event
import unittest

import torch

from hyper_parallel.distributed.context_parallel.batch_plan import CPBatchPlanConfig
from hyper_parallel.models.deepseek_v41.adapter.data.cp_batch import CPPlannedDataLoader


class _CursorSource:
    """Small stateful CPU source with observable speculative advancement."""

    def __init__(self, ahead_at: int, invalid_second: bool = False) -> None:
        self.cursor = 0
        self.ahead_at = ahead_at
        self.advanced = Event()
        self.invalid_second = invalid_second

    def __len__(self) -> int:
        return 9

    def __iter__(self):
        while self.cursor < len(self):
            index = self.cursor
            self.cursor += 1
            if self.cursor >= self.ahead_at:
                self.advanced.set()
            middle = 3 if index % 2 == 0 else 1
            if self.invalid_second and index == 1:
                middle = 0
            yield {"input_ids": torch.full((1, 8), index, dtype=torch.int64),
                   "cu_seq_lens": torch.tensor([0, middle, 8])}

    def state_dict(self) -> dict:
        """Snapshot this source cursor without including queued batches."""
        return {"cursor": self.cursor}

    def load_state_dict(self, state: dict) -> None:
        """Restore the cursor before rebuilding the source iterator."""
        self.cursor = state["cursor"]


class TestCPPlannedDataLoader(unittest.TestCase):
    """Keep speculative CPU work independent of the committed training cursor."""

    def setUp(self) -> None:
        """Use real host plans without any device or distributed initialization."""
        self.config = CPBatchPlanConfig(
            2, 0, ratios=(0, 1, 2), windows=(2,), mode="greedy", block_size=2,
            index_k_comm="all_to_all_v", main_kv_comm="all_to_all_v",
        )

    def test_prefetch_checkpoint_tracks_consumed_microbatches(self) -> None:
        """Feature: CPU prefetch with GAS and checkpoint restore.

        Description: Let the producer advance beyond consumed micro-batches,
            restore their saved cursor, and inspect the next payload and plan.
        Expectation: Only consumed rows are checkpointed; older plans stay valid.
        """
        for gas in (1, 2, 4):
            with self.subTest(gas=gas):
                source = _CursorSource(ahead_at=gas + 1)
                loader = CPPlannedDataLoader(source, self.config, prefetch=True)
                try:
                    iterator = iter(loader)
                    first = next(iterator)
                    for _ in range(gas - 1):
                        next(iterator)
                    snapshot = loader.state_dict()
                    self.assertTrue(source.advanced.wait(timeout=3), "producer did not prefetch")
                    self.assertEqual(snapshot, {"cursor": gas})
                    self.assertEqual(loader.state_dict(), snapshot)
                    expected = next(iterator)
                    loader.load_state_dict(snapshot)
                    restored = next(iter(loader))
                    self.assertTrue(torch.equal(expected.payload["input_ids"], restored.payload["input_ids"]))
                    self.assertEqual(expected.plan, restored.plan)
                    self.assertEqual(int(restored.payload["input_ids"][0, 0]), gas)
                    self.assertEqual(first.plan.lengths, (3, 5))
                    with self.assertRaises(FrozenInstanceError):
                        first.plan.lengths = (1, 7)
                finally:
                    loader.close()

    def test_producer_failure_reaches_consumer(self) -> None:
        """Feature: Background planning failure propagation.

        Description: A following CPU batch contains a zero-length document.
        Expectation: The consumer receives the validation error and can close.
        """
        loader = CPPlannedDataLoader(_CursorSource(2, invalid_second=True), self.config, prefetch=True)
        try:
            iterator = iter(loader)
            next(iterator)
            with self.assertRaisesRegex(ValueError, "host integers"):
                next(iterator)
        finally:
            loader.close()
