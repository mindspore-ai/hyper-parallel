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
"""Regression tests for shuffle-before-truncation cyclic sampling."""

import random
import unittest

from hyper_parallel.data.parallel.batch_sampler import build_dataset_batch_sampler


class TestCyclicBatchSampler(unittest.TestCase):
    """Check deterministic coverage and checkpoint cursors without devices."""

    @staticmethod
    def _sampler(rank=0, total=19, sharding=False):
        return build_dataset_batch_sampler(
            total_samples=total, micro_batch_size=2, global_batch_size=8,
            dp_rank=rank, dp_world_size=4, sampler_type="cyclic",
            data_sharding=sharding, seed=42, drop_last=True,
        )

    def _indices(self, rank=0, total=19, epoch=0, sharding=False):
        sampler = self._sampler(rank, total, sharding)
        sampler.set_epoch(epoch)
        return [index for batch in sampler for index in batch]

    def test_epoch_is_reproducible(self):
        """Equal seed and epoch reproduce the full-shuffle DP slice."""
        for epoch in range(3):
            expected = list(range(19))
            random.Random(42 + epoch).shuffle(expected)
            for rank in range(4):
                with self.subTest(epoch=epoch, rank=rank):
                    actual = self._indices(rank, epoch=epoch)
                    self.assertEqual(actual, expected[:16][rank::4])
                    self.assertEqual(actual, self._indices(rank, epoch=epoch))

    def test_dropped_samples_change_and_tail_is_reachable(self):
        """Fixed test epochs rotate omitted IDs and include original tail IDs."""
        seen = set()
        dropped = set()
        for epoch in range(10):
            selected = {index for rank in range(4) for index in self._indices(rank, epoch=epoch)}
            seen.update(selected)
            dropped.add(tuple(sorted(set(range(19)) - selected)))
        self.assertGreater(len(dropped), 1)
        self.assertEqual(seen, set(range(19)))

    def test_dp_ranks_are_disjoint(self):
        """Every complete micro-batch has unique samples across logical DP ranks."""
        for total in (8, 9, 19, 24):
            with self.subTest(total=total):
                selected = [index for rank in range(4) for index in self._indices(rank, total)]
                self.assertEqual(len(selected), total // 8 * 8)
                self.assertEqual(len(set(selected)), len(selected))
                self.assertTrue(all(0 <= index < total for index in selected))

    def test_resume_preserves_remaining_order(self):
        """Restoring each optimizer boundary reproduces uninterrupted suffixes."""
        for epoch in (0, 2):
            for rank in range(4):
                for consumed_batches in (0, 1, 2):
                    with self.subTest(epoch=epoch, rank=rank, consumed=consumed_batches):
                        source = self._sampler(rank)
                        source.set_epoch(epoch)
                        iterator = iter(source)
                        for _ in range(consumed_batches):
                            next(iterator)
                        state = source.state_dict()
                        expected = list(iterator)
                        restored = self._sampler(rank)
                        restored.load_state_dict(state)
                        restored.set_epoch(epoch)
                        self.assertEqual(list(restored), expected)
                        restored.set_epoch(epoch + 1)
                        self.assertEqual(
                            [index for batch in restored for index in batch],
                            self._indices(rank, epoch=epoch + 1),
                        )

    def test_divisible_dataset_retains_previous_order(self):
        """Without a partial tail the previous full-region shuffle is unchanged."""
        for epoch in (0, 1):
            expected = list(range(24))
            random.Random(42 + epoch).shuffle(expected)
            for rank in range(4):
                self.assertEqual(self._indices(rank, 24, epoch), expected[rank::4])

    def test_sharded_bucket_order_is_unchanged(self):
        """The independent fixed-bucket shuffling branch retains its policy."""
        for rank in range(4):
            offsets = list(range(4))
            random.Random(43).shuffle(offsets)
            expected = [rank * 4 + offset for offset in offsets]
            self.assertEqual(self._indices(rank, 19, 1, True), expected)
