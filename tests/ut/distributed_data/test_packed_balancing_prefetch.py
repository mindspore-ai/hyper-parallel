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
"""Tests for process-backed external delivery and staging lifecycle."""

import threading
import unittest
from typing import Any
from unittest.mock import patch

from hyper_parallel.distributed_data import (
    CausalAttentionCostModel, DistributedDatasetConfig, SampleMetadata, build_distributed_dataloader,
)


from tests.common.mark_utils import arg_mark


class _Mesh:
    mesh_shape = (1,)
    mesh_dim_names = ("dp",)
    rank_list = (0,)


def _loader():
    return build_distributed_dataloader(
        None, _Mesh(), DistributedDatasetConfig(seq_len=10, local_batch_size=1, communication_backend="gloo"),
        external_step_source=[[[0]], [[1]]], metadata_fn=lambda _: SampleMetadata(1),
        pack_fn=lambda samples, _: samples, collate_fn=list, cost_model=CausalAttentionCostModel(), device="cpu",
    )


class TestPackedBalancingPrefetch(unittest.TestCase):
    """Verify order, nonblocking requests, errors and deterministic cleanup."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_hooks_are_idempotent_and_publish_only_delivered_stats(self) -> None:
        """Feature: External prefetch.
        Description: Repeat both hooks before consuming the initial result.
        Expectation: Only one request is pending and stats advance at delivery.
        """
        loader = _loader()
        try:
            loader.prefetch_plan()
            pending = loader._future
            loader.prefetch()
            loader.prefetch_plan()
            self.assertIs(loader._future, pending)
            loader.wait_for_prefetch()
            self.assertEqual(loader.step, 0)
            self.assertIsNone(loader.last_balance_stats)
            self.assertEqual(next(loader), [[0]])
            self.assertEqual(loader.step, 1)
            self.assertEqual(next(loader), [[1]])
            with self.assertRaises(StopIteration):
                next(loader)
        finally:
            loader.close()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_preparation_does_not_block_prefetch(self) -> None:
        """Feature: Nonblocking preparation.
        Description: Gate metadata, planning and collation separately.
        Expectation: Prefetch returns while preparation remains pending.
        """
        for method in ("_read_step", "_make_plan", "_finish_batch"):
            with self.subTest(method=method):
                loader = _loader()
                started, release = threading.Event(), threading.Event()
                original = getattr(loader._source, method)

                def delayed(*args: Any) -> Any:
                    """Gate source work until the test releases it."""
                    started.set()
                    if not release.wait(5):
                        raise RuntimeError("test preparation timed out")
                    return original(*args)

                try:
                    with patch.object(loader._source, method, side_effect=delayed):
                        loader.prefetch()
                        self.assertTrue(started.wait(5))
                        self.assertFalse(loader._future.done())
                        self.assertEqual(loader.step, 0)
                        release.set()
                        self.assertEqual(next(loader), [[0]])
                finally:
                    release.set()
                    loader.close()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_speculative_errors_reach_next_consumer(self) -> None:
        """Feature: Ordered error delivery.
        Description: Fail the successor after the current batch is ready.
        Expectation: The current batch is delivered before the successor error.
        """
        for method in ("_read_step", "_make_plan", "_finish_batch"):
            with self.subTest(method=method):
                loader = _loader()
                try:
                    loader.prefetch()
                    loader.wait_for_prefetch()
                    with patch.object(loader._source, method, side_effect=ValueError("worker failed")):
                        self.assertEqual(next(loader), [[0]])
                        with self.assertRaisesRegex(ValueError, "worker failed"):
                            next(loader)
                finally:
                    loader.close()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_exhaustion_epoch_and_early_close(self) -> None:
        """Feature: Process lifecycle.
        Description: Exhaust, re-enter, reset and close an external loader.
        Expectation: Epoch replay is exact and close is idempotent.
        """
        loader = _loader()
        try:
            self.assertEqual(list(loader), [[[0]], [[1]]])
            self.assertEqual(list(loader), [[[0]], [[1]]])
            loader.set_epoch(1)
            self.assertEqual(next(loader), [[0]])
            loader.close()
            loader.close()
            with self.assertRaises(StopIteration):
                next(loader)
        finally:
            loader.close()
