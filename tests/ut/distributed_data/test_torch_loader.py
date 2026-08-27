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
"""Unit tests for the native PyTorch local data-loading backend."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from hyper_parallel.distributed_data.planner import DistributedBatchPlanner
from hyper_parallel.distributed_data.schema import SampleMeta
from hyper_parallel.distributed_data.torch_loader import TorchLocalDataLoader
from tests.common.mark_utils import arg_mark

_WORKER_ID_ENV = "HYPER_DATA_TEST_WORKER_ID"


class _WorkerDataset:
    """Return process identity with every synthetic map-style sample."""

    def __len__(self) -> int:
        """Return enough synthetic entries to include the failure sentinel."""
        return 100

    def __getitem__(self, sample_id: int) -> tuple[int, int, str | None]:
        """Return one sample or raise the synthetic failure sentinel."""
        if sample_id == 99:
            raise ValueError("synthetic worker failure")
        return sample_id, os.getpid(), os.environ.get(_WORKER_ID_ENV)


def _metadata_fn(sample: tuple[int, int, str | None], sample_id: int) -> SampleMeta:
    """Derive deterministic metadata inside a DataLoader worker."""
    return SampleMeta(sample_id=sample_id, text_tokens=sample[0] + 1)


def _collate_fn(samples: list[tuple[int, int, str | None]]) -> dict[str, object]:
    """Record that final sidecar collation runs in a DataLoader worker."""
    return {
        "sample_ids": tuple(sample[0] for sample in samples),
        "sample_pids": tuple(sample[1] for sample in samples),
        "worker_ids": tuple(sample[2] for sample in samples),
        "collate_pid": os.getpid(),
    }


def _worker_init_fn(worker_id: int) -> None:
    """Publish the deterministic local worker ID to the synthetic dataset."""
    os.environ[_WORKER_ID_ENV] = str(worker_id)


class TestTorchLocalDataLoader(unittest.TestCase):
    """Validate planned and online reads through native DataLoader workers."""

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_sidecar_plan_uses_sample_tasks_and_preserves_batch_order(self) -> None:
        """
        Feature: Owner-local data loading
        Description: Multiple workers should load samples before caller-side final collation.
        Expectation: The operation produces the expected data, ordering, state, or error.
        """
        planner = DistributedBatchPlanner(data_parallel_size=1, raw_sample_size=2, micro_batch_num=2)
        plan = planner.plan(
            tuple(SampleMeta(sample_id=index) for index in range(4)),
            step=0,
            sample_offset_start=0,
        )
        loader = TorchLocalDataLoader(
            _WorkerDataset(),
            metadata_fn=None,
            collate_fn=_collate_fn,
            num_workers=2,
            prefetch_factor=2,
            pin_memory=False,
            worker_init_fn=_worker_init_fn,
            online_metadata=False,
        )
        try:
            batches = [loader.fetch(plan, 0, index) for index in range(2)]
            next_plan = planner.plan(
                tuple(SampleMeta(sample_id=index) for index in range(4, 8)),
                step=1,
                sample_offset_start=4,
            )
            next_batch = loader.fetch(next_plan, 0, 0)

            expected_ids = [
                tuple(sample.meta.sample_id for sample in plan.samples_for(0, index))
                for index in range(2)
            ]
            self.assertEqual([batch["sample_ids"] for batch in batches], expected_ids)
            self.assertEqual(
                next_batch["sample_ids"],
                tuple(sample.meta.sample_id for sample in next_plan.samples_for(0, 0)),
            )
            self.assertTrue(all(batch["collate_pid"] == os.getpid() for batch in batches))
            self.assertTrue(all(set(batch["worker_ids"]) == {"0", "1"} for batch in batches))
            self.assertTrue(all(len(set(batch["sample_pids"])) == 2 for batch in batches))
        finally:
            loader.close()

        loader.close()

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_online_metadata_runs_per_sample_in_workers_without_pinning_candidates(self) -> None:
        """
        Feature: Owner-local data loading
        Description: Online reads should use workers but return raw candidate samples for planning.
        Expectation: The operation produces the expected data, ordering, state, or error.
        """
        loader = TorchLocalDataLoader(
            _WorkerDataset(),
            metadata_fn=_metadata_fn,
            collate_fn=None,
            num_workers=2,
            prefetch_factor=2,
            pin_memory=False,
            worker_init_fn=_worker_init_fn,
            online_metadata=True,
        )
        try:
            loaded = loader.load_online((4, 5, 6, 7))
            next_loaded = loader.load_online((8, 9))

            self.assertEqual([sample.metadata.sample_id for sample in loaded], [4, 5, 6, 7])
            self.assertEqual([sample.metadata.text_tokens for sample in loaded], [5, 6, 7, 8])
            self.assertEqual([sample.data[2] for sample in loaded], ["0", "1", "0", "1"])
            self.assertNotIn(os.getpid(), {sample.data[1] for sample in loaded})
            self.assertEqual([sample.metadata.sample_id for sample in next_loaded], [8, 9])
        finally:
            loader.close()

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_native_worker_failure_reaches_the_caller(self) -> None:
        """
        Feature: Owner-local data loading
        Description: DataLoader should retain its native worker traceback and exception type.
        Expectation: The operation produces the expected data, ordering, state, or error.
        """
        loader = TorchLocalDataLoader(
            _WorkerDataset(),
            metadata_fn=_metadata_fn,
            collate_fn=None,
            num_workers=1,
            prefetch_factor=1,
            pin_memory=False,
            worker_init_fn=None,
            online_metadata=True,
        )
        try:
            with self.assertRaisesRegex(ValueError, "synthetic worker failure"):
                loader.load_online((99,))
        finally:
            loader.close()

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_forwards_native_worker_prefetch_options(self) -> None:
        """
        Feature: Owner-local data loading
        Description: Worker queues should be native while final-batch pinning stays external.
        Expectation: The operation produces the expected data, ordering, state, or error.
        """
        with patch("torch.utils.data.DataLoader") as dataloader_type:
            loader = TorchLocalDataLoader(
                _WorkerDataset(),
                metadata_fn=None,
                collate_fn=_collate_fn,
                num_workers=3,
                prefetch_factor=4,
                pin_memory=True,
                worker_init_fn=_worker_init_fn,
                online_metadata=False,
            )

        try:
            _, kwargs = dataloader_type.call_args
            self.assertEqual(kwargs["num_workers"], 3)
            self.assertEqual(kwargs["prefetch_factor"], 4)
            self.assertTrue(kwargs["persistent_workers"])
            self.assertEqual(kwargs["multiprocessing_context"], "spawn")
            self.assertFalse(kwargs["pin_memory"])
        finally:
            loader.close()

    @arg_mark(
        plat_marks=["cpu_linux"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_rejects_pinning_before_online_redistribution(self) -> None:
        """
        Feature: Owner-local data loading
        Description: Raw online candidates should not consume the final-batch pin-memory budget.
        Expectation: The operation produces the expected data, ordering, state, or error.
        """
        with self.assertRaisesRegex(ValueError, "only after planning and redistribution"):
            TorchLocalDataLoader(
                _WorkerDataset(),
                metadata_fn=_metadata_fn,
                collate_fn=None,
                num_workers=0,
                prefetch_factor=2,
                pin_memory=True,
                worker_init_fn=None,
                online_metadata=True,
            )
