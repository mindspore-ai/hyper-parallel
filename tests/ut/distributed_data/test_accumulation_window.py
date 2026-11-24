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
"""Native accumulation-window membership, planning, and checkpoint replay."""

from __future__ import annotations

import unittest
from itertools import product
from threading import Event, current_thread
from contextlib import nullcontext
from unittest.mock import Mock, PropertyMock, patch

import torch
from torch.utils.data import DataLoader

from hyper_parallel.data.parallel import build_dataset_batch_sampler
from hyper_parallel.distributed_data import DistributedDatasetConfig, SampleMetadata, WorkloadCost
from hyper_parallel.distributed_data import build_distributed_dataloader
from hyper_parallel.distributed_data.batch_sampler import _ReaderPlanInput
from hyper_parallel.distributed_data.planner import DynamicPackingPlanner
from hyper_parallel.distributed_data.schema import SampleKey
from hyper_parallel.distributed_data.transport import DataPlaneTransport


class _Mesh:
    mesh_shape = (1,)
    mesh_dim_names = ("dp",)
    rank_list = (0,)


def _loader(
        *, metadata_mode: bool = False, window: int = 3, broadcast_plan: bool = True,
        samples: object = None, accumulation: int = 10, workers: int = 0, persistent_workers: bool = True,
        device: str = "cpu", log_balance: bool = False, device_prefetch_depth: int = 1,
) -> object:
    """Build the native loader with independent planning and optimizer windows."""
    if samples is None:
        samples = [{"id": index} for index in range(44)]
    sampler = build_dataset_batch_sampler(
        total_samples=len(samples), micro_batch_size=2, global_batch_size=2 * accumulation, dp_world_size=1, dp_rank=0,
    )
    options = {"metadata": [SampleMetadata(1) for _ in range(len(samples))]} if metadata_mode else {
        "metadata_fn": lambda _: SampleMetadata(1),
    }
    loader = build_distributed_dataloader(
        samples, _Mesh(),
        DistributedDatasetConfig(
            seq_len=8, local_batch_size=2, communication_backend="gloo",
            balance_window_size=window, metadata_mode=metadata_mode, broadcast_plan=broadcast_plan,
            log_balance=log_balance,
            device_prefetch_depth=device_prefetch_depth,
        ),
        dataloader_kwargs={"num_workers": workers, "persistent_workers": workers > 0 and persistent_workers},
        batch_sampler=sampler, cost_model=lambda item: item.cost, device=device, **options,
    )
    return loader


def _ids(batch: tuple[dict, ...]) -> list[int]:
    return [sample["id"] for sample in batch]


class _GatedDataset:
    """Hold the next window's first read until the test releases it."""

    def __init__(self, error: Exception | None = None) -> None:
        """Initialize a controlled future read and optional callback failure."""
        self.started = Event()
        self.release = Event()
        self.reads = []
        self.thread = None
        self.error = error

    def __len__(self) -> int:
        """Return three complete optimizer windows."""
        return 18

    def __getitem__(self, index: int) -> dict:
        """Record a read and gate the first sample in the second window."""
        self.reads.append(index)
        if index == 6:
            self.thread = current_thread()
            self.started.set()
            if not self.release.wait(timeout=10):
                raise RuntimeError("Test did not release the next-window read")
            if self.error is not None:
                raise self.error
        return {"id": index}


class TestAccumulationWindow(unittest.TestCase):
    """Verify joint planning without changing optimizer sample membership."""

    def test_window_stops_at_optimizer_and_epoch_boundaries(self) -> None:
        """Feature: Bounded accumulation windows.
        Description: Plan 22 micro-batches with accumulation ten and a three-round window.
        Expectation: Windows follow 3+3+3+1 per optimizer step, with a final two-round tail.
        """
        for metadata_mode, broadcast_plan, caller_collectives in product((False, True), repeat=3):
            with self.subTest(metadata=metadata_mode, broadcast=broadcast_plan, caller=caller_collectives):
                backend = "hccl" if caller_collectives else "gloo"
                with _loader(metadata_mode=metadata_mode, broadcast_plan=broadcast_plan) as loader, patch.object(
                        DataPlaneTransport, "communication_backend", new_callable=PropertyMock, return_value=backend,
                ):
                    counts = []
                    original = loader._plan_native_batch

                    def record(metadata: list[_ReaderPlanInput], record_count=counts.append, original=original) -> object:
                        """Record source-window size before planning."""
                        record_count(len(metadata[0].original_batches))
                        return original(metadata)

                    with patch.object(loader, "_plan_native_batch", side_effect=record):
                        batches = []
                        for batch in loader:
                            batches.append(_ids(batch))
                            loader.prefetch()
                            loader.prefetch()
                    self.assertEqual(batches, [list(range(start, start + 2)) for start in range(0, 44, 2)])
                    self.assertEqual(counts, [3, 3, 3, 1, 3, 3, 3, 1, 2, 0])

    def test_incomplete_step_replay_with_repeated_prefetch_hooks(self) -> None:
        """Feature: Checkpoint before an incomplete optimizer step.
        Description: Save after one delivery with zero, one or repeated explicit prefetch requests.
        Expectation: Resume repeats the entire incomplete step without storing window plans or samples.
        """
        for metadata_mode, prefetch_calls in product((False, True), (0, 1, 2)):
            with self.subTest(metadata=metadata_mode, prefetch_calls=prefetch_calls), patch.object(
                    DataPlaneTransport, "communication_backend", new_callable=PropertyMock, return_value="hccl",
            ):
                with _loader(metadata_mode=metadata_mode) as loader:
                    self.assertEqual(_ids(next(loader)), [0, 1])
                    for _ in range(prefetch_calls):
                        loader.prefetch()
                    saved = loader.state_dict()
                    owner = "metadata_reader" if metadata_mode else "dataset_reader"
                    self.assertEqual(saved["_index_sampler_state"]["consumed_samples"], 0)
                    self.assertNotIn(owner, saved)
                    self.assertNotIn("window_plans", saved)
                    expected = [_ids(batch) for batch in loader]
                with _loader(metadata_mode=metadata_mode) as restored:
                    restored.load_state_dict(saved)
                    self.assertEqual([_ids(batch) for batch in restored], [[0, 1], *expected])

    def test_window_end_checkpoint_waits_for_complete_optimizer_step(self) -> None:
        """Feature: Complete-step cursor across several balancing windows.
        Description: Inspect progress after the first window and after the full ten-round step.
        Expectation: Finishing a subwindow does not advance the checkpoint boundary.
        """
        with _loader() as loader:
            for _ in range(3):
                next(loader)
            saved = loader.state_dict()
            self.assertEqual(saved["_index_sampler_state"]["consumed_samples"], 0)
            self.assertNotIn("dataset_reader", saved)
            self.assertNotIn("window_plans", saved)
            for _ in range(7):
                next(loader)
            saved = loader.state_dict()
            self.assertEqual(saved["_index_sampler_state"]["consumed_samples"], 20)
        with _loader() as restored:
            restored.load_state_dict(saved)
            self.assertEqual(_ids(next(restored)), [20, 21])

    def test_window_size_configuration(self) -> None:
        """Feature: Window configuration.
        Description: Reject invalid sizes and consume windows larger than accumulation ten.
        Expectation: Invalid values raise; a large window stops at ten rounds.
        """
        for value in (0, -1, True, 1.5, "3"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "balance_window_size"):
                DistributedDatasetConfig(seq_len=8, local_batch_size=2, balance_window_size=value)
        with _loader(window=20) as loader:
            next(loader)
            state = loader.state_dict()
            self.assertEqual(len(loader._current.batches), 9)
            self.assertEqual(state["_index_sampler_state"]["consumed_samples"], 0)
            self.assertNotIn("window_plans", state)

    def test_joint_plan_reduces_sum_of_round_makespans(self) -> None:
        """Feature: Balancing across micro-batches.
        Description: Reassign two rounds with rank costs (9, 1), (9, 1) as (9, 9), (1, 1).
        Expectation: Estimated accumulated makespan drops from 18 to 10 and every occurrence is retained.
        """
        samples = tuple(
            SampleMetadata(1, cost=WorkloadCost(llm=cost), key=SampleKey(index % 2, index, index))
            for index, cost in enumerate((9, 1, 9, 1))
        )
        planner = DynamicPackingPlanner(
            data_parallel_size=2, local_batch_size=1, seq_len=8, cost_model=lambda item: item.cost,
        )
        plans = planner.plan_window(samples, reference_bins=[(item.key,) for item in samples], step=4)
        self.assertEqual([plan.step for plan in plans], [4, 5])
        self.assertEqual([[cost.llm for cost in plan.rank_costs] for plan in plans], [[9, 9], [1, 1]])
        self.assertEqual({key for plan in plans for key in plan.selected_keys}, {item.key for item in samples})
        self.assertEqual(planner.last_balance_decision["original_objective"], 18)
        self.assertEqual(planner.last_balance_decision["accepted_objective"], 10)

    def test_next_window_read_overlaps_all_current_micro_batches(self) -> None:
        """Feature: Two-window source buffering.
        Description: Block the next optimizer window after the current window's first batch is delivered.
        Expectation: Current batches and the last complete checkpoint do not wait for the blocked read.
        """
        for caller_collectives in (False, True):
            with self.subTest(caller=caller_collectives):
                dataset = _GatedDataset()
                loader = _loader(samples=dataset, accumulation=3)
                backend = "hccl" if caller_collectives else "gloo"
                try:
                    with patch.object(
                            DataPlaneTransport, "communication_backend",
                            new_callable=PropertyMock, return_value=backend,
                    ):
                        self.assertEqual(_ids(next(loader)), [0, 1])
                        self.assertTrue(dataset.started.wait(timeout=5))
                        self.assertNotEqual(dataset.thread, current_thread())
                        self.assertEqual(dataset.reads, list(range(7)))
                        state = loader.state_dict()
                        reader = loader._dataset_reader
                        self.assertEqual(reader._sampler.consumed_samples, 12)
                        self.assertEqual(state["_index_sampler_state"]["consumed_samples"], 0)
                        self.assertNotIn("dataset_reader", state)
                        self.assertEqual(_ids(next(loader)), [2, 3])
                        loader.prefetch()
                        self.assertEqual(_ids(next(loader)), [4, 5])
                        self.assertFalse(dataset.release.is_set())
                        dataset.release.set()
                        self.assertEqual(
                            [_ids(batch) for batch in loader], [list(range(i, i + 2)) for i in range(6, 18, 2)],
                        )
                finally:
                    dataset.release.set()
                    loader.close()
                self.assertFalse(dataset.thread.is_alive())
                with _loader(samples=[{"id": i} for i in range(18)], accumulation=3) as restored:
                    restored.load_state_dict(state)
                    self.assertEqual(
                        [_ids(batch) for batch in restored], [list(range(i, i + 2)) for i in range(0, 18, 2)],
                    )

    def test_completed_prefetch_is_bounded_and_excluded_from_checkpoint(self) -> None:
        """Feature: Bounded speculative source progress.
        Description: Finish the next window read before consuming the current window.
        Expectation: No third window is read and saving retains only delivered progress.
        """
        dataset = _GatedDataset()
        dataset.release.set()
        with _loader(samples=dataset, accumulation=3) as loader:
            self.assertEqual(_ids(next(loader)), [0, 1])
            reader = loader._dataset_reader
            loader.wait_for_prefetch()
            self.assertEqual(dataset.reads, list(range(12)))
            self.assertEqual(loader.state_dict()["_index_sampler_state"]["consumed_samples"], 0)
            self.assertNotIn("dataset_reader", loader.state_dict())
            self.assertEqual(_ids(next(loader)), [2, 3])
            self.assertEqual(_ids(next(loader)), [4, 5])
            state = loader.state_dict()
            self.assertEqual(state["_index_sampler_state"]["consumed_samples"], 6)
            self.assertNotIn("dataset_reader", state)
        self.assertFalse(dataset.thread.is_alive())
        with _loader(samples=[{"id": i} for i in range(18)], accumulation=3) as restored:
            restored.load_state_dict(state)
            self.assertEqual(_ids(next(restored)), [6, 7])

    def test_next_window_failure_preserves_current_window(self) -> None:
        """Feature: Speculative source failures.
        Description: Fail the next-window read while the current window is still being delivered.
        Expectation: Current batches finish; the original failure surfaces when the next window is requested.
        """
        for caller_collectives in (False, True):
            with self.subTest(caller=caller_collectives):
                error = ValueError("next window failed")
                dataset = _GatedDataset(error=error)
                dataset.release.set()
                loader = _loader(samples=dataset, accumulation=3)
                backend = "hccl" if caller_collectives else "gloo"
                try:
                    with patch.object(
                            DataPlaneTransport, "communication_backend",
                            new_callable=PropertyMock, return_value=backend,
                    ):
                        self.assertEqual(_ids(next(loader)), [0, 1])
                        with self.assertRaises(ValueError) as failure:
                            loader.wait_for_prefetch()
                        self.assertIs(failure.exception, error)
                        self.assertEqual(_ids(next(loader)), [2, 3])
                        self.assertEqual(_ids(next(loader)), [4, 5])
                        with self.assertRaises(ValueError) as failure:
                            next(loader)
                        self.assertIs(failure.exception, error)
                        self.assertEqual(loader.state_dict()["_index_sampler_state"]["consumed_samples"], 6)
                finally:
                    loader.close()
                self.assertFalse(dataset.thread.is_alive())

    def test_window_prefetch_with_persistent_workers_and_new_epoch(self) -> None:
        """Feature: Persistent Dataset workers with window prefetch.
        Description: Load windows across epochs and restart an incomplete step with one persistent worker.
        Expectation: Worker reads remain serialized and the incomplete step replays in full.
        """
        expected = [list(range(start, start + 2)) for start in range(0, 44, 2)]
        for metadata_mode in (False, True):
            with self.subTest(metadata=metadata_mode):
                with _loader(workers=1, metadata_mode=metadata_mode, accumulation=3) as loader:
                    self.assertEqual(_ids(next(loader)), [0, 1])
                    state = loader.state_dict()
                    self.assertEqual([_ids(batch) for batch in loader], expected[1:])
                    loader.set_epoch(1)
                    self.assertEqual([_ids(batch) for batch in loader], expected)
                with _loader(workers=1, metadata_mode=metadata_mode, accumulation=3) as restored:
                    restored.load_state_dict(state)
                    self.assertEqual([_ids(batch) for batch in restored], expected)

    def test_window_thread_binds_device_before_reading(self) -> None:
        """Feature: Rank-local device context for speculative reads.
        Description: Read three windows with a mocked nonzero CUDA device.
        Expectation: The source thread binds its device once before constructing DataLoader iterators.
        """
        bindings = []
        read_threads = []
        original_iter = DataLoader.__iter__

        def record_device(device: torch.device) -> None:
            """Record which thread selects the accelerator."""
            bindings.append((current_thread(), device))

        def check_device(dataloader: DataLoader) -> object:
            """Check device selection before the future window creates its workers."""
            thread = current_thread()
            read_threads.append(thread)
            self.assertIn((thread, torch.device("cuda:3")), bindings)
            return original_iter(dataloader)

        module = Mock()
        module.stream.side_effect = lambda _: nullcontext()
        module.set_device.side_effect = record_device
        with patch("hyper_parallel.distributed_data.api._create_device_prefetcher", return_value=None), patch.object(
                torch, "cuda", module,
        ), patch.object(DataLoader, "__iter__", check_device), _loader(
                samples=[{"id": i} for i in range(18)], accumulation=3, device="cuda:3",
        ) as loader:
            self.assertEqual([_ids(batch) for batch in loader], [list(range(i, i + 2)) for i in range(0, 18, 2)])
        self.assertEqual(len(read_threads), 3)
        self.assertIs(read_threads[0], read_threads[1])
        self.assertIs(read_threads[1], read_threads[2])
        self.assertEqual(len(bindings), 1)
        self.assertEqual(bindings, [(read_threads[0], torch.device("cuda:3"))])
        self.assertIn((read_threads[1], torch.device("cuda:3")), bindings)
        self.assertFalse(read_threads[1].is_alive())

    def test_window_thread_device_failure_reaches_consumer(self) -> None:
        """Feature: Source thread initialization failures.
        Description: Fail device selection before any source read begins.
        Expectation: The first delivery raises the original error without hanging or advancing progress.
        """
        error = RuntimeError("source device binding failed")
        module = Mock()
        module.set_device.side_effect = error
        with patch("hyper_parallel.distributed_data.api._create_device_prefetcher", return_value=None), patch.object(
                torch, "cuda", module,
        ), _loader(samples=[{"id": i} for i in range(18)], accumulation=3, device="cuda:3") as loader:
            with self.assertRaises(RuntimeError) as failure:
                next(loader)
            self.assertIs(failure.exception, error)
            self.assertEqual(loader.state_dict()["_index_sampler_state"]["consumed_samples"], 0)

    def test_nonpersistent_workers_exit_normally_after_future_reads(self) -> None:
        """Feature: Forked workers in the window reader.
        Description: Recreate Dataset workers for every source window on the background thread.
        Expectation: All samples are delivered and every worker exits with code zero.
        """
        workers = []
        original_iter = DataLoader.__iter__

        def record_workers(dataloader: DataLoader) -> object:
            """Retain process handles to inspect their exit codes after reading."""
            iterator = original_iter(dataloader)
            workers.extend(iterator._workers)
            return iterator

        with patch.object(DataLoader, "__iter__", record_workers), _loader(
                samples=[{"id": i} for i in range(18)], accumulation=3, workers=1, persistent_workers=False,
        ) as loader:
            self.assertEqual([_ids(batch) for batch in loader], [list(range(i, i + 2)) for i in range(0, 18, 2)])
        self.assertEqual(len(workers), 3)
        self.assertEqual([worker.exitcode for worker in workers], [0, 0, 0])
