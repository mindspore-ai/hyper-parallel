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
"""Window preparation, bounded lookahead, and optimizer-boundary checkpoint progress."""

import copy
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from itertools import product
from threading import Event, current_thread
from unittest.mock import Mock, patch

import torch

from tests.ut.distributed_data.test_accumulation_window import _GatedDataset, _ids, _loader
from tests.ut.distributed_data.test_batch_sampler import _TrackedDataset, _loader as _native_loader, _sampler


class TestOptimizerStepPrefetch(unittest.TestCase):
    """Check bounded input buffering and complete-step checkpoint progress."""

    def test_prefetch_is_bounded_by_current_and_next_window(self) -> None:
        """Feature: Window-sized double buffering.
        Description: Prefetch with accumulation smaller, larger or not divisible by the configured window.
        Expectation: Reads and staging stop after two windows; checkpoints still advance only at full steps.
        """
        for metadata_mode, sizes in product((False, True), ((1, 3, 1, 1), (5, 3, 3, 2),
                                                         (10, 3, 3, 3), (10, 1, 1, 1))):
            accumulation, window, first_size, next_size = sizes
            with self.subTest(metadata=metadata_mode, accumulation=accumulation, window=window):
                dataset = _TrackedDataset(size=4 * accumulation + 2)
                with _loader(samples=dataset, accumulation=accumulation, window=window,
                             metadata_mode=metadata_mode) as loader, patch.object(
                        loader, "_stage_batch", wraps=loader._stage_batch,
                ) as stage, patch.object(
                        loader._planning_reader(), "prepare_next_step",
                        wraps=loader._planning_reader().prepare_next_step,
                ) as prepare:
                    delivered = [_ids(next(loader))]
                    loader.wait_for_prefetch()
                    prepared = first_size + next_size
                    self.assertEqual([call.args for call in prepare.call_args_list], [(first_size,), (next_size,)])
                    self.assertEqual(dataset.reads, list(range(2 * prepared)))
                    if not metadata_mode:
                        self.assertIsNone(loader._dataset_reader._window)
                    self.assertEqual(stage.call_count, prepared)
                    self.assertEqual(len(loader._current.batches), first_size - 1)
                    self.assertEqual(len(loader._future.result().batches), next_size)
                    for _ in range(3):
                        loader.prefetch()
                    loader.wait_for_prefetch()
                    self.assertEqual(dataset.reads, list(range(2 * prepared)))
                    self.assertEqual(stage.call_count, prepared)
                    for batch in loader:
                        delivered.append(_ids(batch))
                        boundary = len(delivered) // accumulation * accumulation
                        self.assertEqual(loader.state_dict()["_index_sampler_state"]["consumed_samples"], 2 * boundary)
                    self.assertEqual(delivered, [list(range(i, i + 2)) for i in range(0, len(dataset), 2)])

    def test_restart_without_partial_gradients_replays_optimizer_update(self) -> None:
        """Feature: Training recovery at optimizer boundaries.
        Description: Interrupt the second step after one backward and restore weights, moments, RNG and data.
        Expectation: Replaying all three micro-batches reproduces the uninterrupted parameter update exactly.
        """
        weight = torch.nn.Parameter(torch.tensor(0.5))
        # CPU replay must not probe NPU kernels when torch_npu patches Adam.
        optimizer = torch.optim.Adam([weight], lr=0.01, foreach=False)
        generator = torch.Generator().manual_seed(17)

        def backward(batch: tuple[dict, ...], parameter: torch.Tensor) -> None:
            """Accumulate one stochastic micro-batch into the current optimizer step."""
            inputs = torch.tensor(_ids(batch), dtype=torch.float32) / 20
            inputs = inputs * torch.rand(inputs.shape, generator=generator)
            ((inputs * parameter - 1).square().mean() / 3).backward()

        with _loader(accumulation=3) as loader:
            for _ in range(3):
                backward(next(loader), weight)
            optimizer.step()
            optimizer.zero_grad()
            saved_weight = torch.detach(weight).clone()
            saved_optimizer = copy.deepcopy(optimizer.state_dict())
            saved_rng = generator.get_state()
            boundary = loader.state_dict()
            backward(next(loader), weight)
            self.assertIsNotNone(weight.grad)
            data_state = loader.state_dict()
            self.assertEqual(data_state, boundary)
            for _ in range(2):
                backward(next(loader), weight)
            optimizer.step()
            expected_weight = torch.detach(weight).clone()
            expected_moments = copy.deepcopy(optimizer.state_dict()["state"])

        restored_weight = torch.nn.Parameter(saved_weight)
        restored_optimizer = torch.optim.Adam([restored_weight], lr=0.01, foreach=False)
        restored_optimizer.load_state_dict(saved_optimizer)
        generator.set_state(saved_rng)
        self.assertIsNone(restored_weight.grad)
        with _loader(accumulation=3) as restored:
            restored.load_state_dict(data_state)
            ids = []
            for _ in range(3):
                batch = next(restored)
                ids.extend(_ids(batch))
                backward(batch, restored_weight)
            restored_optimizer.step()
            self.assertEqual(ids, list(range(6, 12)))
            torch.testing.assert_close(restored_weight, expected_weight, rtol=0, atol=0)
            for key, value in expected_moments[0].items():
                torch.testing.assert_close(restored_optimizer.state_dict()["state"][0][key], value, rtol=0, atol=0)

    def test_short_tail_does_not_advance_optimizer_checkpoint(self) -> None:
        """Feature: Incomplete epoch tails.
        Description: Deliver a full three-round step followed by a two-round tail.
        Expectation: The tail replays before EOF; after EOF the exhausted state restores and resets correctly.
        """
        options = {"samples": [{"id": i} for i in range(10)], "accumulation": 3}
        with _loader(**options) as loader:
            next(loader)
            with self.assertRaisesRegex(ValueError, "fresh or exhausted"):
                loader.set_epoch(1)
            for _ in range(4):
                next(loader)
            state = loader.state_dict()
            self.assertEqual(state["_index_sampler_state"]["consumed_samples"], 6)
            with self.assertRaises(StopIteration):
                next(loader)
            exhausted = loader.state_dict()
        with _loader(**options) as restored:
            restored.load_state_dict(state)
            self.assertEqual([_ids(batch) for batch in restored], [[6, 7], [8, 9]])
        with _loader(**options) as restored:
            restored.load_state_dict(exhausted)
            self.assertEqual(list(restored), [])
            restored.set_epoch(1)
            self.assertEqual([_ids(batch) for batch in restored], [list(range(i, i + 2)) for i in range(0, 10, 2)])

    def test_full_step_is_ready_and_only_one_successor_is_prefetched(self) -> None:
        """Feature: Complete-step double buffering.
        Description: Finish source reads, packing and staging for two steps before consuming the first.
        Expectation: Only twelve samples are read and checkpoint progress waits for all three deliveries.
        """
        dataset = _GatedDataset()
        dataset.release.set()
        staged = []
        with _loader(samples=dataset, accumulation=3) as loader:
            original = loader._stage_batch

            def stage(batch: object) -> object:
                """Record completed per-micro-batch staging on the producer."""
                staged.append((_ids(batch), current_thread()))
                return original(batch)

            with patch.object(loader, "_stage_batch", side_effect=stage):
                self.assertEqual(_ids(next(loader)), [0, 1])
                loader.wait_for_prefetch()
                self.assertEqual(dataset.reads, list(range(12)))
                self.assertEqual([ids for ids, _ in staged], [list(range(i, i + 2)) for i in range(0, 12, 2)])
                self.assertEqual({thread for _, thread in staged}, {loader._prefetch_thread})
                self.assertIsNot(loader._prefetch_thread, current_thread())
                state = loader.state_dict()
                self.assertEqual(state["_index_sampler_state"]["consumed_samples"], 0)
                self.assertNotIn("dataset_reader", state)
                self.assertNotIn("optimizer_step_remaining", state)
                for _ in range(3):
                    loader.prefetch()
                self.assertEqual(dataset.reads, list(range(12)))
                self.assertEqual(_ids(next(loader)), [2, 3])
                self.assertEqual(_ids(next(loader)), [4, 5])
                self.assertEqual(loader.state_dict()["_index_sampler_state"]["consumed_samples"], 6)
                self.assertEqual(dataset.reads, list(range(12)))

    def test_slow_next_window_does_not_block_current_delivery_or_checkpoint(self) -> None:
        """Feature: Independent current and future windows.
        Description: Block the next window in reading, planning, exchange or staging, including inside a step.
        Expectation: Current batches and their checkpoints remain available before the future is released.
        """
        for stage, accumulation in product(("read", "plan", "exchange", "stage"), (3, 10)):
            with self.subTest(stage=stage, accumulation=accumulation):
                self._check_blocked_stage(stage, accumulation)

    def _check_blocked_stage(self, stage: str, accumulation: int) -> None:
        """Hold one future preparation stage and verify current-step delivery and replay."""
        dataset = _GatedDataset()
        entered, release = Event(), Event()
        loader = _loader(samples=dataset, accumulation=accumulation)
        source = loader
        owner, method = {
            "read": (source, "_prepare_local_metadata"), "plan": (source._planner, "plan_window"),
            "exchange": (source._data_plane, "exchange_prepared"), "stage": (source, "_stage_batch"),
        }[stage]
        original = getattr(owner, method)
        dataset.release.set()

        def gated(*args: object, **kwargs: object) -> object:
            """Hold the first operation after the current window has been prepared."""
            if source._step >= 3:
                entered.set()
                if not release.wait(timeout=10):
                    raise RuntimeError("Test did not release future preparation")
            return original(*args, **kwargs)

        try:
            with patch.object(owner, method, side_effect=gated):
                self.assertEqual(_ids(next(loader)), [0, 1])
                self.assertTrue(entered.wait(timeout=5))
                state = loader.state_dict()
                self.assertEqual(_ids(next(loader)), [2, 3])
                self.assertEqual(_ids(next(loader)), [4, 5])
                self.assertFalse(release.is_set())
                release.set()
                expected = [_ids(batch) for batch in loader]
        finally:
            release.set()
            loader.close()
        with _loader(samples=[{"id": i} for i in range(18)], accumulation=accumulation) as resumed:
            resumed.load_state_dict(state)
            self.assertEqual([_ids(batch) for batch in resumed], [[0, 1], [2, 3], [4, 5], *expected])

    def test_resume_replays_incomplete_steps_across_subwindows(self) -> None:
        """Feature: Checkpointing window speculation at optimizer boundaries.
        Description: Resume online/offline loaders inside and between differently sized balancing windows.
        Expectation: Complete steps stay consumed and incomplete steps replay from their beginning.
        """
        for metadata, accumulation, window in product((False, True), (1, 3, 5, 10), (1, 3)):
            with self.subTest(metadata=metadata, accumulation=accumulation, window=window):
                options = {"metadata_mode": metadata, "accumulation": accumulation,
                           "window": window}
                states = []
                with _loader(**options) as loader:
                    delivered = []
                    for batch in loader:
                        delivered.append(_ids(batch))
                        if len(delivered) in (1, 3, accumulation, accumulation + 1):
                            states.append((len(delivered), loader.state_dict()))
                    loader.set_epoch(1)
                    self.assertEqual([_ids(batch) for batch in loader], delivered)
                for count, state in states:
                    with _loader(**options) as resumed:
                        resumed.load_state_dict(state)
                        boundary = count // accumulation * accumulation
                        self.assertEqual(state["_index_sampler_state"]["consumed_samples"], 2 * boundary)
                        self.assertEqual([_ids(batch) for batch in resumed], delivered[boundary:])

    def test_window_prefetch_preserves_three_three_three_one_planning(self) -> None:
        """Feature: Independent optimizer and balancing windows.
        Description: Buffer at most three rounds at a time while accumulating ten rounds per optimizer step.
        Expectation: Planner windows remain 3+3+3+1 and the epoch tail stays within its optimizer step.
        """
        with _loader(accumulation=10, window=3) as loader:
            counts = []
            original = loader._plan_native_batch

            def record(metadata: list) -> object:
                """Record native rounds presented to each planning invocation."""
                counts.append(len(metadata[0].original_batches))
                return original(metadata)

            with patch.object(loader, "_plan_native_batch", side_effect=record):
                batches = [_ids(batch) for batch in loader]
            self.assertEqual(batches, [list(range(i, i + 2)) for i in range(0, 44, 2)])
            self.assertEqual(counts, [3, 3, 3, 1, 3, 3, 3, 1, 2, 0])

    def test_future_failure_is_deferred_until_current_step_finishes(self) -> None:
        """Feature: Speculative error delivery.
        Description: Fail the next optimizer step while three current batches are ready.
        Expectation: Current batches finish and the original error appears at the next step boundary.
        """
        error = ValueError("future step failed")
        dataset = _GatedDataset(error=error)
        dataset.release.set()
        with _loader(samples=dataset, accumulation=3) as loader:
            self.assertEqual(_ids(next(loader)), [0, 1])
            with self.assertRaises(ValueError) as caught:
                loader.wait_for_prefetch()
            self.assertIs(caught.exception, error)
            self.assertEqual(_ids(next(loader)), [2, 3])
            self.assertEqual(_ids(next(loader)), [4, 5])
            state = loader.state_dict()
            self.assertEqual(state["_index_sampler_state"]["consumed_samples"], 6)
            with self.assertRaises(ValueError) as caught:
                next(loader)
            self.assertIs(caught.exception, error)

    def test_host_device_binding_and_training_stream_delivery(self) -> None:
        """Feature: Accelerator micro-batch ownership.
        Description: Prepare CPU-shaped test inputs under a mocked CUDA stream and event API.
        Expectation: Delivery records the caller stream and only explicit drains wait for its marker.
        """
        module = Mock()
        module.stream.side_effect = lambda _: nullcontext()
        event = Mock()
        module.Event.return_value = event
        with patch.object(torch, "cuda", module), patch(
                "hyper_parallel.distributed_data.api._create_device_prefetcher", return_value=None,
        ), patch("hyper_parallel.distributed_data.distributed_dataloader._record_stream") as record, _loader(
                accumulation=10, samples=[{"id": i} for i in range(12)], device="cuda:3",
        ) as loader:
            loader.prefetch()
            loader._future.result(timeout=5)
            loader.prefetch()
            event.synchronize.assert_not_called()
            self.assertEqual(_ids(next(loader)), [0, 1])
            self.assertEqual(len(loader._device_batches), 1)
            self.assertEqual(module.set_device.call_count, 1)
            module.set_device.assert_called_with(torch.device("cuda:3"))
            event.record.assert_called_once_with(module.current_stream.return_value)
            event.synchronize.assert_not_called()
            module.current_stream.return_value.wait_event.assert_not_called()
            record.assert_called_once_with(({"id": 0}, {"id": 1}), module.current_stream.return_value)
            loader.wait_for_prefetch()
            event.synchronize.assert_called_once_with()
            module.Stream.return_value.synchronize.assert_not_called()
            module.synchronize.assert_not_called()
        self.assertEqual(module.Stream.return_value.synchronize.call_count, 1)

    def test_epoch_reset_reuses_producer_and_training_stream(self) -> None:
        """Feature: Loader-owned producer lifetime.
        Description: Consume full, partial and dropped epochs with online or offline metadata.
        Expectation: Epoch changes reuse the Host producer; delivery keeps using the caller stream.
        """
        for metadata_mode, accumulation, size in product((False, True), (1, 3), (1, 10, 12)):
            with self.subTest(metadata=metadata_mode, accumulation=accumulation, size=size):
                module = Mock()
                module.stream.side_effect = lambda _: nullcontext()
                with patch.object(torch, "cuda", module), patch(
                        "hyper_parallel.distributed_data.api._create_device_prefetcher", return_value=None,
                ), patch("hyper_parallel.distributed_data.distributed_dataloader._record_stream"), _loader(
                        metadata_mode=metadata_mode, accumulation=accumulation,
                        samples=[{"id": i} for i in range(size)], device="cuda:3",
                ) as loader:
                    expected = [list(range(i, i + 2)) for i in range(0, size - 1, 2)]
                    self.assertEqual([_ids(batch) for batch in loader], expected)
                    worker = loader._prefetch_thread
                    for epoch in (1, 2):
                        self.assertTrue(worker.is_alive())
                        self.assertFalse(loader._closed)
                        loader.prefetch()
                        self.assertIsNone(loader._future)
                        self.assertIsNone(next(loader, None))
                        loader.set_epoch(epoch)
                        self.assertEqual(loader.state_dict()["_index_sampler_state"]["epoch"], epoch)
                        self.assertEqual(loader.state_dict()["_index_sampler_state"]["consumed_samples"], 0)
                        self.assertEqual([_ids(batch) for batch in loader], expected)
                        self.assertIs(loader._prefetch_thread, worker)
                        module.Event.return_value.record.assert_called_with(module.current_stream.return_value)
                    self.assertEqual(module.Stream.call_count, 1)
                    module.Stream.assert_called_with(device=torch.device("cuda:3"))
                    self.assertEqual(module.set_device.call_count, 1)
                    module.set_device.assert_called_with(torch.device("cuda:3"))
                    module.Stream.return_value.synchronize.assert_not_called()
                self.assertFalse(worker.is_alive())
                self.assertEqual(module.Stream.return_value.synchronize.call_count, 1)
                self.assertIsNone(next(loader, None))
                with self.assertRaisesRegex(ValueError, "closed loader"):
                    loader.set_epoch(3)

    def test_epoch_reset_waits_for_undelivered_prefetch(self) -> None:
        """Feature: Safe epoch changes before first delivery.
        Description: Reset while the initial speculative step is blocked on the producer.
        Expectation: Reset waits for production, discards old inputs and reuses the thread for the new epoch.
        """
        entered, release, resetting = Event(), Event(), Event()
        with _loader(accumulation=3) as loader:
            prepare, wait = loader._prepare_window, loader.wait_for_prefetch

            def blocked() -> object:
                """Hold production until the reset has begun waiting."""
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("prefetch was not released")
                return prepare()

            def drain() -> None:
                """Observe the reset boundary before draining the queued work."""
                resetting.set()
                wait()

            with patch.object(loader, "_prepare_window", side_effect=blocked), patch.object(
                    loader, "wait_for_prefetch", side_effect=drain,
            ), ThreadPoolExecutor(max_workers=1) as executor:
                loader.prefetch()
                worker = loader._prefetch_thread
                try:
                    self.assertTrue(entered.wait(5))
                    reset = executor.submit(loader.set_epoch, 1)
                    self.assertTrue(resetting.wait(5))
                    self.assertFalse(reset.done())
                    self.assertEqual(loader._dataset_reader._sampler.epoch, 0)
                finally:
                    release.set()
                reset.result(timeout=5)
            self.assertIsNone(loader._future)
            self.assertEqual(loader.state_dict()["_index_sampler_state"]["epoch"], 1)
            self.assertEqual([_ids(batch) for batch in loader], [list(range(i, i + 2)) for i in range(0, 44, 2)])
            self.assertIs(loader._prefetch_thread, worker)

    def test_explicit_prefetch_wait_drains_current_tail_without_a_future(self) -> None:
        """Feature: Explicit device drain at an incomplete epoch tail.
        Description: Deliver a short step that has no successor, then explicitly wait for prefetch.
        Expectation: Delivery does not CPU-wait; the explicit wait drains the tail's event.
        """
        module = Mock()
        module.stream.side_effect = lambda _: nullcontext()
        with patch.object(torch, "cuda", module), patch(
                "hyper_parallel.distributed_data.api._create_device_prefetcher", return_value=None,
        ), patch("hyper_parallel.distributed_data.distributed_dataloader._record_stream"), _loader(
                accumulation=3, samples=[{"id": i} for i in range(2)], device="cuda:3",
        ) as loader:
            self.assertEqual(_ids(next(loader)), [0, 1])
            self.assertIsNone(loader._future)
            self.assertTrue(loader._current.exhausted)
            event = module.Event.return_value
            event.synchronize.assert_not_called()
            module.current_stream.return_value.wait_event.assert_not_called()
            loader.wait_for_prefetch()
            event.synchronize.assert_called_once_with()
            self.assertEqual(loader.state_dict()["_index_sampler_state"]["consumed_samples"], 0)
        self.assertEqual(module.Stream.return_value.synchronize.call_count, 1)

    def test_device_initialization_failure_reaches_consumer(self) -> None:
        """Feature: Failed producer initialization.
        Description: Fail device binding before the background thread reads any source data.
        Expectation: next() raises the original error and close() drains without hanging.
        """
        error = RuntimeError("device binding failed")
        module = Mock()
        module.set_device.side_effect = error
        with patch("hyper_parallel.distributed_data.api._create_device_prefetcher", return_value=None), patch.object(
                torch, "cuda", module,
        ), _loader(accumulation=3, device="cuda:3") as loader:
            with self.assertRaises(RuntimeError) as caught:
                next(loader)
            self.assertIs(caught.exception, error)
            self.assertEqual(loader.state_dict()["_index_sampler_state"]["consumed_samples"], 0)

    def test_initial_sampler_offset_requires_optimizer_boundary(self) -> None:
        """Feature: Complete-step initial cursors.
        Description: Start one round into an optimizer step without saved gradients.
        Expectation: Construction rejects the partial-step cursor.
        """
        dataset = _TrackedDataset(size=18)
        sampler = _sampler(total_samples=18, global_batch_size=6, consumed_samples=2)
        with self.assertRaisesRegex(ValueError, "complete optimizer step"):
            _native_loader(dataset, sampler=sampler, balance_window_size=3)

    def test_same_epoch_after_prefetch_preserves_undelivered_inputs(self) -> None:
        """Feature: Idempotent initial epoch selection.
        Description: Prefetch a full step before setting the current epoch again.
        Expectation: The prepared step and delivered cursor remain unchanged.
        """
        dataset = _TrackedDataset(size=18)
        sampler = _sampler(total_samples=18, global_batch_size=6, consumed_samples=6)
        with _native_loader(dataset, sampler=sampler, balance_window_size=3) as loader:
            loader.prefetch()
            loader.wait_for_prefetch()
            state = loader.state_dict()
            future = loader._future
            loader.set_epoch(0)
            self.assertEqual(loader.state_dict(), state)
            self.assertIs(loader._future, future)
            self.assertEqual(dataset.reads, list(range(6, 12)))
            self.assertEqual([_ids(batch) for batch in loader], [list(range(i, i + 2)) for i in range(6, 18, 2)])

    def test_legacy_checkpoint_requires_complete_optimizer_step(self) -> None:
        """Feature: Legacy checkpoint boundaries.
        Description: Restore complete-step checkpoints and reject every partial-step representation.
        Expectation: Valid cursors replay exactly and partial cursors raise without skipping data.
        """
        for metadata_mode, window in product((False, True), (1, 3)):
            with self.subTest(metadata_mode=metadata_mode, window=window):
                dataset = _TrackedDataset(size=18)
                sampler = _sampler(total_samples=18, global_batch_size=6)
                options = {"metadata_mode": metadata_mode, "balance_window_size": window}
                with _native_loader(dataset, sampler=sampler, **options) as loader:
                    for _ in range(3):
                        next(loader)
                    state = loader.state_dict()
                    state["optimizer_step_remaining"] = 3
                    expected = [_ids(batch) for batch in loader]
                sampler = _sampler(total_samples=18, global_batch_size=6)
                with _native_loader(dataset, sampler=sampler, **options) as resumed:
                    resumed.load_state_dict(state)
                    self.assertNotIn("optimizer_step_remaining", resumed.state_dict())
                    self.assertEqual([_ids(batch) for batch in resumed], expected)
                for field in ("step", "remaining", "plans", "window", "cursor"):
                    invalid = copy.deepcopy(state)
                    owner = "metadata_reader" if metadata_mode else "dataset_reader"
                    if field == "step":
                        invalid["step"] = 1
                    elif field == "remaining":
                        invalid["optimizer_step_remaining"] = 2
                    elif field == "plans":
                        invalid["window_plans"] = (None,)
                    elif field == "window":
                        invalid[owner] = {"window": {"rounds": 3, "delivered_rounds": 1}}
                    else:
                        invalid["_index_sampler_state"]["consumed_samples"] += 2
                    with self.subTest(field=field), _native_loader(
                            dataset, sampler=_sampler(total_samples=18, global_batch_size=6), **options,
                    ) as resumed, self.assertRaisesRegex(ValueError, "complete optimizer step"):
                        resumed.load_state_dict(invalid)

    def test_workers_and_epoch_reset_preserve_step_replay(self) -> None:
        """Feature: Reusable source workers on the step producer.
        Description: Read two epochs and restart an incomplete step with persistent or transient workers.
        Expectation: The incomplete step replays in full while the producer survives epoch changes.
        """
        for persistent in (False, True):
            with self.subTest(persistent=persistent):
                options = {"samples": [{"id": i} for i in range(18)], "accumulation": 3,
                           "workers": 1, "persistent_workers": persistent}
                with _loader(**options) as loader:
                    self.assertEqual(_ids(next(loader)), [0, 1])
                    state = loader.state_dict()
                    expected = [_ids(batch) for batch in loader]
                    worker = loader._prefetch_thread
                    self.assertTrue(worker.is_alive())
                    loader.set_epoch(1)
                    self.assertEqual([_ids(batch) for batch in loader], [[0, 1], *expected])
                    self.assertIs(loader._prefetch_thread, worker)
                self.assertFalse(worker.is_alive())
                with _loader(**options) as resumed:
                    resumed.load_state_dict(state)
                    self.assertEqual([_ids(batch) for batch in resumed], [[0, 1], *expected])
