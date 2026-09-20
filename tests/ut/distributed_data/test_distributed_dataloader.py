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
"""Regression tests for source-selected steps, direct reads, and prefetch."""

import copy
import unittest
from contextlib import nullcontext
from collections.abc import Callable
from threading import current_thread
from typing import Any
from unittest.mock import Mock, PropertyMock, patch

import torch

from hyper_parallel.auto_models.components.datasets.parallel import build_dataset_batch_sampler
from hyper_parallel.distributed_data import DistributedDatasetConfig, SampleMetadata, build_distributed_dataloader
from hyper_parallel.distributed_data.distributed_dataloader import _ReaderSnapshot
from hyper_parallel.distributed_data.planner import DynamicPackingPlanner
from hyper_parallel.distributed_data.schema import BufferedSampleMetadata, DistributedPackingPlan, SampleKey
from hyper_parallel.distributed_data.transport import DataPlaneTransport
from tests.common.mark_utils import arg_mark


class _StandaloneMesh:
    mesh_shape = (1,)
    mesh_dim_names = ("dp",)
    rank_list = (0,)


class _StepReader:
    """Model a user-owned producer with explicitly selected pack boundaries."""

    def __init__(self, steps: list[list[list[dict[str, int]]]]) -> None:
        """Store externally selected step boundaries for deterministic replay."""
        self.steps = steps
        self.epoch = 0
        self.position = 0
        self.exhausted = False
        self.reference_bins = ()
        self.prepare_calls = 0
        self.payloads = {}

    @property
    def batch_position(self) -> int:
        """Return the committed local-step cursor."""
        return self.position

    def prepare_next_step(self) -> None:
        """Expose a whole pending step without consuming a future one."""
        self.prepare_calls += 1
        if self.reference_bins or self.exhausted:
            return
        if self.position == len(self.steps):
            self.exhausted = True
            return
        bins = []
        for samples in self.steps[self.position]:
            items = []
            for sample in samples:
                key = SampleKey(0, sample["id"], len(self.payloads))
                self.payloads[key] = sample
                items.append(BufferedSampleMetadata(
                    key, SampleMetadata(sample["tokens"], sample_id=sample["id"]), len(self.payloads) - 1,
                ))
            bins.append(tuple(items))
        self.reference_bins = tuple(bins)

    def metadata(self) -> tuple[BufferedSampleMetadata, ...]:
        """Return only the current step's metadata."""
        return tuple(item for packing_bin in self.reference_bins for item in packing_bin)

    def selected_payloads(self, keys: set[SampleKey]) -> tuple[tuple[SampleKey, Any], ...]:
        """Route buffered payloads without reading by Dataset index."""
        return tuple((key, self.payloads[key]) for key in keys)

    def commit(self, keys: set[SampleKey]) -> None:
        """Advance only after the entire selected step was consumed."""
        if keys != set(self.payloads):
            raise ValueError("Step commit must preserve every selected occurrence.")
        self.position += 1
        self.reference_bins = ()
        self.payloads = {}

    def state_dict(self) -> dict[str, int]:
        """Exclude speculative fills from the saved cursor."""
        return {"epoch": self.epoch, "position": self.position}

    def load_state_dict(self, state: dict[str, int]) -> None:
        """Restore the committed cursor and discard pending data."""
        self.set_epoch(state["epoch"])
        self.position = state["position"]

    def set_epoch(self, epoch: int) -> None:
        """Restart the deterministic source."""
        self.epoch = epoch
        self.position = 0
        self.exhausted = False
        self.reference_bins = ()
        self.payloads = {}


def _steps():
    return [
        [[{"id": 0, "tokens": 6}, {"id": 1, "tokens": 4}]],
        [[{"id": 2, "tokens": 6}, {"id": 3, "tokens": 4}]],
        [[{"id": 4, "tokens": 6}, {"id": 5, "tokens": 4}]],
    ]


def _external_loader(reader=None, **options):
    reader = _StepReader(_steps()) if reader is None else reader
    loader = build_distributed_dataloader(
        None, _StandaloneMesh(), DistributedDatasetConfig(seq_len=10, local_batch_size=1, **options),
        external_step_reader=reader, device="cpu", cost_model=lambda metadata: metadata.cost,
    )
    return loader


def _sampler(size=6, local_batch_size=2):
    return build_dataset_batch_sampler(
        total_samples=size, micro_batch_size=local_batch_size, global_batch_size=local_batch_size,
        dp_world_size=1, dp_rank=0,
    )


class TestDistributedDataLoaderEndToEnd(unittest.TestCase):
    """Verify that each plan consumes one externally determined step."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_accelerator_collectives_stay_on_caller(self) -> None:
        """Feature: Ordered accelerator collective launches.
        Description: Consume steps with implicit or repeated explicit prefetch hooks.
        Expectation: Collectives stay on the caller; CPU work reuses one background thread.
        """
        for explicit in (False, True):
            with self.subTest(explicit=explicit):
                loader = _external_loader()
                events = []

                def record(name: str, callback: Callable[..., Any]) -> Callable[..., Any]:
                    """Record thread ownership without changing callback semantics."""
                    def run(*args: Any, **kwargs: Any) -> Any:
                        """Execute one instrumented pipeline stage."""
                        events.append((name, current_thread().name))
                        return callback(*args, **kwargs)
                    return run

                with patch.object(DataPlaneTransport, "communication_backend", new_callable=PropertyMock,
                                  return_value="hccl"), patch.object(
                        loader, "_fill_local_reader", side_effect=record("read", loader._fill_local_reader),
                ), patch.object(
                        loader, "_build_plan_control", side_effect=record("plan", loader._build_plan_control),
                ), patch.object(
                        loader._data_plane, "gather_object_to_planner",
                        side_effect=record("gather", loader._data_plane.gather_object_to_planner),
                ), patch.object(
                        loader._data_plane, "broadcast_from_planner",
                        side_effect=record("broadcast", loader._data_plane.broadcast_from_planner),
                ), patch.object(
                        loader._data_plane, "begin_exchange_prepared",
                        side_effect=record("exchange", loader._data_plane.begin_exchange_prepared),
                ):
                    self.assertEqual(next(loader)[0][0]["id"], 0)
                    worker = loader._prefetch_thread
                    if explicit:
                        loader.prefetch_plan()
                        loader.prefetch_plan()
                        loader.prefetch()
                        loader.prefetch()
                    state = loader.state_dict()
                    self.assertEqual(state["step"], 1)
                    self.assertEqual(next(loader)[0][0]["id"], 2)
                    self.assertIs(worker, loader._prefetch_thread)
                    self.assertEqual(len(list(loader)), 1)
                for name, thread in events:
                    expected = "MainThread" if name in ("gather", "broadcast", "exchange") else worker.name
                    self.assertEqual(thread, expected)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_sampler_paths_stage_h2d_before_model_broadcast(self) -> None:
        """Feature: Shared sampler H2D staging.
        Description: Consume and checkpoint online and metadata sampler batches.
        Expectation: Broadcast waits for device copies without committing speculative reads.
        """
        for metadata_mode in (False, True):
            with self.subTest(metadata_mode=metadata_mode):
                accelerator = Mock()
                accelerator.stream.side_effect = lambda _: nullcontext()
                accelerator.Event.return_value.query.return_value = False
                copies = []

                def move(batch: Any, device: torch.device) -> dict[str, Any]:
                    """Record an asynchronous copy without requiring accelerator hardware."""
                    copies.append((batch, device, current_thread().name))
                    return {"data": batch, "storage": Mock()}

                options = {"metadata": [SampleMetadata(1)] * 6} if metadata_mode else {
                    "metadata_fn": lambda _: SampleMetadata(1),
                }
                with patch.object(torch, "cuda", accelerator), patch(
                        "hyper_parallel.distributed_data.device_prefetch._pin_memory", side_effect=lambda value: value,
                ):
                    with build_distributed_dataloader(
                            list(range(6)), _StandaloneMesh(),
                            DistributedDatasetConfig(seq_len=10, local_batch_size=2, communication_backend="gloo"),
                            batch_sampler=_sampler(), device="cuda:0", move_fn=move,
                            cost_model=lambda metadata: metadata.cost, **options,
                    ) as loader:
                        broadcast = Mock(side_effect=lambda value: value)
                        loader._model_transport = Mock(broadcast=broadcast)
                        first = next(loader)
                        self.assertEqual(first["data"], (0, 1))
                        self.assertIs(broadcast.call_args.args[0], first)
                        accelerator.current_stream.return_value.wait_event.assert_called_once()
                        first["storage"].record_stream.assert_called_once()
                        loader.wait_for_prefetch()
                        self.assertEqual(len(copies), 2)
                        self.assertTrue(all(thread != "MainThread" for _, _, thread in copies))
                        state = loader.state_dict()
                        self.assertEqual(state["step"], 1)
                        self.assertEqual(len(copies), 2)
                        self.assertEqual(next(loader)["data"], (2, 3))

    def test_plan_uses_reader_snapshots_in_collective_order(self) -> None:
        """Non-readers are filtered without changing source-selected membership."""
        loader = _external_loader()
        first = BufferedSampleMetadata(SampleKey(0, 8), SampleMetadata(4), 7)
        second = BufferedSampleMetadata(SampleKey(2, 3), SampleMetadata(6), 42)
        snapshots = (
            _ReaderSnapshot(0, False, (first,), ((first,),)),
            _ReaderSnapshot(1, True, ()),
            _ReaderSnapshot(2, False, (second,), ((second,),)),
        )
        planner = DynamicPackingPlanner(
            data_parallel_size=2, seq_len=10, local_batch_size=1, cost_model=lambda metadata: metadata.cost,
        )
        with (
                patch.object(loader, "_dataset_reader_ranks", frozenset((0, 2))),
                patch.object(loader, "_planner", planner),
                patch.object(planner, "plan", wraps=planner.plan) as plan_step,
        ):
            plan = loader._build_plan_control(snapshots)

        samples = plan_step.call_args.args[0]
        self.assertEqual(tuple(item.key for item in samples), (first.key, second.key))
        self.assertEqual(tuple(item.global_sample_position for item in samples), (0, 1))
        self.assertEqual(plan_step.call_args.kwargs["reference_bins"], ((first.key,), (second.key,)))
        self.assertEqual(set(plan.selected_keys), {first.key, second.key})

    def test_readers_must_exhaust_at_the_same_step(self) -> None:
        """Partial EOF remains an error in both native and external step modes."""
        loader = _external_loader()
        sample = BufferedSampleMetadata(SampleKey(0, 0), SampleMetadata(1), 0)
        partial = (_ReaderSnapshot(0, False, (sample,), ((sample,),)), _ReaderSnapshot(1, True, ()))
        exhausted = (_ReaderSnapshot(0, True, ()), _ReaderSnapshot(1, True, ()))
        for native in (False, True):
            with (
                    self.subTest(native=native),
                    patch.object(loader, "_dataset_reader_ranks", frozenset((0, 1))),
                    patch.object(loader, "_batch_sampler_mode", native),
            ):
                with self.assertRaisesRegex(ValueError, "exhausted at different"):
                    loader._build_plan_control(partial)
                self.assertIsNone(loader._build_plan_control(exhausted))

    def test_missing_payload_fails_before_committing_reader(self) -> None:
        """Exact Constructor membership checks prevent silently dropped samples."""
        reader = _StepReader(_steps())
        loader = _external_loader(reader)

        with patch.object(reader, "selected_payloads", return_value=()):
            with self.assertRaisesRegex(ValueError, "payload keys do not match"):
                next(loader)

        self.assertEqual(reader.position, 0)

    def test_active_step_requires_non_none_batch(self) -> None:
        """A collator returning the EOF sentinel remains invalid for an active step."""
        loader = _external_loader()
        with patch.object(loader._data_constructor, "construct", return_value=None):
            with self.assertRaisesRegex(ValueError, "non-None batch"):
                next(loader)
        self.assertEqual(loader._dataset_reader.position, 0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_explicit_and_inferred_metadata_require_batch_sampler(self) -> None:
        """Feature: Producer-defined step boundaries.
        Description: Metadata alone cannot activate a streaming selector, even for an empty Dataset.
        Expectation: Missing step boundaries fail before metadata or payload reads.
        """
        class Dataset:
            requires_distributed_packing = True

            def __len__(self) -> int:
                """Return the aligned Dataset length."""
                return 2

            def __getitem__(self, index: int) -> Any:
                """Expose payload access independently from metadata lookup."""
                raise AssertionError("Build must not read payloads.")

            def get_sample_metadata(self, index: int) -> SampleMetadata:
                """Reject metadata reads before validating the step source."""
                raise AssertionError("Rejected build must not scan metadata.")

        for dataset, options in (
                ([0, 1], {"metadata": [SampleMetadata(1)] * 2}),
                ([], {"metadata": []}),
                (Dataset(), {}),
        ):
            with self.subTest(dataset=type(dataset).__name__), patch(
                    "hyper_parallel.distributed_data.api.create_data_groups",
            ) as groups:
                with self.assertRaisesRegex(ValueError, "Metadata mode requires batch_sampler"):
                    build_distributed_dataloader(
                        dataset, _StandaloneMesh(), DistributedDatasetConfig(seq_len=10, local_batch_size=1),
                        **options,
                    )
                groups.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_online_dataset_still_requires_external_reader(self) -> None:
        """Feature: Producer-defined step boundaries.
        Description: Removing metadata streaming must not re-enable implicit online selection.
        Expectation: Raw online datasets cannot trigger implicit step selection.
        """
        with self.assertRaisesRegex(ValueError, "requires external_step_reader"):
            build_distributed_dataloader(
                [0], _StandaloneMesh(), DistributedDatasetConfig(seq_len=10, local_batch_size=1),
                metadata_fn=lambda _: SampleMetadata(1),
            )

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_native_metadata_plans_before_reads_without_payload_exchange(self) -> None:
        """Feature: Producer-defined step boundaries.
        Description: One sampler yield supplies indices; only assigned samples are materialized.
        Expectation: Only planned indices are read and payload exchange is unused.
        """
        events = []

        class Dataset:
            def __len__(self) -> int:
                """Return the aligned Dataset length."""
                return 6

            def __getitem__(self, index: int) -> Any:
                """Expose payload access independently from metadata lookup."""
                events.append(("read", index))
                return {"id": index}

        loader = build_distributed_dataloader(
            Dataset(), _StandaloneMesh(), DistributedDatasetConfig(seq_len=10, local_batch_size=2),
            batch_sampler=_sampler(), metadata=[SampleMetadata(1)] * 6,
            device="cpu", cost_model=lambda metadata: metadata.cost,
        )
        original_plan = loader._planner.plan

        def plan(*args: Any, **kwargs: Any) -> DistributedPackingPlan:
            """Record planning before the first payload read."""
            events.append(("plan", None))
            return original_plan(*args, **kwargs)

        with patch.object(loader._planner, "plan", side_effect=plan), patch.object(
                loader._data_plane, "exchange_prepared", side_effect=AssertionError("metadata must skip A2A"),
        ), patch.object(
                loader._data_plane, "gather_object_to_planner",
                wraps=loader._data_plane.gather_object_to_planner,
        ) as gather, patch.object(
                loader._data_plane, "broadcast_from_planner", wraps=loader._data_plane.broadcast_from_planner,
        ) as broadcast:
            self.assertEqual(sorted(sample["id"] for sample in next(loader)), [0, 1])
            loader.wait_for_prefetch()
        self.assertEqual(events[0][0], "plan")
        self.assertEqual(sorted(index for kind, index in events if kind == "read"), [0, 1, 2, 3])
        self.assertEqual(gather.call_count, 2)
        self.assertEqual(broadcast.call_count, 2)
        self.assertEqual(len(broadcast.call_args.args[0].selected_keys), 2)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_external_steps_do_not_borrow_future_samples(self) -> None:
        """Feature: Producer-defined step boundaries.
        Description: Complete pack boundaries, not seq_len or lookahead, control membership.
        Expectation: Each fill exposes one whole step regardless of read-ahead settings.
        """
        reader = _StepReader(_steps())
        loader = _external_loader(reader, buffer_size_multiplier=1e308)
        for position in range(3):
            batch = next(loader)
            self.assertEqual(sorted(sample["id"] for row in batch for sample in row),
                             [position * 2, position * 2 + 1])
            loader.wait_for_prefetch()
            self.assertEqual(reader.position, position + 1)
            self.assertEqual(reader.prepare_calls, position + 2)
        self.assertFalse(hasattr(loader, "_step_sample_selector"))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_planner_exception_propagates_before_broadcast(self) -> None:
        """Feature: Producer-defined step boundaries.
        Description: Planner failure must not be turned into a refill request.
        Expectation: Planning failure is raised directly instead of broadcasting control data.
        """
        loader = _external_loader()
        with patch.object(loader._planner, "plan", side_effect=ValueError("bad plan")), patch.object(
                loader._data_plane, "broadcast_from_planner",
        ) as broadcast:
            with self.assertRaisesRegex(ValueError, "bad plan"):
                next(loader)
        broadcast.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_end_of_stream_and_epoch_reset(self) -> None:
        """Feature: Producer-defined step boundaries.
        Description: None denotes EOF; a fresh epoch replays the same explicitly selected steps.
        Expectation: Exhaustion remains stable and set_epoch replays source-defined steps.
        """
        loader = _external_loader()
        expected = list(loader)
        with self.assertRaises(StopIteration):
            next(loader)
        loader.set_epoch(1)
        self.assertEqual(list(loader), expected)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_double_buffer_reuses_worker_and_preserves_pending_result(self) -> None:
        """Feature: Producer-defined step boundaries.
        Description: The background producer prepares exactly one next step while training consumes.
        Expectation: The same worker prepares the next uncommitted step.
        """
        reader = _StepReader(_steps())
        loader = _external_loader(reader)
        next(loader)
        thread = loader._prefetch_thread
        loader.wait_for_prefetch()
        self.assertEqual(reader.position, 1)
        batch = next(loader)
        self.assertEqual(sorted(sample["id"] for row in batch for sample in row), [2, 3])
        self.assertIs(loader._prefetch_thread, thread)
        list(loader)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_background_error_is_raised_by_foreground(self) -> None:
        """Feature: Producer-defined step boundaries.
        Description: A producer exception wakes the consuming thread instead of hanging it.
        Expectation: The consumer receives the original background exception.
        """
        loader = _external_loader()
        with patch.object(loader, "_prepare_next_batch", side_effect=RuntimeError("reader failed")):
            with self.assertRaisesRegex(RuntimeError, "reader failed"):
                next(loader)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_metadata_and_external_checkpoints_replay_pending_step(self) -> None:
        """Feature: Producer-defined step boundaries.
        Description: Saved progress describes delivered steps, not speculative reads.
        Expectation: A resumed loader reproduces the same remaining steps.
        """
        samples = [sample for step in _steps() for row in step for sample in row]

        def metadata_loader() -> Any:
            """Build a metadata-first sampler loader for checkpoint replay."""
            loader = build_distributed_dataloader(
                samples, _StandaloneMesh(),
                DistributedDatasetConfig(seq_len=10, local_batch_size=2),
                metadata=[SampleMetadata(sample["tokens"]) for sample in samples], batch_sampler=_sampler(),
                device="cpu", cost_model=lambda metadata: metadata.cost,
            )
            return loader

        for build in (metadata_loader, _external_loader):
            with self.subTest(build=build):
                loader = build()
                next(loader)
                loader.wait_for_prefetch()
                state = copy.deepcopy(loader.state_dict())
                expected = list(loader)
                resumed = build()
                resumed.load_state_dict(state)
                self.assertEqual(list(resumed), expected)
