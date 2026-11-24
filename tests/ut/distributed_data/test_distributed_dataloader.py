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
from dataclasses import replace
from itertools import product
from threading import Event, current_thread
from typing import Any
from unittest.mock import Mock, PropertyMock, patch

import torch

from hyper_parallel.data.parallel import build_dataset_batch_sampler
from hyper_parallel.distributed_data import DistributedDatasetConfig, SampleMetadata, build_distributed_dataloader
from hyper_parallel.distributed_data.batch_sampler import _ReaderPlanInput
from hyper_parallel.distributed_data.planner import DynamicPackingPlanner
from hyper_parallel.distributed_data.schema import DistributedPackingPlan, SampleKey, WorkloadCost
from hyper_parallel.distributed_data.transport import DataPlaneTransport


class _StandaloneMesh:
    mesh_shape = (1,)
    mesh_dim_names = ("dp",)
    rank_list = (0,)


def _steps():
    return [
        [[{"id": 0, "tokens": 6}, {"id": 1, "tokens": 4}]],
        [[{"id": 2, "tokens": 6}, {"id": 3, "tokens": 4}]],
        [[{"id": 4, "tokens": 6}, {"id": 5, "tokens": 4}]],
    ]


def _native_loader(**options):
    samples = [step[0] for step in _steps()]
    options.setdefault("communication_backend", "gloo")
    return build_distributed_dataloader(
        samples, _StandaloneMesh(), DistributedDatasetConfig(seq_len=10, local_batch_size=1, **options),
        batch_sampler=_sampler(size=len(samples), local_batch_size=1),
        metadata_fn=lambda row: SampleMetadata(sum(sample["tokens"] for sample in row)),
        device="cpu", cost_model=lambda metadata: metadata.cost,
    )


def _sampler(size=6, local_batch_size=2):
    return build_dataset_batch_sampler(
        total_samples=size, micro_batch_size=local_batch_size, global_batch_size=local_batch_size,
        dp_world_size=1, dp_rank=0,
    )


def _window_reader_inputs(source_bins, reference_bins, local_size):
    """Reverse each Reader's metadata order independently of its original pack layout."""
    return tuple(_ReaderPlanInput(
        metadata=tuple(reversed(tuple(
            item for index, group in enumerate(source_bins)
            if index // local_size % 2 == rank for item in group
        ))),
        original_batches=tuple(
            reference_bins[(step * 2 + rank) * local_size:(step * 2 + rank + 1) * local_size]
            for step in range(3)
        ),
    ) for rank in range(2))


class TestDistributedDataLoaderEndToEnd(unittest.TestCase):
    """Verify that each native sampler plan consumes one complete step."""

    def test_collectives_and_construction_share_one_producer(self) -> None:
        """Feature: Independent single-round prefetch.
        Description: Consume steps with implicit or repeated explicit prefetch hooks.
        Expectation: Reads, planning, communication and staging all run on one background thread.
        """
        for explicit, broadcast_plan in product((False, True), repeat=2):
            with self.subTest(explicit=explicit, broadcast_plan=broadcast_plan), _native_loader(
                    broadcast_plan=broadcast_plan,
            ) as loader:
                events = []

                def record(name: str, callback: Callable[..., Any], record_event=events.append) -> Callable[..., Any]:
                    """Record stage ownership without changing its behavior."""
                    def run(*args: Any, **kwargs: Any) -> Any:
                        """Record the producer thread before running a stage."""
                        record_event((name, current_thread()))
                        return callback(*args, **kwargs)
                    return run

                with patch.object(
                        loader, "_prepare_local_metadata", side_effect=record("read", loader._prepare_local_metadata),
                ), patch.object(
                        loader, "_build_plan_control", side_effect=record("plan", loader._build_plan_control),
                ), patch.object(
                        loader._data_plane, "all_gather_object",
                        side_effect=record("gather", loader._data_plane.all_gather_object),
                ), patch.object(
                        loader._data_plane, "gather_object_to_planner",
                        side_effect=record("gather", loader._data_plane.gather_object_to_planner),
                ), patch.object(
                        loader._data_plane, "begin_exchange_prepared",
                        side_effect=record("exchange", loader._data_plane.begin_exchange_prepared),
                ), patch.object(
                        loader, "_stage_batch", side_effect=record("stage", loader._stage_batch),
                ), patch.object(
                        loader, "_broadcast_batch", side_effect=record("model_broadcast", loader._broadcast_batch),
                ):
                    self.assertEqual(next(loader)[0][0]["id"], 0)
                    worker = loader._prefetch_thread
                    if explicit:
                        for _ in range(2):
                            loader.prefetch()
                    loader.wait_for_prefetch()
                    self.assertEqual(loader.state_dict()["_index_sampler_state"]["consumed_samples"], 1)
                    self.assertEqual(next(loader)[0][0]["id"], 2)
                    self.assertIs(worker, loader._prefetch_thread)
                    self.assertEqual(len(list(loader)), 1)
                self.assertEqual({name for name, _ in events},
                                 {"read", "plan", "gather", "exchange", "stage", "model_broadcast"})
                self.assertEqual({thread for _, thread in events}, {worker})
                self.assertIsNot(worker, current_thread())


    def test_single_round_hooks_and_checkpoint_do_not_wait_for_future_work(self) -> None:
        """Feature: Nonblocking single-round training hooks.
        Description: Hold the next step at each data preparation stage without calling Trainer hooks.
        Expectation: Background preparation starts automatically and hooks/checkpoints return while it is blocked.
        """
        for stage in ("read", "gather", "plan", "exchange", "stage", "broadcast"):
            with self.subTest(stage=stage), _native_loader() as loader:
                started, release = Event(), Event()
                owner, method = {
                    "read": (loader, "_prepare_local_metadata"),
                    "gather": (loader._data_plane, "gather_object_to_planner"),
                    "plan": (loader, "_build_plan_control"),
                    "exchange": (loader._data_plane, "begin_exchange_prepared"),
                    "stage": (loader, "_stage_batch"),
                    "broadcast": (loader, "_broadcast_batch"),
                }[stage]
                original = getattr(owner, method)

                def gated(
                        *args: Any, loader=loader, started=started, release=release, original=original, **kwargs: Any,
                ) -> Any:
                    """Hold the next step until foreground checks finish."""
                    if loader._step == 1:
                        started.set()
                        if not release.wait(5):
                            raise RuntimeError("future stage timed out")
                    return original(*args, **kwargs)

                try:
                    with patch.object(owner, method, side_effect=gated):
                        self.assertEqual(next(loader)[0][0]["id"], 0)
                        self.assertTrue(started.wait(5))
                        for _ in range(2):
                            loader.prefetch()
                        self.assertEqual(loader.state_dict()["_index_sampler_state"]["consumed_samples"], 1)
                        self.assertFalse(release.is_set())
                        release.set()
                        self.assertEqual(next(loader)[0][0]["id"], 2)
                        loader.wait_for_prefetch()
                finally:
                    release.set()


    def test_metadata_wait_error_reaches_consumer(self) -> None:
        """Feature: Metadata completion error propagation.
        Description: Fail metadata gathering for the next single-round step.
        Expectation: The original error reaches the consumer without advancing delivered progress.
        """
        with _native_loader() as loader:
            failure = ValueError("metadata gather failed")
            original = loader._data_plane.gather_object_to_planner

            def gather(value: Any) -> Any:
                """Fail only the speculative step's metadata exchange."""
                if loader._step == 1:
                    raise failure
                return original(value)

            with patch.object(loader._data_plane, "gather_object_to_planner", side_effect=gather):
                next(loader)
                loader.prefetch()
                with self.assertRaises(ValueError) as caught:
                    next(loader)
            self.assertIs(caught.exception, failure)
            self.assertEqual(loader.state_dict()["_index_sampler_state"]["consumed_samples"], 1)


    def test_non_planner_rank_builds_batches_without_plan_broadcast(self) -> None:
        """Feature: Independent planning on every data-plane rank.
        Description: Run online and metadata loading through both prefetch paths on a non-Planner rank.
        Expectation: All batches and EOF are produced without centralized gather or plan broadcast.
        """
        for backend, metadata_mode in product(("gloo", "hccl"), (False, True)):
            with self.subTest(backend=backend, metadata_mode=metadata_mode):
                options = {"metadata": [SampleMetadata(1)] * 6} if metadata_mode else {
                    "metadata_fn": lambda _: SampleMetadata(1),
                }
                with build_distributed_dataloader(
                        list(range(6)), _StandaloneMesh(),
                        DistributedDatasetConfig(
                            seq_len=10, local_batch_size=2, communication_backend="gloo",
                            metadata_mode=metadata_mode, broadcast_plan=False,
                        ),
                        batch_sampler=_sampler(), device="cpu", cost_model=lambda metadata: metadata.cost, **options,
                ) as loader, patch.object(
                        DataPlaneTransport, "communication_backend", new_callable=PropertyMock, return_value=backend,
                ), patch.object(
                        DataPlaneTransport, "planner_rank", new_callable=PropertyMock, return_value=1,
                ), patch.object(
                        loader._data_plane, "broadcast_from_planner", side_effect=AssertionError("plan broadcast used"),
                ), patch.object(
                        loader._data_plane, "gather_object_to_planner",
                        side_effect=AssertionError("central gather used"),
                ), patch.object(loader._planner, "plan", wraps=loader._planner.plan) as plan:
                    self.assertEqual(list(loader), [(0, 1), (2, 3), (4, 5)])
                    self.assertEqual(plan.call_count, 3)

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

                def move(batch: Any, device: torch.device, record_copy=copies.append) -> dict[str, Any]:
                    """Record an asynchronous copy without requiring accelerator hardware."""
                    record_copy((batch, device, current_thread().name))
                    return {"data": batch, "storage": Mock()}

                options = {"metadata": [SampleMetadata(1)] * 6} if metadata_mode else {
                    "metadata_fn": lambda _: SampleMetadata(1),
                }
                with patch.object(torch, "cuda", accelerator), patch(
                        "hyper_parallel.distributed_data.device_prefetch._pin_memory", side_effect=lambda value: value,
                ), patch(
                        "hyper_parallel.distributed_data.device_prefetch._move_to_device", side_effect=move,
                ):
                    with build_distributed_dataloader(
                            list(range(6)), _StandaloneMesh(),
                            DistributedDatasetConfig(
                                seq_len=10, local_batch_size=2, communication_backend="gloo",
                                metadata_mode=metadata_mode,
                            ),
                            batch_sampler=_sampler(), device="cuda:0",
                            cost_model=lambda metadata: metadata.cost, **options,
                    ) as loader:
                        broadcast = Mock(side_effect=lambda value: value)
                        loader._model_transport = Mock(broadcast=broadcast)
                        first = next(loader)
                        self.assertEqual(first["data"], (0, 1))
                        self.assertIs(broadcast.call_args_list[0].args[0], first)
                        self.assertGreaterEqual(accelerator.current_stream.return_value.wait_event.call_count, 1)
                        self.assertGreaterEqual(first["storage"].record_stream.call_count, 1)
                        loader.wait_for_prefetch()
                        self.assertEqual(len(copies), 2)
                        self.assertTrue(all(thread == current_thread().name for _, _, thread in copies))
                        state = loader.state_dict()
                        self.assertEqual(state["_index_sampler_state"]["consumed_samples"], 2)
                        self.assertEqual(len(copies), 2)
                        self.assertEqual(next(loader)["data"], (2, 3))

    def test_plan_preserves_source_groups_and_filters_non_readers(self) -> None:
        """Feature: Native sampler planning.
        Description: Non-reader ranks in a non-contiguous group are filtered without changing selected indices.
        Expectation: The planner receives one singleton bin per selected Dataset output.
        """
        loader = _native_loader()
        first = SampleMetadata(4, key=SampleKey(2, 8, 7))
        second = SampleMetadata(6, key=SampleKey(6, 3, 42))
        gathered_metadata = (
            _ReaderPlanInput((first,), (((first.key,),),)), _ReaderPlanInput(),
            _ReaderPlanInput((second,), (((second.key,),),)),
        )
        planner = DynamicPackingPlanner(
            data_parallel_size=2, seq_len=10, local_batch_size=1, cost_model=lambda metadata: metadata.cost,
        )
        with (
                patch.object(loader, "_dataset_reader_ranks", frozenset((2, 6))),
                patch.object(loader, "_topology", replace(loader._topology, constructor_ranks=(2, 6))),
                patch.object(loader._data_plane, "_ranks", (2, 4, 6)),
                patch.object(loader, "_planner", planner),
                patch.object(planner, "plan", wraps=planner.plan) as plan_step,
        ):
            plan = loader._build_plan_control(gathered_metadata)

        samples = plan_step.call_args.args[0]
        self.assertEqual(tuple(item.key for item in samples), (first.key, second.key))
        self.assertEqual(tuple(item.key.global_sample_position for item in samples), (7, 42))
        self.assertEqual(plan_step.call_args.kwargs["reference_bins"], ((first.key,), (second.key,)))
        self.assertEqual(set(plan.selected_keys), {first.key, second.key})

    def test_reader_only_rank_routes_payloads_without_constructing(self) -> None:
        """Feature: Separate data-plane roles.
        Description: Process an assigned plan on an online or metadata Reader without Constructor ownership.
        Expectation: Online payloads are routed; direct reads and construction remain Constructor-only.
        """
        for metadata_mode in (False, True):
            with self.subTest(metadata_mode=metadata_mode), _native_loader() as loader:
                loader._dataset_reader.prepare_next_step(1)
                samples = loader._dataset_reader.metadata()
                plan = loader._planner.plan(samples, reference_bins=((samples[0].key,),), step=0)
                transport = Mock()
                transport.exchange_prepared.return_value = {}
                constructor = Mock()
                direct_reader = Mock()
                with (
                        patch.object(loader, "_topology", replace(
                            loader._topology, constructor_rank=1, constructor_ranks=(1,),
                        )),
                        patch.object(loader, "_metadata_mode", metadata_mode),
                        patch.object(loader, "_data_plane", transport),
                        patch.object(loader, "_data_constructor", constructor),
                        patch.object(loader, "_direct_sample_loader", direct_reader),
                        patch.object(loader, "_next_plan_control", return_value=plan),
                ):
                    self.assertIsNone(loader._produce_on_data_plane())
                constructor.construct.assert_not_called()
                direct_reader.fetch_samples.assert_not_called()
                self.assertEqual(loader._pending_local_keys, {samples[0].key})
                if metadata_mode:
                    transport.prepare_exchange.assert_not_called()
                    transport.exchange_prepared.assert_not_called()
                else:
                    transport.prepare_exchange.assert_called_once_with({1: [(samples[0].key, _steps()[0][0])]})
                    transport.exchange_prepared.assert_called_once_with(transport.prepare_exchange.return_value)

    def test_readers_must_exhaust_at_the_same_step(self) -> None:
        """Feature: Native sampler exhaustion.
        Description: Dataset Reader ranks reach EOF at different rounds.
        Expectation: The planner rejects partial exhaustion before constructing a batch.
        """
        loader = _native_loader()
        sample = SampleMetadata(1, key=SampleKey(0, 0))
        partial = (_ReaderPlanInput((sample,), (((sample.key,),),)), _ReaderPlanInput())
        exhausted = (_ReaderPlanInput(), _ReaderPlanInput())
        with (
                patch.object(loader, "_dataset_reader_ranks", frozenset((0, 1))),
                patch.object(loader, "_topology", replace(loader._topology, constructor_ranks=(0, 1))),
                patch.object(loader._data_plane, "_ranks", (0, 1)),
        ):
            with self.assertRaisesRegex(ValueError, "exhausted at different"):
                loader._build_plan_control(partial)
            self.assertIsNone(loader._build_plan_control(exhausted))

    def test_window_reference_preserves_source_groups_and_mesh_order(self) -> None:
        """Feature: Reader-owned original layouts.
        Description: Plan repeated and split records with collective ranks in a different order from DP ranks.
        Expectation: Source bins, original objective, accepted plans and fallback match the explicit reference.
        """
        costs = (9, 8, 2, 1, 7, 6, 4, 3, 10, 9, 2, 1)
        for local_size, split, gain in product((1, 2), (False, True), (0.0, 0.99)):
            with self.subTest(local_size=local_size, split=split, gain=gain), _native_loader() as loader:
                source_bins = tuple(tuple(
                    SampleMetadata(
                        1, cost=WorkloadCost(llm=costs[position] / (2 if split else 1)),
                        key=SampleKey((6, 2)[position // local_size % 2], position % 3, 24 + position, sequence),
                    ) for sequence in range(2 if split else 1)
                ) for position in range(6 * local_size))
                reference_bins = tuple(tuple(item.key for item in group) for group in source_bins)
                inputs = _window_reader_inputs(source_bins, reference_bins, local_size)
                planner = DynamicPackingPlanner(
                    data_parallel_size=2, seq_len=16, local_batch_size=local_size,
                    cost_model=lambda metadata: metadata.cost, min_balance_gain=gain,
                )
                expected_samples = tuple(item for group in source_bins for item in group)
                expected_plans = planner.plan_window(expected_samples, reference_bins=reference_bins, step=6)
                expected_decision = dict(planner.last_balance_decision)
                with (
                        patch.object(loader, "_topology", replace(loader._topology, constructor_ranks=(6, 2))),
                        patch.object(loader, "_dataset_reader_ranks", frozenset((2, 6))),
                        patch.object(loader._data_plane, "_ranks", (2, 4, 6)),
                        patch.object(loader, "_planner", planner),
                        patch.object(loader, "_step", 6),
                        patch.object(planner, "plan_window", wraps=planner.plan_window) as plan_window,
                ):
                    plans = loader._build_plan_control((inputs[1], _ReaderPlanInput(), inputs[0]))
                self.assertEqual(plan_window.call_args.kwargs["reference_bins"], reference_bins)
                self.assertEqual(plan_window.call_args.args[0], expected_samples)
                self.assertEqual(plans, expected_plans)
                self.assertEqual(planner.last_balance_decision, expected_decision)
                self.assertEqual(expected_decision["original_objective"], 18 if local_size == 1 else 49)
                if gain:
                    self.assertFalse(expected_decision["accepted"])
                    self.assertEqual(tuple(
                        packing_bin.sample_keys
                        for plan in plans for batch in plan.local_batches for packing_bin in batch
                    ), reference_bins)

    def test_original_layout_rejects_unequal_or_incomplete_windows(self) -> None:
        """Feature: Source-window shape validation.
        Description: Supply unequal round counts or an incomplete local batch.
        Expectation: Planning fails before zip could silently truncate a Reader's window.
        """
        key = SampleKey(0, 0)
        item = SampleMetadata(1, key=key)
        full = _ReaderPlanInput((item,), (((key,), (key,)),) * 2)
        invalid = (
            _ReaderPlanInput((item,), full.original_batches[:1]),
            _ReaderPlanInput((item,), (((key,),),) * 2),
        )
        with _native_loader() as loader:
            planner = DynamicPackingPlanner(
                data_parallel_size=2, seq_len=16, local_batch_size=2, cost_model=lambda metadata: metadata.cost,
            )
            with patch.object(loader, "_planner", planner):
                for other in invalid:
                    with self.subTest(batches=other.original_batches), self.assertRaisesRegex(
                            ValueError, "equal complete micro-batch windows",
                    ):
                        loader._plan_native_batch([full, other])

    def test_missing_payload_fails_before_committing_reader(self) -> None:
        """Feature: Native payload validation.
        Description: The Reader returns an incomplete selected payload set.
        Expectation: Construction fails before advancing the sampler cursor.
        """
        loader = _native_loader()
        reader = loader._dataset_reader

        with patch.object(reader, "selected_payloads", return_value=()):
            with self.assertRaisesRegex(ValueError, "payload keys do not match"):
                next(loader)

        self.assertEqual(reader.batch_position, 0)

    def test_active_step_requires_non_none_batch(self) -> None:
        """Feature: Native batch construction.
        Description: The constructor returns the end-of-stream sentinel for an active step.
        Expectation: The loader rejects the invalid non-None batch contract.
        """
        loader = _native_loader()
        with patch.object(loader._data_constructor, "construct", return_value=None):
            with self.assertRaisesRegex(ValueError, "non-None batch"):
                next(loader)
        self.assertEqual(loader._dataset_reader.batch_position, 0)

    def test_metadata_mode_requires_batch_sampler(self) -> None:
        """Feature: Producer-defined step boundaries.
        Description: Metadata alone cannot activate a streaming selector, even for an empty Dataset.
        Expectation: Missing step boundaries fail before metadata or payload reads.
        """
        class Dataset:
            def __len__(self) -> int:
                """Return the aligned Dataset length."""
                return 2

            def __getitem__(self, index: int) -> Any:
                """Expose payload access independently from metadata lookup."""
                raise AssertionError("Build must not read payloads.")

        for dataset, options in (
                ([0, 1], {"metadata": [SampleMetadata(1)] * 2}),
                ([], {"metadata": []}),
                (Dataset(), {}),
        ):
            metadata_mode = "metadata" in options or isinstance(dataset, Dataset)
            with self.subTest(dataset=type(dataset).__name__), patch(
                    "hyper_parallel.distributed_data.api.create_data_groups",
            ) as groups:
                with self.assertRaisesRegex(ValueError, "Metadata mode requires batch_sampler"):
                    build_distributed_dataloader(
                        dataset, _StandaloneMesh(), DistributedDatasetConfig(
                            seq_len=10, local_batch_size=1, metadata_mode=metadata_mode,
                        ),
                        **options,
                    )
                groups.assert_not_called()

    def test_online_dataset_requires_explicit_step_source(self) -> None:
        """Feature: Producer-defined step boundaries.
        Description: Removing metadata streaming must not re-enable implicit online selection.
        Expectation: Raw online datasets cannot trigger implicit step selection.
        """
        with self.assertRaisesRegex(ValueError, "requires batch_sampler or external_step_source"):
            build_distributed_dataloader(
                [0], _StandaloneMesh(), DistributedDatasetConfig(seq_len=10, local_batch_size=1),
                metadata_fn=lambda _: SampleMetadata(1),
            )

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
            Dataset(), _StandaloneMesh(), DistributedDatasetConfig(
                seq_len=10, local_batch_size=2, metadata_mode=True,
            ),
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

    def test_sampler_steps_do_not_borrow_future_samples(self) -> None:
        """Feature: Producer-defined step boundaries.
        Description: Sampler rounds control membership without a read-ahead selector.
        Expectation: Each delivered round preserves its samples and advances one committed step.
        """
        loader = _native_loader()
        reader = loader._dataset_reader
        with patch.object(reader, "prepare_next_step", wraps=reader.prepare_next_step) as prepare:
            for position in range(3):
                batch = next(loader)
                self.assertEqual(sorted(sample["id"] for row in batch for sample in row),
                                 [position * 2, position * 2 + 1])
                loader.wait_for_prefetch()
                self.assertEqual(loader.state_dict()["_index_sampler_state"]["consumed_samples"], position + 1)
                self.assertEqual(prepare.call_count, position + 2)
        self.assertFalse(hasattr(loader, "_step_sample_selector"))

    def test_planner_exception_propagates_before_broadcast(self) -> None:
        """Feature: Producer-defined step boundaries.
        Description: Planner failure must not be turned into a refill request.
        Expectation: Planning failure is raised directly instead of broadcasting control data.
        """
        loader = _native_loader()
        with patch.object(loader._planner, "plan", side_effect=ValueError("bad plan")), patch.object(
                loader._data_plane, "broadcast_from_planner",
        ) as broadcast:
            with self.assertRaisesRegex(ValueError, "bad plan"):
                next(loader)
        broadcast.assert_not_called()

    def test_end_of_stream_and_epoch_reset(self) -> None:
        """Feature: Producer-defined step boundaries.
        Description: None denotes EOF; a fresh epoch replays the same explicitly selected steps.
        Expectation: Exhaustion remains stable and set_epoch replays source-defined steps.
        """
        loader = _native_loader()
        expected = list(loader)
        with self.assertRaises(StopIteration):
            next(loader)
        loader.set_epoch(1)
        self.assertEqual(list(loader), expected)

    def test_double_buffer_reuses_worker_and_preserves_pending_result(self) -> None:
        """Feature: Producer-defined step boundaries.
        Description: The background producer prepares exactly one next step while training consumes.
        Expectation: The same worker prepares the next uncommitted step.
        """
        loader = _native_loader()
        reader = loader._dataset_reader
        next(loader)
        thread = loader._prefetch_thread
        loader.wait_for_prefetch()
        self.assertEqual(loader.state_dict()["_index_sampler_state"]["consumed_samples"], 1)
        batch = next(loader)
        self.assertEqual(sorted(sample["id"] for row in batch for sample in row), [2, 3])
        self.assertIs(loader._prefetch_thread, thread)
        list(loader)

    def test_background_error_is_raised_by_foreground(self) -> None:
        """Feature: Producer-defined step boundaries.
        Description: A producer exception wakes the consuming thread instead of hanging it.
        Expectation: The consumer receives the original background exception.
        """
        loader = _native_loader()
        with patch.object(loader, "_prepare_next_batch", side_effect=RuntimeError("reader failed")):
            with self.assertRaisesRegex(RuntimeError, "reader failed"):
                next(loader)

    def test_metadata_and_online_sampler_checkpoints_replay_pending_step(self) -> None:
        """Feature: Producer-defined step boundaries.
        Description: Saved progress describes delivered steps, not speculative reads.
        Expectation: A resumed loader reproduces the same remaining steps.
        """
        samples = [sample for step in _steps() for row in step for sample in row]

        def metadata_loader() -> Any:
            """Build a metadata-first sampler loader for checkpoint replay."""
            loader = build_distributed_dataloader(
                samples, _StandaloneMesh(),
                DistributedDatasetConfig(seq_len=10, local_batch_size=2, metadata_mode=True),
                metadata=[SampleMetadata(sample["tokens"]) for sample in samples], batch_sampler=_sampler(),
                device="cpu", cost_model=lambda metadata: metadata.cost,
            )
            return loader

        for build in (metadata_loader, _native_loader):
            with self.subTest(build=build):
                loader = build()
                next(loader)
                loader.wait_for_prefetch()
                state = copy.deepcopy(loader.state_dict())
                expected = list(loader)
                resumed = build()
                resumed.load_state_dict(state)
                self.assertEqual(list(resumed), expected)
