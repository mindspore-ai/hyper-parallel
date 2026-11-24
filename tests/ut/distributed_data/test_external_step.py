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
"""Tests for the source-only external step API."""

import unittest
from collections.abc import Iterator
from contextlib import closing, nullcontext
from itertools import islice
from typing import Any
from unittest.mock import Mock, PropertyMock, patch

import torch

from hyper_parallel.distributed_data import (
    CausalAttentionCostModel,
    DistributedDatasetConfig,
    SampleMetadata,
    WorkloadCost,
    build_distributed_dataloader,
)
from hyper_parallel.distributed_data.transport import DataPlaneTransport


class _Mesh:
    mesh_shape = (1,)
    mesh_dim_names = ("dp",)
    rank_list = (0,)


class _Source:
    """Source that emits one raw-sample bin per step without checkpoint hooks."""

    def __init__(self) -> None:
        """Initialize two deterministic source steps and their cursor."""
        self.steps = [[[{"id": 0, "tokens": 4}]], [[{"id": 1, "tokens": 5}]]]
        self.position = 0

    def __iter__(self) -> Iterator[list[list[dict[str, int]]]]:
        """Yield complete local steps while advancing the source cursor."""
        while self.position < len(self.steps):
            step = self.steps[self.position]
            self.position += 1
            yield step

    def set_epoch(self, epoch: int) -> None:
        """Restart the deterministic source for any epoch.

        Args:
            epoch: Requested epoch; this fixture repeats the same samples.
        """
        del epoch
        self.position = 0


def _build(source, **options):
    return build_distributed_dataloader(
        None,
        _Mesh(),
        DistributedDatasetConfig(seq_len=16, local_batch_size=1, **options),
        metadata_fn=lambda sample: SampleMetadata(sample["tokens"], sample_id=sample["id"]),
        pack_fn=lambda samples, seq_len: tuple(samples) if sum(item["tokens"] for item in samples) <= seq_len else None,
        collate_fn=list,
        external_step_source=source,
        device="cpu", cost_model=lambda metadata: metadata.cost,
    )


def _samples() -> list[dict]:
    return [
        {"input_ids": torch.tensor([index, index + 1]),
         "metadata": SampleMetadata(2, cost=WorkloadCost(llm=index + 1))}
        for index in range(4)
    ]


def _pack(samples: list[dict], seq_len: int) -> dict:
    del seq_len
    return {"input_ids": torch.cat([sample["input_ids"] for sample in samples])}


class TestExternalStepSource(unittest.TestCase):
    """Verify source iteration, metadata, and externally owned recovery."""

    def test_device_prefetch_depth_requires_native_sampler(self) -> None:
        """Feature: External-step prefetch configuration.
        Description: Request device lookahead depths supported only by the native sampler path.
        Expectation: Invalid external depths fail before any process groups are constructed.
        """
        for depth in (0, 2):
            with self.subTest(depth=depth), patch(
                    "hyper_parallel.distributed_data.packed_balancing._create_locality_groups",
            ) as groups, self.assertRaisesRegex(ValueError, "device_prefetch_depth requires a native batch_sampler"):
                _build(_Source(), device_prefetch_depth=depth)
            groups.assert_not_called()

    def test_non_planner_rank_consumes_source_without_plan_broadcast(self) -> None:
        """Feature: Independent planning for external steps.
        Description: Disable plan broadcast on a non-Planner rank with either collective schedule.
        Expectation: The rank plans both source steps locally and returns all samples.
        """
        for backend in ("gloo", "hccl"):
            with self.subTest(backend=backend), closing(
                    _build(_Source(), broadcast_plan=False),
            ) as loader, patch.object(
                    DataPlaneTransport, "communication_backend", new_callable=PropertyMock, return_value=backend,
            ), patch.object(
                    DataPlaneTransport, "planner_rank", new_callable=PropertyMock, return_value=1,
            ), patch.object(
                    loader._transport, "broadcast_from_planner", side_effect=AssertionError("plan broadcast used"),
            ), patch.object(loader._planner, "plan", wraps=loader._planner.plan) as plan:
                self.assertEqual(list(loader), [[({"id": 0, "tokens": 4},)], [({"id": 1, "tokens": 5},)]])
                self.assertEqual(plan.call_count, 2)

    def test_source_steps_and_optional_epoch_hook(self) -> None:
        """Feature: External complete-step sources.
        Description: Consume two selected steps with the automatic balancing pipeline.
        Expectation: Membership is preserved and set_epoch restarts the external source.
        """
        source = _Source()
        loader = _build(source)
        try:
            with self.assertNoLogs("hyper_parallel.distributed_data.balance_logging", level="INFO"):
                expected = list(loader)
            self.assertEqual(expected, [[({"id": 0, "tokens": 4},)], [({"id": 1, "tokens": 5},)]])
            loader.set_epoch(1)
            self.assertEqual(list(loader), expected)
        finally:
            loader.close()

    def test_source_does_not_require_checkpoint_hooks(self) -> None:
        """Feature: External source lifecycle validation.
        Description: Consume plain iterable steps, then wrap a caller-restored source.
        Expectation: HP consumes the supplied position without requiring checkpoint or epoch hooks.
        """
        source = [[[{"id": 0, "tokens": 4}]], [[{"id": 1, "tokens": 5}]]]
        loader = _build(source)
        try:
            self.assertEqual(list(loader), [[({"id": 0, "tokens": 4},)], [({"id": 1, "tokens": 5},)]])
        finally:
            loader.close()
        # The caller reconstructs its source at the next step before wrapping it.
        resumed = _build(source[1:])
        try:
            self.assertEqual(list(resumed), [[({"id": 1, "tokens": 5},)]])
        finally:
            resumed.close()


class TestExternalStepCallbacks(unittest.TestCase):
    """Verify raw-step callbacks, buffering and source lifecycle without process groups."""

    def test_device_handoff_preserves_complete_batch_and_host_view(self) -> None:
        """Feature: Complete external-step device handoff.
        Description: Collate two steps as lists or tuples and stage separate pinned copies.
        Expectation: Delivery preserves both bins and publishes only the delivered step's original Host view.
        """
        for container in (list, tuple):
            with self.subTest(container=container):
                samples = _samples()
                accelerator = Mock()
                accelerator.stream.side_effect = lambda _stream: nullcontext()
                accelerator.Event.return_value.query.return_value = False
                host_batches = []

                def collate(bins: list[Any], container=container, record_batch=host_batches.append) -> Any:
                    """Retain original collated views to distinguish them from pinned staging."""
                    batch = container(bins)
                    record_batch(batch)
                    return batch

                def pin(batch: Any) -> Any:
                    """Model distinct pinned storage without requiring an accelerator."""
                    return type(batch)({"input_ids": item["input_ids"].clone()} for item in batch)

                def move(batch: Any, device: torch.device) -> Any:
                    """Model a device batch while retaining its container structure."""
                    self.assertEqual(device, torch.device("cuda:0"))
                    return type(batch)(
                        {"input_ids": item["input_ids"].tolist(), "storage": Mock()} for item in batch
                    )

                with patch.object(torch, "cuda", accelerator), patch(
                        "hyper_parallel.distributed_data.device_prefetch._pin_memory", side_effect=pin,
                ), patch(
                        "hyper_parallel.distributed_data.device_prefetch._move_to_device", side_effect=move,
                ), closing(build_distributed_dataloader(
                        None, _Mesh(), DistributedDatasetConfig(
                            seq_len=8, local_batch_size=2, communication_backend="gloo",
                        ),
                        external_step_source=[[samples[:2], samples[2:]], [samples[:2], samples[2:]]],
                        metadata_fn=lambda sample: sample["metadata"], pack_fn=_pack, collate_fn=collate,
                        device="cuda:0", cost_model=lambda metadata: metadata.cost,
                )) as loader:
                    expected = [[0, 1, 1, 2], [2, 3, 3, 4]]
                    for step in range(2):
                        batch = next(loader)
                        loader.wait_for_prefetch()
                        self.assertIsInstance(batch, container)
                        self.assertIs(loader.last_host_batch, host_batches[step])
                        self.assertEqual([item["input_ids"] for item in batch], expected)
                        self.assertEqual([item["input_ids"].tolist() for item in loader.last_host_batch], expected)
                        for item in batch:
                            item["storage"].record_stream.assert_called_once_with(accelerator.current_stream.return_value)
                    self.assertEqual(accelerator.current_stream.return_value.wait_event.call_count, 2)

    def test_shared_metadata_keeps_independent_occurrences(self) -> None:
        """Feature: Local-step metadata binding.
        Description: Return one shared metadata object for all raw samples in a step.
        Expectation: The planner sees distinct keys and leaves the original metadata unchanged.
        """
        samples = _samples()
        shared_metadata = SampleMetadata(2, cost=WorkloadCost(llm=1))
        estimated = []

        def estimate(metadata: SampleMetadata) -> WorkloadCost:
            """Record the runtime key seen by the workload model."""
            estimated.append(metadata)
            return WorkloadCost(llm=2)

        with closing(build_distributed_dataloader(
                None, _Mesh(), DistributedDatasetConfig(seq_len=8, local_batch_size=1),
                external_step_source=[[samples]], metadata_fn=lambda _: shared_metadata,
                pack_fn=_pack, collate_fn=list, device="cpu", cost_model=estimate,
        )) as loader:
            batch = next(loader)
            self.assertEqual(batch[0]["input_ids"].tolist(), [0, 1, 1, 2, 2, 3, 3, 4])
            self.assertEqual(loader.last_balance_stats["cost_after"], (8,))
        self.assertEqual(len({item.key for item in estimated}), 4)
        self.assertTrue(all(item is not shared_metadata for item in estimated))
        self.assertIsNone(shared_metadata.key)
        self.assertEqual(shared_metadata.cost, WorkloadCost(llm=1))

    def test_source_limit_stops_prefetch_and_reports_custom_cost(self) -> None:
        """Feature: Buffered local balancing.
        Description: Apply a custom cost model and truncate the external source with islice.
        Expectation: Logging uses modeled costs and prefetch does not read beyond the source limit.
        """
        samples = _samples()
        reads = []

        def source() -> Iterator[list[list[dict]]]:
            """Record every physical source read, including speculative reads."""
            for step in range(2):
                reads.append(step)
                yield [samples[step * 2:step * 2 + 2]]

        config = DistributedDatasetConfig(seq_len=8, local_batch_size=1, log_balance=True)
        with closing(build_distributed_dataloader(
                None, _Mesh(), config, device="cpu",
                external_step_source=islice(source(), 1), metadata_fn=lambda sample: sample["metadata"],
                pack_fn=_pack, collate_fn=list,
                cost_model=lambda metadata: WorkloadCost(llm=metadata.cost.llm * 10),
        )) as loader:
            with self.assertLogs("hyper_parallel.distributed_data.balance_logging", level="INFO") as logs:
                batch = next(loader)
            self.assertEqual(batch[0]["input_ids"].tolist(), [0, 1, 1, 2])
            stats = dict(loader.last_balance_stats)
            self.assertEqual(stats["cost_before"], (30,))
            self.assertEqual(stats["bins_before"][0][0], {"samples": 2, "seq_len": 4, "cost": 30, "padding": 4})
            self.assertEqual(stats["bins_after"][0][0], {"samples": 2, "seq_len": 4, "cost": 30, "padding": 4})
            self.assertIn("mb0: seq_len=4, samples=2, cost=30", logs.output[0])
            self.assertIn("[dp-balance] iteration 1 :", logs.output[0])
            self.assertIn("dp0: send 0 samples, recv 0 samples", logs.output[0])
            self.assertEqual(loader.group_ranks, (0,))
            with self.assertRaises(StopIteration):
                next(loader)
        self.assertEqual(reads, [0])

    def test_padding_cost_is_included_in_rank_and_bin_statistics(self) -> None:
        """Feature: Padding-aware external-step diagnostics.
        Description: Plan two underfilled packs with the causal Attention square model.
        Expectation: Both layouts report document and PAD costs matching the planner objective.
        """
        config = DistributedDatasetConfig(seq_len=12, local_batch_size=2, log_balance=True)
        with closing(build_distributed_dataloader(
                None, _Mesh(), config, device="cpu", external_step_source=[[[_samples()[0]], [_samples()[1]]]],
                metadata_fn=lambda sample: sample["metadata"], pack_fn=_pack, collate_fn=list,
                cost_model=CausalAttentionCostModel(),
        )) as loader:
            with self.assertLogs("hyper_parallel.distributed_data.balance_logging", level="INFO") as logs:
                next(loader)
            stats = loader.last_balance_stats
            for layout in ("before", "after"):
                self.assertEqual(stats[f"cost_{layout}"], (208,))
                self.assertEqual(stats[f"bins_{layout}"][0], (
                    {"samples": 1, "seq_len": 2, "padding": 10, "cost": 104},
                    {"samples": 1, "seq_len": 2, "padding": 10, "cost": 104},
                ))
            self.assertEqual(stats["original_objective"], 208)
            self.assertEqual(stats["accepted_objective"], 208)
            self.assertIn("cost=104, padding=10", logs.output[0])

    def test_default_balancing_requires_backbone_dimensions(self) -> None:
        """Feature: Default cost model.
        Description: Build enabled balancing without model dimensions or a callback.
        Expectation: Startup rejects the missing configuration.
        """
        with self.assertRaisesRegex(ValueError, "model_config"):
            build_distributed_dataloader(
                None, _Mesh(), DistributedDatasetConfig(seq_len=8, local_batch_size=1),
                external_step_source=[], metadata_fn=lambda sample: sample["metadata"],
                pack_fn=_pack, collate_fn=list, device="cpu",
            )

    def test_epoch_and_checkpoint_contract(self) -> None:
        """Feature: External source lifecycle.
        Description: Change epochs and request unsupported local checkpoint operations.
        Expectation: Epochs reach the source and checkpoint operations fail explicitly.
        """
        source = Mock()
        source.__iter__ = Mock(side_effect=lambda: iter([[_samples()]]))
        with closing(build_distributed_dataloader(
                None, _Mesh(), DistributedDatasetConfig(seq_len=8, local_batch_size=1),
                external_step_source=source, metadata_fn=lambda sample: sample["metadata"],
                pack_fn=_pack, collate_fn=list, device="cpu", cost_model=lambda metadata: metadata.cost,
        )) as loader:
            first = next(loader)[0]["input_ids"]
            loader.set_epoch(3)
            source.set_epoch.assert_called_once_with(3)
            self.assertEqual(loader.step, 0)
            self.assertIsNone(loader.last_host_batch)
            torch.testing.assert_close(next(loader)[0]["input_ids"], first)
            with self.assertRaisesRegex(NotImplementedError, "checkpoint"):
                loader.state_dict()
            with self.assertRaisesRegex(NotImplementedError, "checkpoint"):
                loader.load_state_dict({})

    def test_background_collator_error_reaches_consumer(self) -> None:
        """Feature: Background failure propagation.
        Description: Raise from a user collator on the producer thread.
        Expectation: The waiting training thread receives the original exception.
        """
        failure = RuntimeError("collation failed")
        with closing(build_distributed_dataloader(
                None, _Mesh(), DistributedDatasetConfig(seq_len=8, local_batch_size=1),
                external_step_source=[[_samples()]], metadata_fn=lambda sample: sample["metadata"],
                pack_fn=Mock(side_effect=failure), collate_fn=list,
                device="cpu", cost_model=lambda metadata: metadata.cost,
        )) as loader:
            with self.assertRaises(RuntimeError) as caught:
                next(loader)
            self.assertIs(caught.exception, failure)
