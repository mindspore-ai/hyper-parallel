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
"""Unit tests for bounded-prefetch distributed dataset orchestration."""

from __future__ import annotations

import threading
import unittest
from contextlib import nullcontext
from typing import Any, Callable, Sequence
from unittest.mock import patch

from hyper_parallel import distributed_data
from hyper_parallel.distributed_data import api as distributed_data_api
from hyper_parallel.distributed_data.distributed_dataset import DistributedDataset
from hyper_parallel.distributed_data.distributor import (
    LocalMicroBatchDistributor,
    LocalMetadataSynchronizer,
    LocalSampleRedistributor,
)
from hyper_parallel.distributed_data.fetcher import (
    MicroBatchFetcher,
    StridedMetadataSource,
    StridedOnlineSampleSource,
)
from hyper_parallel.distributed_data.planner import DistributedBatchPlanner
from hyper_parallel.distributed_data.schema import BatchPlan, SampleMeta, WorkloadCost
from hyper_parallel.distributed_data.topology import DataTopology


class TestDistributedDataPublicApi(unittest.TestCase):
    """Validate the stable package-level construction interface."""

    def test_exports_only_supported_user_contracts(self) -> None:
        """Internal planning and communication components should stay module-scoped."""
        self.assertEqual(
            distributed_data.__all__,
            [
                "CostModel",
                "DistributedDataStep",
                "DistributedDataset",
                "DistributedDatasetConfig",
                "LinearMultimodalCostModel",
                "SampleMeta",
                "TensorShardSpec",
                "WorkloadCost",
                "build_distributed_dataset",
            ],
        )
        for internal_name in (
            "DataTopology",
            "DatasetStateTracker",
            "DistributedBatchPlanner",
            "TorchMicroBatchDistributor",
        ):
            self.assertFalse(hasattr(distributed_data, internal_name))

    def test_builder_selects_online_or_sidecar_metadata(self) -> None:
        """The public builder should require exactly one metadata path."""
        config = distributed_data.DistributedDatasetConfig(micro_batch_size=1, micro_batch_num=1)

        def metadata_fn(sample: Any, sample_id: int) -> SampleMeta:
            """Build unused metadata for boundary validation."""
            del sample
            return SampleMeta(sample_id=sample_id)

        with self.assertRaisesRegex(ValueError, "requires metadata_fn"):
            distributed_data.build_distributed_dataset([], None, config)
        with self.assertRaisesRegex(ValueError, "either online metadata_fn or sidecar metadata"):
            distributed_data.build_distributed_dataset(
                [],
                None,
                config,
                metadata_fn=metadata_fn,
                metadata=[],
            )
        with self.assertRaisesRegex(ValueError, "sample_transport"):
            distributed_data.DistributedDatasetConfig(
                micro_batch_size=1,
                micro_batch_num=1,
                sample_transport="object_p2p",
            )
        with self.assertRaisesRegex(ValueError, "pin_memory must be a boolean"):
            distributed_data.DistributedDatasetConfig(
                micro_batch_size=1,
                micro_batch_num=1,
                pin_memory=1,
            )
        with self.assertRaisesRegex(ValueError, "double_buffer must be a boolean"):
            distributed_data.DistributedDatasetConfig(
                micro_batch_size=1,
                micro_batch_num=1,
                double_buffer=1,
            )
        with self.assertRaisesRegex(ValueError, "num_workers must be a non-negative integer"):
            distributed_data.DistributedDatasetConfig(
                micro_batch_size=1,
                micro_batch_num=1,
                num_workers=-1,
            )
        with self.assertRaisesRegex(ValueError, "prefetch_factor must be a positive integer"):
            distributed_data.DistributedDatasetConfig(
                micro_batch_size=1,
                micro_batch_num=1,
                prefetch_factor=0,
            )
        config = distributed_data.DistributedDatasetConfig(
            micro_batch_size=1,
            micro_batch_num=1,
            prefetch_steps=1,
            double_buffer=True,
            num_workers=2,
            prefetch_factor=3,
        )
        self.assertTrue(config.double_buffer)
        self.assertEqual(config.num_workers, 2)
        self.assertEqual(config.prefetch_factor, 3)

    def test_builder_creates_dedicated_data_groups_in_global_order(self) -> None:
        """Every rank should derive the same metadata and model-group creation sequence."""
        topology = DataTopology.from_layout(
            mesh_shape=(2, 2),
            mesh_dim_names=("dp_shard", "tp"),
            rank_list=(0, 1, 2, 3),
            global_rank=0,
        )
        metadata = [
            SampleMeta(sample_id=index)
            for index in range(4)
        ]
        created: list[tuple[tuple[int, ...], str]] = []
        group_ranks: dict[str, tuple[int, ...]] = {}
        barriers: list[str | None] = []

        def create_named_group(ranks: Sequence[int], group_name: str) -> str:
            """Record one synthetic dedicated group creation."""
            normalized_ranks = tuple(ranks)
            created.append((normalized_ranks, group_name))
            group_ranks[group_name] = normalized_ranks
            return group_name

        with (
            patch("hyper_parallel.distributed_data.api.DataTopology.from_mesh", return_value=topology),
            patch.object(distributed_data_api.platform, "create_named_group", side_effect=create_named_group),
            patch.object(
                distributed_data_api.platform,
                "barrier",
                side_effect=lambda group=None: barriers.append(group),
            ),
            patch.object(
                distributed_data_api.platform,
                "get_process_group_ranks",
                side_effect=lambda group: group_ranks[group],
            ),
        ):
            loader = distributed_data.build_distributed_dataset(
                [f"sample-{index}" for index in range(4)],
                mesh=object(),
                config=distributed_data.DistributedDatasetConfig(micro_batch_size=1, micro_batch_num=1),
                metadata=metadata,
            )

        try:
            self.assertEqual([ranks for ranks, _ in created], [(0, 2), (0, 1), (2, 3)])
            self.assertTrue(created[0][1].startswith("hp_data_metadata_"))
            self.assertTrue(all(name.startswith("hp_data_model_") for _, name in created[1:]))
            self.assertEqual(barriers, [created[0][1], None, created[1][1], None, None])
        finally:
            loader.close()

    def test_online_metadata_preserves_dataset_sample_id(self) -> None:
        """Online metadata must retain the key used to read the map-style dataset."""
        source = StridedOnlineSampleSource(
            ["sample"],
            lambda _sample, sample_id: SampleMeta(sample_id=sample_id + 1),
            shard_rank=0,
            num_shards=1,
        )

        with self.assertRaisesRegex(ValueError, "must preserve sample_id"):
            source.get(0)

    def test_builder_configures_owner_torch_dataloader(self) -> None:
        """The public options should configure one owner-local native DataLoader."""
        dataset = ["sample"]
        metadata = [SampleMeta(sample_id=0)]
        topology = DataTopology.from_layout(
            mesh_shape=(1,),
            mesh_dim_names=("dp_shard",),
            rank_list=(0,),
            global_rank=0,
        )

        def worker_init_fn(worker_id: int) -> None:
            """Provide a synthetic callback for constructor forwarding."""
            del worker_id

        with (
            patch("hyper_parallel.distributed_data.api.DataTopology.from_mesh", return_value=topology),
            patch("hyper_parallel.distributed_data.api.TorchLocalDataLoader") as local_loader_type,
        ):
            loader = distributed_data.build_distributed_dataset(
                dataset,
                mesh=object(),
                config=distributed_data.DistributedDatasetConfig(
                    micro_batch_size=1,
                    micro_batch_num=1,
                    num_workers=2,
                    prefetch_factor=3,
                    pin_memory=True,
                ),
                metadata=metadata,
                collate_fn=tuple,
                worker_init_fn=worker_init_fn,
            )

        try:
            local_loader_type.assert_called_once_with(
                dataset,
                metadata_fn=None,
                collate_fn=tuple,
                num_workers=2,
                prefetch_factor=3,
                pin_memory=True,
                worker_init_fn=worker_init_fn,
                online_metadata=False,
            )
        finally:
            loader.close()
        local_loader_type.return_value.close.assert_called_once_with()

    def test_builder_runs_both_metadata_paths_through_torch_dataloader(self) -> None:
        """The native local loader should preserve both public planning modes."""
        topology = DataTopology.from_layout(
            mesh_shape=(1,),
            mesh_dim_names=("dp_shard",),
            rank_list=(0,),
            global_rank=0,
        )
        dataset = ["sample-0", "sample-1"]
        config = distributed_data.DistributedDatasetConfig(micro_batch_size=2, micro_batch_num=1)

        def metadata_fn(sample: str, sample_id: int) -> SampleMeta:
            """Derive minimal online metadata for the public builder path."""
            del sample
            return SampleMeta(sample_id=sample_id)

        for online in (False, True):
            with (
                self.subTest(online=online),
                patch("hyper_parallel.distributed_data.api.DataTopology.from_mesh", return_value=topology),
            ):
                kwargs = {"metadata_fn": metadata_fn} if online else {
                    "metadata": [SampleMeta(sample_id=0), SampleMeta(sample_id=1)]
                }
                loader = distributed_data.build_distributed_dataset(
                    dataset,
                    mesh=object(),
                    config=config,
                    collate_fn=tuple,
                    **kwargs,
                )
                try:
                    step = next(loader)
                    micro_batch = next(step)

                    self.assertEqual(micro_batch.data, ("sample-0", "sample-1"))
                    loader.commit(step.replay_id)
                finally:
                    loader.close()


class _PeerMetadataSynchronizer:
    """Add a deterministic second owner's contribution without collectives."""

    def gather(
        self,
        local_metadata: Sequence[SampleMeta],
        data_owner_ranks: tuple[int, ...],
    ) -> tuple[SampleMeta, ...]:
        """Return local metadata followed by a synthetic peer contribution."""
        if data_owner_ranks != (0, 1):
            raise ValueError(f"Unexpected owners {data_owner_ranks}.")
        peer_metadata = tuple(
            SampleMeta(
                sample_id=metadata.sample_id + 4,
                cost_hint=WorkloadCost(encoder=metadata.cost_hint.encoder + 0.5),
            )
            for metadata in local_metadata
        )
        return tuple(local_metadata) + peer_metadata


class _RecordingFetcher:
    """Record which heavyweight entries the current owner actually reads."""

    def __init__(self, dataset: list[Any]) -> None:
        """Initialize recording over a small map-style dataset."""
        self._dataset = dataset
        self.sample_ids = []

    def fetch(self, metadata: SampleMeta) -> Any:
        """Record and return one sample."""
        self.sample_ids.append(metadata.sample_id)
        return self._dataset[metadata.sample_id]


class _PinnableValue:
    """Record the thread used to pin one synthetic tensor leaf."""

    def __init__(self, value: str, thread_names: list[str], *, fail: bool = False) -> None:
        """Initialize one synthetic tensor-like value."""
        self.value = value
        self._thread_names = thread_names
        self._fail = fail

    def pin_memory(self) -> str:
        """Return a pinned marker or raise a synthetic allocation error."""
        self._thread_names.append(threading.current_thread().name)
        if self._fail:
            raise RuntimeError("synthetic pin failure")
        return f"pinned-{self.value}"


class _FailFetcher:
    """Fail if a non-owner rank attempts heavyweight data access."""

    def fetch(self, metadata: SampleMeta) -> Any:
        """Reject every fetch attempt."""
        raise AssertionError(f"Non-owner unexpectedly fetched {metadata.sample_id}.")


class _RecordingMapDataset:
    """Record raw map-style reads performed by an online data owner."""

    def __init__(self, samples: list[str]) -> None:
        """Initialize deterministic raw samples."""
        self._samples = samples
        self.sample_ids: list[int] = []

    def __len__(self) -> int:
        """Return global raw sample count."""
        return len(self._samples)

    def __getitem__(self, sample_id: int) -> str:
        """Record and return one raw sample."""
        self.sample_ids.append(sample_id)
        return self._samples[sample_id]


class _SyntheticSampleRedistributor:
    """Provide peer samples while recording owner-loaded raw samples."""

    def __init__(self) -> None:
        """Initialize an empty redistribution record."""
        self.local_sample_batches: list[tuple[Any, ...]] = []
        self.plans: list[BatchPlan] = []
        self.thread_names: list[str] = []

    def redistribute(
        self,
        local_samples: Sequence[Any],
        plan: BatchPlan,
        topology: DataTopology,
    ) -> dict[int, Any]:
        """Return samples planned for data rank zero."""
        self.thread_names.append(threading.current_thread().name)
        local_samples = tuple(local_samples)
        self.local_sample_batches.append(local_samples)
        self.plans.append(plan)
        local_count = len(local_samples)
        source_start = topology.data_rank * local_count
        available = {
            source_start + local_index: sample
            for local_index, sample in enumerate(local_samples)
        }
        for sample in plan.samples:
            available.setdefault(sample.source_position, f"peer-{sample.meta.sample_id}")
        return {
            sample.source_position: available[sample.source_position]
            for sample in plan.samples
            if sample.target_data_rank == topology.data_rank
        }


class _ReceivingMicroBatchDistributor:
    """Stand in for owner-to-TP-peer microbatch communication."""

    def __init__(self, plan: BatchPlan) -> None:
        """Initialize the plan that a remote owner publishes."""
        self._plan = plan
        self._micro_batch_index = 0
        self.thread_names: list[str] = []

    def distribute(
        self,
        micro_batch: Any | None,
        plan: BatchPlan | None,
        topology: DataTopology,
    ) -> tuple[BatchPlan, Any]:
        """Return a synthetic microbatch while asserting owner-only reads."""
        if micro_batch is not None or plan is not None or topology.is_data_owner:
            raise AssertionError("Non-owner distribution received owner-only inputs.")
        self.thread_names.append(threading.current_thread().name)
        result = f"received-{self._micro_batch_index}"
        self._micro_batch_index += 1
        return self._plan, result


class _ReceivingPlanSequenceDistributor:
    """Return one remote online microbatch plan per distribution call."""

    def __init__(self, plans: Sequence[BatchPlan]) -> None:
        """Initialize the ordered remote plan sequence."""
        self._plans = tuple(plans)
        self._micro_batch_index = 0

    def distribute(
        self,
        micro_batch: Any | None,
        plan: BatchPlan | None,
        topology: DataTopology,
    ) -> tuple[BatchPlan, Any]:
        """Return the next remote plan while asserting non-owner behavior."""
        if micro_batch is not None or plan is not None or topology.is_data_owner:
            raise AssertionError("Non-owner online distribution received owner-only inputs.")
        index = self._micro_batch_index
        self._micro_batch_index += 1
        return self._plans[index], f"received-online-{index}"


class _BlockingMicroBatchDistributor:
    """Block selected full-pipeline distribution calls for overlap assertions."""

    def __init__(self, blocked_calls: tuple[int, ...]) -> None:
        """Initialize one event pair for each zero-based blocked call."""
        self._blocked_calls = set(blocked_calls)
        self._lock = threading.Lock()
        self._calls = 0
        self.started = {index: threading.Event() for index in blocked_calls}
        self.release = {index: threading.Event() for index in blocked_calls}
        self.thread_names: list[str] = []

    def distribute(
        self,
        micro_batch: Any | None,
        plan: BatchPlan | None,
        topology: DataTopology,
    ) -> tuple[BatchPlan, Any]:
        """Record the producer thread and wait when this call is selected."""
        with self._lock:
            call_index = self._calls
            self._calls += 1
        self.thread_names.append(threading.current_thread().name)
        if call_index in self._blocked_calls:
            self.started[call_index].set()
            if not self.release[call_index].wait(timeout=5):
                raise RuntimeError(f"Timed out releasing synthetic distribution call {call_index}.")
        return LocalMicroBatchDistributor().distribute(micro_batch, plan, topology)

    def release_all(self) -> None:
        """Release every synthetic blocked distribution call."""
        for event in self.release.values():
            event.set()


class _BlockingMetadataSynchronizer(_PeerMetadataSynchronizer):
    """Block one online metadata collective to prove it runs ahead of consumption."""

    def __init__(self, blocked_call: int) -> None:
        """Initialize a zero-based blocking call and its coordination events."""
        self._blocked_call = blocked_call
        self._calls = 0
        self.started = threading.Event()
        self.release = threading.Event()
        self.thread_names: list[str] = []

    def gather(
        self,
        local_metadata: Sequence[SampleMeta],
        data_owner_ranks: tuple[int, ...],
    ) -> tuple[SampleMeta, ...]:
        """Block the selected gather before returning deterministic peer metadata."""
        call_index = self._calls
        self._calls += 1
        self.thread_names.append(threading.current_thread().name)
        if call_index == self._blocked_call:
            self.started.set()
            if not self.release.wait(timeout=5):
                raise RuntimeError("Timed out releasing synthetic metadata gather.")
        return super().gather(local_metadata, data_owner_ranks)


class _FakeReadyEvent:
    """Record data-stream readiness synchronization across producer and consumer threads."""

    def __init__(self) -> None:
        """Initialize empty record and wait logs."""
        self.recorded: list[tuple[Any, str]] = []
        self.waited: list[tuple[Any, str]] = []

    def record(self, stream: Any) -> None:
        """Record the stream and thread that completed one data slot."""
        self.recorded.append((stream, threading.current_thread().name))

    def wait(self, stream: Any) -> None:
        """Record the compute stream and thread consuming one data slot."""
        self.waited.append((stream, threading.current_thread().name))


class _FakeStreamPlatform:
    """Provide stream/event primitives without requiring accelerator hardware."""

    def __init__(self) -> None:
        """Initialize the generated readiness-event list."""
        self.events: list[_FakeReadyEvent] = []

    @staticmethod
    def get_stream_context() -> Callable[[Any], Any]:
        """Return a context factory accepting the synthetic data stream."""
        return nullcontext

    def new_event(self) -> _FakeReadyEvent:
        """Create and retain one synthetic readiness event."""
        event = _FakeReadyEvent()
        self.events.append(event)
        return event

    @staticmethod
    def get_current_stream() -> str:
        """Return the synthetic training compute stream."""
        return "compute-stream"


def _build_loader(
    prefetch_steps: int = 2,
    *,
    double_buffer: bool = False,
    micro_batch_distributor: Any | None = None,
    prepare_micro_batch: Any | None = None,
    data_stream: Any = None,
):
    metadata = [
        SampleMeta(
            sample_id=index,
            cost_hint=WorkloadCost(encoder=float(index + 1)),
        )
        for index in range(4)
    ]
    topology = DataTopology.from_layout(
        mesh_shape=(2,),
        mesh_dim_names=("dp_shard",),
        rank_list=(0, 1),
        global_rank=0,
    )
    planner = DistributedBatchPlanner(data_parallel_size=2, micro_batch_size=1, micro_batch_num=2)
    fetcher = _RecordingFetcher([f"sample-{index}" for index in range(8)])
    loader = DistributedDataset(
        topology=topology,
        metadata_source=StridedMetadataSource(metadata, shard_rank=0, num_shards=1),
        planner=planner,
        micro_batch_fetcher=MicroBatchFetcher(fetcher, tuple),
        metadata_synchronizer=_PeerMetadataSynchronizer(),
        micro_batch_distributor=micro_batch_distributor or LocalMicroBatchDistributor(),
        prefetch_steps=prefetch_steps,
        double_buffer=double_buffer,
        prepare_micro_batch=prepare_micro_batch,
        data_stream=data_stream,
    )
    return loader, fetcher


def _build_pinning_loader(samples: list[Any], *, online: bool, pin_memory: bool) -> DistributedDataset:
    topology = DataTopology.from_layout(
        mesh_shape=(1,),
        mesh_dim_names=("dp_shard",),
        rank_list=(0,),
        global_rank=0,
    )
    planner = DistributedBatchPlanner(data_parallel_size=1, micro_batch_size=len(samples), micro_batch_num=1)

    def collate_fn(values: list[Any]) -> dict[str, Any]:
        """Create a nested batch that exercises recursive pinning."""
        return {"values": values, "nested": (values[0], ["text"])}

    if online:
        def metadata_fn(sample: Any, sample_id: int) -> SampleMeta:
            """Build online metadata for one synthetic raw sample."""
            del sample
            return SampleMeta(sample_id=sample_id)

        online_source = StridedOnlineSampleSource(samples, metadata_fn, shard_rank=0, num_shards=1)
        return DistributedDataset(
            topology=topology,
            metadata_source=None,
            planner=planner,
            micro_batch_fetcher=MicroBatchFetcher(_FailFetcher(), collate_fn),
            metadata_synchronizer=LocalMetadataSynchronizer(),
            micro_batch_distributor=LocalMicroBatchDistributor(),
            prefetch_steps=1,
            pin_memory=pin_memory,
            online_sample_source=online_source,
            sample_redistributor=LocalSampleRedistributor(),
        )

    metadata = [
        SampleMeta(sample_id=index)
        for index in range(len(samples))
    ]
    return DistributedDataset(
        topology=topology,
        metadata_source=StridedMetadataSource(metadata, shard_rank=0, num_shards=1),
        planner=planner,
        micro_batch_fetcher=MicroBatchFetcher(_RecordingFetcher(samples), collate_fn),
        metadata_synchronizer=LocalMetadataSynchronizer(),
        micro_batch_distributor=LocalMicroBatchDistributor(),
        prefetch_steps=1,
        pin_memory=pin_memory,
    )


def _build_single_microbatch_double_buffer(micro_batch_distributor: Any) -> DistributedDataset:
    """Build two one-microbatch steps for cross-step look-ahead tests."""
    metadata = [
        SampleMeta(sample_id=index)
        for index in range(2)
    ]
    topology = DataTopology.from_layout(
        mesh_shape=(1,),
        mesh_dim_names=("dp_shard",),
        rank_list=(0,),
        global_rank=0,
    )
    return DistributedDataset(
        topology=topology,
        metadata_source=StridedMetadataSource(metadata, shard_rank=0, num_shards=1),
        planner=DistributedBatchPlanner(data_parallel_size=1, micro_batch_size=1, micro_batch_num=1),
        micro_batch_fetcher=MicroBatchFetcher(_RecordingFetcher(["sample-0", "sample-1"]), tuple),
        metadata_synchronizer=LocalMetadataSynchronizer(),
        micro_batch_distributor=micro_batch_distributor,
        prefetch_steps=2,
        double_buffer=True,
    )


class TestDistributedDataset(unittest.TestCase):
    """Validate owner-only reads, bounded look-ahead, and exact resume."""

    def test_fetches_one_microbatch_at_a_time_and_requires_commit(self) -> None:
        """A step should plan globally but retain at most one look-ahead microbatch."""
        loader, fetcher = _build_loader()
        try:
            step = next(loader)

            self.assertLessEqual(len(fetcher.sample_ids), 1)
            self.assertEqual(loader.consumed_offset, 0)
            self.assertEqual(loader.prepared_offset, 4)
            with self.assertRaisesRegex(ValueError, "Fully consume and commit"):
                next(loader)
            with self.assertRaisesRegex(ValueError, "Consume every microbatch"):
                loader.commit("not-ready")

            first = next(step)
            with self.assertRaisesRegex(ValueError, "Consume every microbatch"):
                _ = step.replay_id
            second = next(step)

            self.assertEqual([first.micro_batch_index, second.micro_batch_index], [0, 1])
            self.assertEqual(first.replay_id, second.replay_id)
            self.assertTrue(step.is_complete)
            loader.commit(step.replay_id)
            self.assertEqual(loader.consumed_offset, 2)
        finally:
            loader.close()

    def test_double_buffer_prefetches_full_sidecar_pipeline_and_next_step(self) -> None:
        """The alternate slot should finish distribution for the next execution unit."""
        distributor = _BlockingMicroBatchDistributor(blocked_calls=(1, 2))
        prepare_threads: list[str] = []

        def prepare_micro_batch(micro_batch: Any) -> Any:
            """Record that device preparation moved onto the data producer."""
            prepare_threads.append(threading.current_thread().name)
            return micro_batch

        loader, _ = _build_loader(
            double_buffer=True,
            micro_batch_distributor=distributor,
            prepare_micro_batch=prepare_micro_batch,
        )
        try:
            step = next(loader)
            first = next(step)

            self.assertEqual(first.micro_batch_index, 0)
            self.assertTrue(distributor.started[1].wait(timeout=2))
            self.assertEqual(loader.consumed_offset, 0)
            self.assertTrue(all(name.startswith("hp-data-buffer") for name in distributor.thread_names))
            self.assertTrue(all(name.startswith("hp-data-buffer") for name in prepare_threads))

            distributor.release[1].set()
            second = next(step)

            self.assertEqual(second.micro_batch_index, 1)
            self.assertTrue(distributor.started[2].wait(timeout=2))
            self.assertEqual(loader.consumed_offset, 0)

            distributor.release[2].set()
            loader.commit(step.replay_id)
            next_step = next(loader)
            self.assertEqual(next(next_step).micro_batch_index, 0)
        finally:
            distributor.release_all()
            loader.close()

    def test_double_buffer_with_one_prefetched_step_overlaps_microbatches(self) -> None:
        """One step reservation should still allow overlap within that optimizer step."""
        distributor = _BlockingMicroBatchDistributor(blocked_calls=(1,))
        loader, _ = _build_loader(
            prefetch_steps=1,
            double_buffer=True,
            micro_batch_distributor=distributor,
        )
        try:
            step = next(loader)
            first = next(step)

            self.assertEqual(first.micro_batch_index, 0)
            self.assertTrue(distributor.started[1].wait(timeout=2))

            distributor.release[1].set()
            second = next(step)

            self.assertEqual(second.micro_batch_index, 1)
            loader.commit(step.replay_id)
            self.assertEqual(loader.consumed_offset, 2)
        finally:
            distributor.release_all()
            loader.close()

    def test_double_buffer_prefetches_online_collectives_and_redistribution(self) -> None:
        """Online metadata gather and raw A2A should run before the next consumer request."""
        dataset = _RecordingMapDataset([f"raw-{index}" for index in range(6)])

        def metadata_fn(sample: str, sample_id: int) -> SampleMeta:
            """Build deterministic online metadata for the synthetic sample."""
            del sample
            return SampleMeta(
                sample_id=sample_id,
                cost_hint=WorkloadCost(encoder=float(sample_id + 1)),
            )

        topology = DataTopology.from_layout(
            mesh_shape=(2,),
            mesh_dim_names=("dp_shard",),
            rank_list=(0, 1),
            global_rank=0,
        )
        synchronizer = _BlockingMetadataSynchronizer(blocked_call=1)
        redistributor = _SyntheticSampleRedistributor()
        prepare_threads: list[str] = []
        loader = DistributedDataset(
            topology=topology,
            metadata_source=None,
            planner=DistributedBatchPlanner(data_parallel_size=2, micro_batch_size=1, micro_batch_num=3),
            micro_batch_fetcher=MicroBatchFetcher(_FailFetcher(), tuple),
            metadata_synchronizer=synchronizer,
            micro_batch_distributor=LocalMicroBatchDistributor(),
            prefetch_steps=2,
            double_buffer=True,
            prepare_micro_batch=lambda micro_batch: (
                prepare_threads.append(threading.current_thread().name) or micro_batch
            ),
            online_sample_source=StridedOnlineSampleSource(
                dataset,
                metadata_fn,
                shard_rank=0,
                num_shards=2,
                max_entries=3,
            ),
            sample_redistributor=redistributor,
        )
        try:
            step = next(loader)
            first = next(step)

            self.assertEqual(first.micro_batch_index, 0)
            self.assertTrue(synchronizer.started.wait(timeout=2))
            self.assertTrue(all(name.startswith("hp-data-buffer") for name in synchronizer.thread_names))

            synchronizer.release.set()
            remaining = list(step)

            self.assertEqual([micro_batch.micro_batch_index for micro_batch in [first, *remaining]], [0, 1, 2])
            self.assertEqual(dataset.sample_ids, [0, 2, 4])
            self.assertTrue(all(name.startswith("hp-data-buffer") for name in redistributor.thread_names))
            self.assertTrue(all(name.startswith("hp-data-buffer") for name in prepare_threads))
            loader.commit(step.replay_id)
        finally:
            synchronizer.release.set()
            loader.close()

    def test_double_buffer_waits_for_data_stream_readiness_on_compute_stream(self) -> None:
        """A device-ready slot should establish a stream dependency before consumption."""
        fake_platform = _FakeStreamPlatform()
        loader, _ = _build_loader(double_buffer=True, data_stream="data-stream")
        with patch("hyper_parallel.distributed_data.distributed_dataset.platform", fake_platform):
            try:
                step = next(loader)
                first = next(step)

                self.assertEqual(first.micro_batch_index, 0)
                self.assertTrue(fake_platform.events)
                ready_event = fake_platform.events[0]
                self.assertEqual(ready_event.recorded[0][0], "data-stream")
                self.assertTrue(ready_event.recorded[0][1].startswith("hp-data-buffer"))
                self.assertEqual(ready_event.waited, [("compute-stream", threading.current_thread().name)])
            finally:
                loader.close()

    def test_double_buffer_crosses_step_boundary_when_microbatch_count_is_one(self) -> None:
        """Disabling gradient accumulation should still prepare the next optimizer step."""
        distributor = _BlockingMicroBatchDistributor(blocked_calls=(1,))
        loader = _build_single_microbatch_double_buffer(distributor)
        try:
            first_step = next(loader)
            first = next(first_step)

            self.assertEqual(first.micro_batch_index, 0)
            self.assertTrue(distributor.started[1].wait(timeout=2))
            self.assertEqual(loader.consumed_offset, 0)

            distributor.release[1].set()
            loader.commit(first_step.replay_id)
            second_step = next(loader)
            second = next(second_step)

            self.assertEqual(second.micro_batch_index, 0)
            self.assertEqual(second_step.step, 1)
            loader.commit(second_step.replay_id)
            self.assertEqual(loader.consumed_offset, 2)
        finally:
            distributor.release_all()
            loader.close()

    def test_double_buffer_close_drains_running_collective_task(self) -> None:
        """Closing must not cancel a collective that peer ranks may already have entered."""
        distributor = _BlockingMicroBatchDistributor(blocked_calls=(1,))
        loader = _build_single_microbatch_double_buffer(distributor)
        close_complete = threading.Event()

        def close_loader() -> None:
            """Close the loader and publish completion to the consumer thread."""
            loader.close()
            close_complete.set()

        close_thread = threading.Thread(
            target=close_loader,
            name="synthetic-close",
        )
        try:
            step = next(loader)
            next(step)
            self.assertTrue(distributor.started[1].wait(timeout=2))

            close_thread.start()
            self.assertFalse(close_complete.wait(timeout=0.1))

            distributor.release[1].set()
            self.assertTrue(close_complete.wait(timeout=2))
        finally:
            distributor.release_all()
            if close_thread.is_alive():
                close_thread.join(timeout=2)
            loader.close()

    def test_double_buffer_runs_non_owner_distribution_on_data_producer(self) -> None:
        """TP peers must enter microbatch collectives from the same ordered producer role."""
        metadata = [
            SampleMeta(sample_id=index)
            for index in range(2)
        ]
        topology = DataTopology.from_layout(
            mesh_shape=(2,),
            mesh_dim_names=("tp",),
            rank_list=(0, 1),
            global_rank=1,
        )
        planner = DistributedBatchPlanner(data_parallel_size=1, micro_batch_size=1, micro_batch_num=2)
        distributor = _ReceivingMicroBatchDistributor(planner.plan(metadata, step=0, cursor_start=0))
        loader = DistributedDataset(
            topology=topology,
            metadata_source=StridedMetadataSource(metadata, shard_rank=0, num_shards=1),
            planner=planner,
            micro_batch_fetcher=MicroBatchFetcher(_FailFetcher()),
            metadata_synchronizer=LocalMetadataSynchronizer(),
            micro_batch_distributor=distributor,
            prefetch_steps=2,
            double_buffer=True,
        )
        try:
            step = next(loader)
            micro_batches = list(step)

            self.assertEqual([micro_batch.data for micro_batch in micro_batches], ["received-0", "received-1"])
            self.assertTrue(all(name.startswith("hp-data-buffer") for name in distributor.thread_names))
            loader.commit(step.replay_id)
        finally:
            loader.close()

    def test_online_metadata_balances_and_redistributes_one_microbatch_at_a_time(self) -> None:
        """Unavailable metadata must restrict planning and sample reads to each microbatch."""
        dataset = _RecordingMapDataset([f"raw-{index}" for index in range(6)])

        def metadata_fn(sample: str, sample_id: int) -> SampleMeta:
            """Derive deterministic online cost metadata from one raw sample."""
            del sample
            return SampleMeta(
                sample_id=sample_id,
                cost_hint=WorkloadCost(encoder=float(sample_id + 1)),
            )

        topology = DataTopology.from_layout(
            mesh_shape=(2,),
            mesh_dim_names=("dp_shard",),
            rank_list=(0, 1),
            global_rank=0,
        )
        online_source = StridedOnlineSampleSource(
            dataset,
            metadata_fn,
            shard_rank=0,
            num_shards=2,
            max_entries=3,
        )
        redistributor = _SyntheticSampleRedistributor()
        loader = DistributedDataset(
            topology=topology,
            metadata_source=None,
            planner=DistributedBatchPlanner(data_parallel_size=2, micro_batch_size=1, micro_batch_num=3),
            micro_batch_fetcher=MicroBatchFetcher(_FailFetcher(), tuple),
            metadata_synchronizer=_PeerMetadataSynchronizer(),
            micro_batch_distributor=LocalMicroBatchDistributor(),
            prefetch_steps=1,
            online_sample_source=online_source,
            sample_redistributor=redistributor,
        )
        try:
            self.assertEqual(len(loader), 1)
            step = next(loader)

            self.assertLessEqual(len(dataset.sample_ids), 1)
            first = next(step)
            self.assertEqual(len(redistributor.local_sample_batches), 1)
            self.assertLessEqual(len(dataset.sample_ids), 2)

            remaining = list(step)
            micro_batches = [first, *remaining]
            self.assertEqual([micro_batch.micro_batch_index for micro_batch in micro_batches], [0, 1, 2])
            self.assertEqual(dataset.sample_ids, [0, 2, 4])
            self.assertEqual(
                redistributor.local_sample_batches,
                [("raw-0",), ("raw-2",), ("raw-4",)],
            )
            self.assertEqual([plan.micro_batch_start for plan in redistributor.plans], [0, 1, 2])
            self.assertTrue(all(plan.micro_batch_num == 1 for plan in redistributor.plans))
            loader.commit(step.replay_id)
        finally:
            loader.close()

    def test_pin_memory_uses_dedicated_thread_after_collation(self) -> None:
        """Both metadata paths should recursively pin collated batches off the consumer thread."""
        for online in (False, True):
            with self.subTest(online=online):
                thread_names: list[str] = []
                samples = [
                    _PinnableValue("zero", thread_names),
                    _PinnableValue("one", thread_names),
                ]
                loader = _build_pinning_loader(samples, online=online, pin_memory=True)
                try:
                    step = next(loader)
                    batch = next(step).data

                    self.assertEqual(
                        batch,
                        {
                            "values": ["pinned-zero", "pinned-one"],
                            "nested": ("pinned-zero", ["text"]),
                        },
                    )
                    self.assertTrue(thread_names)
                    self.assertTrue(all(name.startswith("hp-data-pin") for name in thread_names))
                finally:
                    loader.close()

    def test_pin_memory_is_disabled_by_default(self) -> None:
        """Leaving pinning disabled should preserve collated object identity."""
        thread_names: list[str] = []
        samples = [_PinnableValue("zero", thread_names)]
        loader = _build_pinning_loader(samples, online=False, pin_memory=False)
        try:
            step = next(loader)
            batch = next(step).data

            self.assertIs(batch["values"][0], samples[0])
            self.assertEqual(thread_names, [])
        finally:
            loader.close()

    def test_pin_memory_errors_reach_the_consumer(self) -> None:
        """Pinning failures should not be swallowed by the background thread."""
        samples = [_PinnableValue("zero", [], fail=True)]
        loader = _build_pinning_loader(samples, online=False, pin_memory=True)
        try:
            step = next(loader)
            with self.assertRaisesRegex(RuntimeError, "synthetic pin failure"):
                next(step)
        finally:
            loader.close()

    def test_checkpoint_ignores_unconsumed_preparation_and_resumes(self) -> None:
        """Restoring must restart from the last optimizer-step commit."""
        loader, _ = _build_loader()
        try:
            step = next(loader)
            before_commit = loader.state_dict()
            self.assertEqual(before_commit["consumed_offset"], 0)
            list(step)
            loader.commit(step.replay_id)
            consumed_state = loader.state_dict()
            self.assertEqual(consumed_state["consumed_offset"], 2)
        finally:
            loader.close()

        restored, _ = _build_loader()
        try:
            restored.load_state_dict(consumed_state)
            resumed_step = next(restored)
            self.assertEqual(resumed_step.cursor_start, 2)
            self.assertEqual(resumed_step.step, 1)
        finally:
            restored.close()

    def test_tp_peer_receives_micro_batches_without_fetching_dataset(self) -> None:
        """Only the data owner may perform map-style I/O for a DP coordinate."""
        metadata = [
            SampleMeta(sample_id=index)
            for index in range(2)
        ]
        topology = DataTopology.from_layout(
            mesh_shape=(2,),
            mesh_dim_names=("tp",),
            rank_list=(0, 1),
            global_rank=1,
        )
        planner = DistributedBatchPlanner(data_parallel_size=1, micro_batch_size=1, micro_batch_num=2)
        plan = planner.plan(metadata, step=0, cursor_start=0)
        loader = DistributedDataset(
            topology=topology,
            metadata_source=StridedMetadataSource(metadata, shard_rank=0, num_shards=1),
            planner=planner,
            micro_batch_fetcher=MicroBatchFetcher(_FailFetcher()),
            metadata_synchronizer=_PeerMetadataSynchronizer(),
            micro_batch_distributor=_ReceivingMicroBatchDistributor(plan),
            prefetch_steps=1,
        )
        try:
            step = next(loader)
            self.assertEqual([micro_batch.data for micro_batch in step], ["received-0", "received-1"])
        finally:
            loader.close()

    def test_online_tp_peer_receives_microbatch_plans_without_reading_dataset(self) -> None:
        """Non-owner model peers should receive online plans without candidate I/O."""
        dataset = _RecordingMapDataset(["raw-0", "raw-1"])

        def metadata_fn(sample: str, sample_id: int) -> SampleMeta:
            """Build metadata that must remain unused on the non-owner peer."""
            del sample
            return SampleMeta(sample_id=sample_id)

        topology = DataTopology.from_layout(
            mesh_shape=(2,),
            mesh_dim_names=("tp",),
            rank_list=(0, 1),
            global_rank=1,
        )
        planner = DistributedBatchPlanner(data_parallel_size=1, micro_batch_size=1, micro_batch_num=2)
        plans = [
            planner.plan_microbatch(
                [SampleMeta(sample_id=index)],
                step=0,
                cursor_start=index,
                micro_batch_index=index,
            )
            for index in range(2)
        ]
        loader = DistributedDataset(
            topology=topology,
            metadata_source=None,
            planner=planner,
            micro_batch_fetcher=MicroBatchFetcher(_FailFetcher()),
            metadata_synchronizer=LocalMetadataSynchronizer(),
            micro_batch_distributor=_ReceivingPlanSequenceDistributor(plans),
            prefetch_steps=1,
            online_sample_source=StridedOnlineSampleSource(dataset, metadata_fn, shard_rank=0, num_shards=1),
            sample_redistributor=LocalSampleRedistributor(),
        )
        try:
            step = next(loader)
            micro_batches = list(step)

            self.assertEqual(
                [micro_batch.data for micro_batch in micro_batches],
                ["received-online-0", "received-online-1"],
            )
            self.assertEqual(dataset.sample_ids, [])
            loader.commit(step.replay_id)
        finally:
            loader.close()

    def test_metadata_shards_can_be_truncated_to_equal_complete_steps(self) -> None:
        """Uneven strided tails must not make owners call different collective counts."""
        metadata = [
            SampleMeta(sample_id=index)
            for index in range(10)
        ]
        sources = [
            StridedMetadataSource(metadata, shard_rank=rank, num_shards=3, max_entries=2)
            for rank in range(3)
        ]

        self.assertEqual([len(source) for source in sources], [2, 2, 2])
