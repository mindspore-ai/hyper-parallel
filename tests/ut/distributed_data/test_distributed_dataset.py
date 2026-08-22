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
from typing import Any, Sequence

from hyper_parallel import distributed_data
from hyper_parallel.distributed_data.distributed_dataset import DistributedDataset
from hyper_parallel.distributed_data.distributor import (
    LocalMetadataSynchronizer,
    LocalOwnerPayloadRedistributor,
    LocalPayloadDistributor,
)
from hyper_parallel.distributed_data.materializer import (
    RankMaterializer,
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
            "TorchPayloadDistributor",
        ):
            self.assertFalse(hasattr(distributed_data, internal_name))

    def test_builder_selects_online_or_sidecar_metadata(self) -> None:
        """The public builder should require exactly one metadata path."""
        config = distributed_data.DistributedDatasetConfig(micro_batch_size=1, micro_batch_count=1)

        def metadata_fn(payload: Any, data_ref: int) -> SampleMeta:
            """Build unused metadata for boundary validation."""
            del payload
            return SampleMeta(sample_id=str(data_ref), source_id="source", data_ref=data_ref)

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
        with self.assertRaisesRegex(ValueError, "owner_payload_transport"):
            distributed_data.DistributedDatasetConfig(
                micro_batch_size=1,
                micro_batch_count=1,
                owner_payload_transport="object_p2p",
            )
        with self.assertRaisesRegex(ValueError, "pin_memory must be a boolean"):
            distributed_data.DistributedDatasetConfig(
                micro_batch_size=1,
                micro_batch_count=1,
                pin_memory=1,
            )


class _PeerMetadataSynchronizer:
    """Add a deterministic second owner's contribution without collectives."""

    def gather(
        self,
        local_metadata: Sequence[SampleMeta],
        owner_ranks: tuple[int, ...],
    ) -> tuple[SampleMeta, ...]:
        """Return local metadata followed by a synthetic peer contribution."""
        if owner_ranks != (0, 1):
            raise ValueError(f"Unexpected owners {owner_ranks}.")
        peer_metadata = tuple(
            SampleMeta(
                sample_id=f"peer-{metadata.sample_id}",
                source_id=metadata.source_id,
                data_ref=metadata.data_ref + 4,
                modality=metadata.modality,
                cost_hint=WorkloadCost(encoder=metadata.cost_hint.encoder + 0.5),
            )
            for metadata in local_metadata
        )
        return tuple(local_metadata) + peer_metadata


class _RecordingMaterializer:
    """Record which heavyweight entries the current owner actually reads."""

    def __init__(self, dataset: list[Any]) -> None:
        """Initialize recording over a small map-style dataset."""
        self._dataset = dataset
        self.data_refs = []

    def materialize(self, metadata: SampleMeta) -> Any:
        """Record and return one sample payload."""
        self.data_refs.append(metadata.data_ref)
        return self._dataset[metadata.data_ref]


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


class _FailMaterializer:
    """Fail if a non-owner rank attempts heavyweight data access."""

    def materialize(self, metadata: SampleMeta) -> Any:
        """Reject every materialization attempt."""
        raise AssertionError(f"Non-owner unexpectedly materialized {metadata.sample_id}.")


class _RecordingMapDataset:
    """Record raw map-style reads performed by an online data owner."""

    def __init__(self, samples: list[str]) -> None:
        """Initialize deterministic raw samples."""
        self._samples = samples
        self.data_refs: list[int] = []

    def __len__(self) -> int:
        """Return global raw sample count."""
        return len(self._samples)

    def __getitem__(self, data_ref: int) -> str:
        """Record and return one raw sample."""
        self.data_refs.append(data_ref)
        return self._samples[data_ref]


class _SyntheticOwnerPayloadRedistributor:
    """Provide peer payloads while recording owner-loaded raw samples."""

    def __init__(self) -> None:
        """Initialize an empty redistribution record."""
        self.local_payload_batches: list[tuple[Any, ...]] = []
        self.plans: list[BatchPlan] = []

    def redistribute(
        self,
        local_payloads: Sequence[Any],
        plan: BatchPlan,
        topology: DataTopology,
    ) -> dict[int, Any]:
        """Return payloads planned for data rank zero."""
        local_payloads = tuple(local_payloads)
        self.local_payload_batches.append(local_payloads)
        self.plans.append(plan)
        local_count = len(local_payloads)
        source_start = topology.data_rank * local_count
        available = {
            source_start + local_index: payload
            for local_index, payload in enumerate(local_payloads)
        }
        for sample in plan.samples:
            available.setdefault(sample.source_position, f"peer-{sample.meta.data_ref}")
        return {
            sample.source_position: available[sample.source_position]
            for sample in plan.samples
            if sample.target_data_rank == topology.data_rank
        }


class _ReceivingPayloadDistributor:
    """Stand in for owner-to-TP-peer payload communication."""

    def __init__(self, plan: BatchPlan) -> None:
        """Initialize the plan that a remote owner publishes."""
        self._plan = plan
        self._micro_batch_index = 0

    def distribute(
        self,
        payload: Any | None,
        plan: BatchPlan | None,
        topology: DataTopology,
    ) -> tuple[BatchPlan, Any]:
        """Return a synthetic payload while asserting owner-only reads."""
        if payload is not None or plan is not None or topology.is_data_owner:
            raise AssertionError("Non-owner payload distribution received owner-only inputs.")
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
        payload: Any | None,
        plan: BatchPlan | None,
        topology: DataTopology,
    ) -> tuple[BatchPlan, Any]:
        """Return the next remote plan while asserting non-owner behavior."""
        if payload is not None or plan is not None or topology.is_data_owner:
            raise AssertionError("Non-owner online distribution received owner-only inputs.")
        index = self._micro_batch_index
        self._micro_batch_index += 1
        return self._plans[index], f"received-online-{index}"


def _build_loader(prefetch_steps: int = 2):
    metadata = [
        SampleMeta(
            sample_id=f"local-{index}",
            source_id="source",
            data_ref=index,
            modality="image_text",
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
    planner = DistributedBatchPlanner(data_world_size=2, micro_batch_size=1, micro_batch_count=2)
    materializer = _RecordingMaterializer([f"sample-{index}" for index in range(8)])
    loader = DistributedDataset(
        topology=topology,
        metadata_source=StridedMetadataSource(metadata, shard_rank=0, num_shards=1),
        planner=planner,
        rank_materializer=RankMaterializer(materializer, tuple),
        metadata_synchronizer=_PeerMetadataSynchronizer(),
        payload_distributor=LocalPayloadDistributor(),
        prefetch_steps=prefetch_steps,
    )
    return loader, materializer


def _build_pinning_loader(samples: list[Any], *, online: bool, pin_memory: bool) -> DistributedDataset:
    topology = DataTopology.from_layout(
        mesh_shape=(1,),
        mesh_dim_names=("dp_shard",),
        rank_list=(0,),
        global_rank=0,
    )
    planner = DistributedBatchPlanner(data_world_size=1, micro_batch_size=len(samples), micro_batch_count=1)

    def collate_fn(values: list[Any]) -> dict[str, Any]:
        """Create a nested batch that exercises recursive pinning."""
        return {"values": values, "nested": (values[0], ["text"])}

    if online:
        def metadata_fn(payload: Any, data_ref: int) -> SampleMeta:
            """Build online metadata for one synthetic raw sample."""
            del payload
            return SampleMeta(sample_id=str(data_ref), source_id="source", data_ref=data_ref)

        online_source = StridedOnlineSampleSource(samples, metadata_fn, shard_rank=0, num_shards=1)
        return DistributedDataset(
            topology=topology,
            metadata_source=None,
            planner=planner,
            rank_materializer=RankMaterializer(_FailMaterializer(), collate_fn),
            metadata_synchronizer=LocalMetadataSynchronizer(),
            payload_distributor=LocalPayloadDistributor(),
            prefetch_steps=1,
            pin_memory=pin_memory,
            online_sample_source=online_source,
            owner_payload_redistributor=LocalOwnerPayloadRedistributor(),
        )

    metadata = [
        SampleMeta(sample_id=str(index), source_id="source", data_ref=index)
        for index in range(len(samples))
    ]
    return DistributedDataset(
        topology=topology,
        metadata_source=StridedMetadataSource(metadata, shard_rank=0, num_shards=1),
        planner=planner,
        rank_materializer=RankMaterializer(_RecordingMaterializer(samples), collate_fn),
        metadata_synchronizer=LocalMetadataSynchronizer(),
        payload_distributor=LocalPayloadDistributor(),
        prefetch_steps=1,
        pin_memory=pin_memory,
    )


class TestDistributedDataset(unittest.TestCase):
    """Validate owner-only reads, bounded look-ahead, and exact resume."""

    def test_materializes_one_microbatch_at_a_time_and_requires_commit(self) -> None:
        """A step should plan globally but retain at most one look-ahead microbatch."""
        loader, materializer = _build_loader()
        try:
            step = next(loader)

            self.assertLessEqual(len(materializer.data_refs), 1)
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
            self.assertFalse(hasattr(step, "payloads"))
            loader.commit(step.replay_id)
            self.assertEqual(loader.consumed_offset, 2)
        finally:
            loader.close()

    def test_online_metadata_balances_and_redistributes_one_microbatch_at_a_time(self) -> None:
        """Unavailable metadata must restrict planning and payload reads to each microbatch."""
        dataset = _RecordingMapDataset([f"raw-{index}" for index in range(6)])

        def metadata_fn(payload: str, data_ref: int) -> SampleMeta:
            """Derive deterministic online cost metadata from one raw sample."""
            return SampleMeta(
                sample_id=payload,
                source_id="source",
                data_ref=data_ref,
                modality="image_text",
                cost_hint=WorkloadCost(encoder=float(data_ref + 1)),
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
        redistributor = _SyntheticOwnerPayloadRedistributor()
        loader = DistributedDataset(
            topology=topology,
            metadata_source=None,
            planner=DistributedBatchPlanner(data_world_size=2, micro_batch_size=1, micro_batch_count=3),
            rank_materializer=RankMaterializer(_FailMaterializer(), tuple),
            metadata_synchronizer=_PeerMetadataSynchronizer(),
            payload_distributor=LocalPayloadDistributor(),
            prefetch_steps=1,
            online_sample_source=online_source,
            owner_payload_redistributor=redistributor,
        )
        try:
            self.assertEqual(len(loader), 1)
            step = next(loader)

            self.assertLessEqual(len(dataset.data_refs), 1)
            first = next(step)
            self.assertEqual(len(redistributor.local_payload_batches), 1)
            self.assertLessEqual(len(dataset.data_refs), 2)

            remaining = list(step)
            payloads = [first, *remaining]
            self.assertEqual([payload.micro_batch_index for payload in payloads], [0, 1, 2])
            self.assertEqual(dataset.data_refs, [0, 2, 4])
            self.assertEqual(
                redistributor.local_payload_batches,
                [("raw-0",), ("raw-2",), ("raw-4",)],
            )
            self.assertEqual([plan.micro_batch_start for plan in redistributor.plans], [0, 1, 2])
            self.assertTrue(all(plan.micro_batch_count == 1 for plan in redistributor.plans))
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

    def test_tp_peer_receives_payload_without_materializing_dataset(self) -> None:
        """Only the data owner may perform map-style I/O for a DP coordinate."""
        metadata = [
            SampleMeta(sample_id=str(index), source_id="source", data_ref=index)
            for index in range(2)
        ]
        topology = DataTopology.from_layout(
            mesh_shape=(2,),
            mesh_dim_names=("tp",),
            rank_list=(0, 1),
            global_rank=1,
        )
        planner = DistributedBatchPlanner(data_world_size=1, micro_batch_size=1, micro_batch_count=2)
        plan = planner.plan(metadata, step=0, cursor_start=0)
        loader = DistributedDataset(
            topology=topology,
            metadata_source=StridedMetadataSource(metadata, shard_rank=0, num_shards=1),
            planner=planner,
            rank_materializer=RankMaterializer(_FailMaterializer()),
            metadata_synchronizer=_PeerMetadataSynchronizer(),
            payload_distributor=_ReceivingPayloadDistributor(plan),
            prefetch_steps=1,
        )
        try:
            step = next(loader)
            self.assertEqual([payload.data for payload in step], ["received-0", "received-1"])
        finally:
            loader.close()

    def test_online_tp_peer_receives_microbatch_plans_without_reading_dataset(self) -> None:
        """Non-owner model peers should receive online plans without candidate I/O."""
        dataset = _RecordingMapDataset(["raw-0", "raw-1"])

        def metadata_fn(payload: str, data_ref: int) -> SampleMeta:
            """Build metadata that must remain unused on the non-owner peer."""
            return SampleMeta(sample_id=payload, source_id="source", data_ref=data_ref)

        topology = DataTopology.from_layout(
            mesh_shape=(2,),
            mesh_dim_names=("tp",),
            rank_list=(0, 1),
            global_rank=1,
        )
        planner = DistributedBatchPlanner(data_world_size=1, micro_batch_size=1, micro_batch_count=2)
        plans = [
            planner.plan_microbatch(
                [SampleMeta(sample_id=str(index), source_id="source", data_ref=index)],
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
            rank_materializer=RankMaterializer(_FailMaterializer()),
            metadata_synchronizer=LocalMetadataSynchronizer(),
            payload_distributor=_ReceivingPlanSequenceDistributor(plans),
            prefetch_steps=1,
            online_sample_source=StridedOnlineSampleSource(dataset, metadata_fn, shard_rank=0, num_shards=1),
            owner_payload_redistributor=LocalOwnerPayloadRedistributor(),
        )
        try:
            step = next(loader)
            payloads = list(step)

            self.assertEqual([payload.data for payload in payloads], ["received-online-0", "received-online-1"])
            self.assertEqual(dataset.data_refs, [])
            loader.commit(step.replay_id)
        finally:
            loader.close()

    def test_metadata_shards_can_be_truncated_to_equal_complete_steps(self) -> None:
        """Uneven strided tails must not make owners call different collective counts."""
        metadata = [
            SampleMeta(sample_id=str(index), source_id="source", data_ref=index)
            for index in range(10)
        ]
        sources = [
            StridedMetadataSource(metadata, shard_rank=rank, num_shards=3, max_entries=2)
            for rank in range(3)
        ]

        self.assertEqual([len(source) for source in sources], [2, 2, 2])
