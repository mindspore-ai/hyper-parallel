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
"""Unit tests for local-batch distributed dataset orchestration."""

from __future__ import annotations

import dataclasses
import inspect
import unittest
from typing import Any, Sequence

from hyper_parallel import distributed_data
from hyper_parallel.distributed_data.data_construct import (
    LocalBatchMetadataView,
    OnlineLocalBatchSource,
    OnlineLocalBatchView,
    SidecarLocalBatchFetcher,
    SidecarLocalBatchSource,
)
from hyper_parallel.distributed_data.distributed_dataset import DistributedDataset
from hyper_parallel.distributed_data.distributor import (
    IdentityModelParallelLocalBatchDistributor,
    LocalMetadataSynchronizer,
)
from hyper_parallel.distributed_data.planner import DistributedBatchPlanner
from hyper_parallel.distributed_data.schema import LocalBatchMeta, OnlineLocalBatchMetadata, WorkloadCost
from hyper_parallel.distributed_data.state import DatasetStateTracker
from hyper_parallel.distributed_data.topology import DataTopology


def _topology(data_parallel_size: int = 1) -> DataTopology:
    return DataTopology.from_layout(
        mesh_shape=(data_parallel_size,),
        mesh_dim_names=("dp",),
        rank_list=tuple(range(data_parallel_size)),
        global_rank=0,
    )


def _sidecar_loader(
    payloads: Sequence[Any],
    *,
    micro_batch_num: int = 2,
    prepare_local_batch: Any = None,
) -> tuple[DistributedDataset, list[int | str]]:
    fetched_ids: list[int | str] = []
    metadata = tuple(LocalBatchMeta(local_batch_id=index) for index in range(len(payloads)))

    def fetch(local_batch_id: int | str) -> Any:
        """Record and return one synthetic local batch."""
        fetched_ids.append(local_batch_id)
        return payloads[int(local_batch_id)]

    source = SidecarLocalBatchSource(metadata, fetch)
    planner = DistributedBatchPlanner(data_parallel_size=1, micro_batch_num=micro_batch_num)
    loader = DistributedDataset(
        topology=_topology(),
        source=source,
        metadata_source=LocalBatchMetadataView(
            source,
            shard_rank=0,
            num_shards=1,
            max_entries=len(payloads),
        ),
        online_source=None,
        planner=planner,
        sidecar_fetcher=SidecarLocalBatchFetcher(source),
        metadata_synchronizer=LocalMetadataSynchronizer(),
        local_batch_redistributor=None,
        model_parallel_distributor=IdentityModelParallelLocalBatchDistributor(),
        prefetch_steps=1,
        prepare_local_batch=prepare_local_batch,
    )
    return loader, fetched_ids


class _PeerMetadataSynchronizer:
    """Append one synthetic peer contribution after rank-zero metadata."""

    def gather(
        self,
        local_metadata: Sequence[OnlineLocalBatchMetadata],
        data_owner_ranks: tuple[int, ...],
    ) -> tuple[OnlineLocalBatchMetadata, ...]:
        """Return rank-zero metadata followed by synthetic peer metadata."""
        if data_owner_ranks != (0, 1):
            raise ValueError(f"Unexpected data owners {data_owner_ranks}.")
        peer = tuple(
            OnlineLocalBatchMetadata(
                LocalBatchMeta(
                    local_batch_id=item.local_batch_meta.local_batch_id + 1,
                    cost_hint=WorkloadCost(encoder=float(10 - index)),
                )
            )
            for index, item in enumerate(local_metadata)
        )
        return tuple(local_metadata) + peer


class _PeerLocalBatchRedistributor:
    """Simulate a whole-step A2A result for data rank zero."""

    def __init__(self) -> None:
        """Initialize redistribution call records."""
        self.calls = 0
        self.local_inputs: tuple[Any, ...] = ()

    @staticmethod
    def describe_local_batch(local_batch: Any) -> None:
        """Return no tensor descriptor for synthetic string payloads."""
        del local_batch

    def redistribute(
        self,
        local_batches: Sequence[Any],
        plan: Any,
        topology: DataTopology,
        global_metadata: Sequence[OnlineLocalBatchMetadata],
    ) -> dict[int, Any]:
        """Return synthetic target-rank payloads keyed by source position."""
        self.calls += 1
        self.local_inputs = tuple(local_batches)
        if topology.data_rank != 0 or len(global_metadata) != 4:
            raise ValueError("Unexpected online whole-step redistribution inputs.")
        return {
            planned.source_position: f"batch-{planned.meta.local_batch_id}"
            for planned in plan.local_batches
            if planned.target_data_rank == topology.data_rank
        }


class TestDistributedDataPublicApi(unittest.TestCase):
    """Validate the source-oriented public API boundary."""

    def test_exports_local_batch_contracts(self) -> None:
        """Public exports should describe complete local batches, not raw samples."""
        for name in (
            "LocalBatch",
            "LocalBatchMeta",
            "OnlineLocalBatchSource",
            "SidecarLocalBatchSource",
        ):
            self.assertIn(name, distributed_data.__all__)
            self.assertTrue(hasattr(distributed_data, name))
        self.assertNotIn("SampleMeta", distributed_data.__all__)

    def test_config_contains_only_distributed_controls(self) -> None:
        """Reading, worker, and raw-sample options belong to the single-card source."""
        field_names = {field.name for field in dataclasses.fields(distributed_data.DistributedDatasetConfig)}

        self.assertEqual(
            field_names,
            {
                "micro_batch_num",
                "prefetch_steps",
                "payload_transport",
                "dp_dim_names",
                "cp_shards",
                "double_buffer",
            },
        )
        with self.assertRaisesRegex(ValueError, "payload_transport"):
            distributed_data.DistributedDatasetConfig(
                micro_batch_num=1,
                payload_transport="object_p2p",
            )

    def test_builder_accepts_a_source_instead_of_dataset_callbacks(self) -> None:
        """The builder should not own metadata derivation, collation, or workers."""
        parameters = inspect.signature(distributed_data.build_distributed_dataset).parameters

        self.assertIn("source", parameters)
        for removed in ("dataset", "metadata", "metadata_fn", "collate_fn", "worker_init_fn"):
            self.assertNotIn(removed, parameters)


class TestLocalBatchSources(unittest.TestCase):
    """Validate the two supported metadata availability points."""

    def test_sidecar_source_does_not_fetch_payload_for_metadata(self) -> None:
        """Reading sidecar entries must not invoke the heavyweight fetch callback."""
        fetched_ids = []
        source = SidecarLocalBatchSource(
            [LocalBatchMeta(local_batch_id="batch-0")],
            fetched_ids.append,
        )

        metadata = source.get_metadata(0)

        self.assertEqual(metadata.local_batch_id, "batch-0")
        self.assertEqual(fetched_ids, [])
        source.fetch(metadata)
        self.assertEqual(fetched_ids, ["batch-0"])

    def test_online_source_derives_metadata_after_materialization(self) -> None:
        """Online metadata should observe the complete processed local batch."""
        events = []

        class _OnlineBatches:
            def __len__(self) -> int:
                """Return one online local batch."""
                return 1

            def __getitem__(self, index: int) -> dict[str, int]:
                """Materialize one online local batch."""
                events.append(("load", index))
                return {"tokens": 32768}

        def metadata_fn(local_batch: dict[str, int], local_batch_id: int) -> LocalBatchMeta:
            """Derive metadata after observing the processed local batch."""
            events.append(("metadata", local_batch["tokens"]))
            return LocalBatchMeta(local_batch_id=local_batch_id, text_tokens=local_batch["tokens"])

        source = OnlineLocalBatchSource(_OnlineBatches(), metadata_fn)

        loaded = source.load(0)

        self.assertEqual(events, [("load", 0), ("metadata", 32768)])
        self.assertEqual(loaded.metadata.text_tokens, 32768)
        self.assertEqual(loaded.data, {"tokens": 32768})


class TestDistributedDataset(unittest.TestCase):
    """Validate planning, materialization, delivery, and checkpoint offsets."""

    def test_sidecar_plans_before_target_rank_fetch_and_commits_offsets(self) -> None:
        """Sidecar entries should produce two local batches per optimizer step."""
        loader, fetched_ids = _sidecar_loader(["batch-0", "batch-1", "batch-2", "batch-3"])
        try:
            self.assertEqual(fetched_ids, [])

            step = next(loader)
            local_batches = tuple(step)

            self.assertEqual([item.local_batch_id for item in local_batches], [0, 1])
            self.assertEqual([item.data for item in local_batches], ["batch-0", "batch-1"])
            self.assertEqual(fetched_ids, [0, 1])
            self.assertEqual({item.plan_id for item in local_batches}, {step.plan_id})
            loader.commit(step.plan_id)
            self.assertEqual(loader.consumed_offset, 2)
            self.assertEqual(loader.state_dict()["local_batches_per_step"], 2)
        finally:
            loader.close()

    def test_online_mode_plans_and_redistributes_the_complete_step(self) -> None:
        """Online mode should load all accumulation slots before one whole-step A2A."""
        loaded_ids = []

        class _OnlineBatches:
            def __len__(self) -> int:
                """Return enough global local batches for two optimizer steps."""
                return 8

            def __getitem__(self, index: int) -> str:
                """Record and materialize one global local-batch index."""
                loaded_ids.append(index)
                return f"batch-{index}"

        def metadata_fn(local_batch: str, local_batch_id: int) -> LocalBatchMeta:
            """Derive synthetic online workload metadata."""
            del local_batch
            return LocalBatchMeta(
                local_batch_id=local_batch_id,
                cost_hint=WorkloadCost(encoder=float(local_batch_id + 1)),
            )

        source = OnlineLocalBatchSource(_OnlineBatches(), metadata_fn)
        planner = DistributedBatchPlanner(data_parallel_size=2, micro_batch_num=2)
        redistributor = _PeerLocalBatchRedistributor()
        loader = DistributedDataset(
            topology=_topology(data_parallel_size=2),
            source=source,
            metadata_source=None,
            online_source=OnlineLocalBatchView(
                source,
                shard_rank=0,
                num_shards=2,
                max_entries=4,
            ),
            planner=planner,
            sidecar_fetcher=None,
            metadata_synchronizer=_PeerMetadataSynchronizer(),
            local_batch_redistributor=redistributor,
            model_parallel_distributor=IdentityModelParallelLocalBatchDistributor(),
            prefetch_steps=1,
        )
        try:
            step = next(loader)
            local_batches = tuple(step)

            self.assertEqual(loaded_ids, [0, 2])
            self.assertEqual(redistributor.calls, 1)
            self.assertEqual(redistributor.local_inputs, ("batch-0", "batch-2"))
            self.assertEqual(len(local_batches), 2)
            self.assertEqual(
                [item.data for item in local_batches],
                [f"batch-{item.local_batch_id}" for item in local_batches],
            )
        finally:
            loader.close()

    def test_prepare_local_batch_runs_after_host_fetch(self) -> None:
        """Device preparation should receive the opaque local-batch payload."""
        prepared = []

        def prepare_local_batch(local_batch: str) -> str:
            """Record and transform one opaque local-batch payload."""
            prepared.append(local_batch)
            return f"prepared:{local_batch}"

        loader, _ = _sidecar_loader(
            ["batch-0", "batch-1"],
            prepare_local_batch=prepare_local_batch,
        )
        try:
            step = next(loader)
            local_batches = tuple(step)

            self.assertEqual(prepared, ["batch-0", "batch-1"])
            self.assertEqual([item.data for item in local_batches], ["prepared:batch-0", "prepared:batch-1"])
        finally:
            loader.close()

    def test_checkpoint_resume_uses_local_batch_offsets(self) -> None:
        """Committed offsets should replay only unconsumed local batches."""
        loader, _ = _sidecar_loader(["batch-0", "batch-1", "batch-2", "batch-3"])
        try:
            first_step = next(loader)
            tuple(first_step)
            loader.commit(first_step.plan_id)
            state = loader.state_dict()
        finally:
            loader.close()

        resumed, _ = _sidecar_loader(["batch-0", "batch-1", "batch-2", "batch-3"])
        try:
            resumed.load_state_dict(state)
            next_step = next(resumed)
            local_batches = tuple(next_step)

            self.assertEqual(next_step.local_batch_offset_start, 2)
            self.assertEqual([item.local_batch_id for item in local_batches], [2, 3])
        finally:
            resumed.close()

    def test_commit_requires_a_fully_consumed_step(self) -> None:
        """Checkpoint state must not advance before all local batches are delivered."""
        loader, _ = _sidecar_loader(["batch-0", "batch-1"])
        try:
            step = next(loader)

            with self.assertRaisesRegex(ValueError, "Consume every local batch"):
                loader.commit("unknown")
            next(step)
            with self.assertRaisesRegex(ValueError, "Consume every local batch"):
                loader.commit("unknown")
        finally:
            loader.close()


class TestDatasetStateTracker(unittest.TestCase):
    """Validate local-batch checkpoint schema boundaries."""

    def test_state_dict_uses_source_and_local_batch_terms(self) -> None:
        """State dictionaries should not expose raw-sample terminology."""
        tracker = DatasetStateTracker(source_size=8, local_batches_per_step=2, prefetch_steps=1)

        state = tracker.state_dict()

        self.assertEqual(state["source_size"], 8)
        self.assertEqual(state["local_batches_per_step"], 2)
        self.assertNotIn("metadata_size", state)
        self.assertNotIn("samples_per_step", state)
