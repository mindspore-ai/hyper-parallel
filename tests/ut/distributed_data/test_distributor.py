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
"""Unit tests for plan-driven local-batch distribution helpers."""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import patch

import torch

from hyper_parallel.distributed_data.distributor import (
    TorchMetadataAllGather,
    TorchModelParallelLocalBatchDistributor,
    TorchPackedBytesLocalBatchRedistributor,
    TorchTensorLocalBatchRedistributor,
    _decode_binary_payload,
    _encode_binary_payload,
    _pack_payload_segment,
    shard_local_batch,
)
from hyper_parallel.distributed_data.planner import DistributedBatchPlanner
from hyper_parallel.distributed_data.schema import (
    LocalBatchMeta,
    OnlineLocalBatchMetadata,
    TensorLocalBatchSpec,
    TensorShardSpec,
    WorkloadCost,
)
from hyper_parallel.distributed_data.topology import DataTopology


class _FakeWork:
    """Record explicit waits on asynchronous fake collectives."""

    def __init__(self) -> None:
        """Initialize an incomplete fake work handle."""
        self.waited = False

    def wait(self) -> None:
        """Record collective completion."""
        self.waited = True


class _FakeDistributed:
    """Minimal torch.distributed backend for collective unit tests."""

    def __init__(
        self,
        group_ranks: tuple[int, ...] = (0, 1),
        gathered_objects: tuple[Any, ...] | None = None,
        size_output: torch.Tensor | None = None,
        variable_output: torch.Tensor | None = None,
    ) -> None:
        """Initialize fake collective rank order and optional contributions."""
        self.group_ranks = group_ranks
        self.gathered_objects = gathered_objects
        self.size_output = size_output
        self.variable_output = variable_output
        self.variable_splits: tuple[list[int], list[int]] | None = None
        self.works: list[_FakeWork] = []
        self.broadcast_count = 0
        self.all_gather_count = 0

    def get_process_group_ranks(self, group: Any) -> list[int]:
        """Return fake group ranks."""
        del group
        return list(self.group_ranks)

    @staticmethod
    def get_backend(group: Any) -> str:
        """Return a CPU collective backend for fake tensors."""
        del group
        return "gloo"

    def all_gather_object(self, output: list[Any], value: Any, group: Any) -> None:
        """Populate an object-gather output list."""
        del group
        self.all_gather_count += 1
        contributions = self.gathered_objects or tuple(value for _ in output)
        output[:] = contributions

    def all_to_all_single(
        self,
        output: torch.Tensor,
        input_tensor: torch.Tensor,
        output_split_sizes: list[int] | None = None,
        input_split_sizes: list[int] | None = None,
        group: Any = None,
        async_op: bool = False,
    ) -> _FakeWork:
        """Populate a receive buffer from configured fake collective output."""
        del input_tensor, group, async_op
        configured_output = self.size_output
        if output_split_sizes is not None or input_split_sizes is not None:
            if output_split_sizes is None or input_split_sizes is None:
                raise ValueError("Fake A2A requires both input and output splits.")
            self.variable_splits = (input_split_sizes, output_split_sizes)
            configured_output = self.variable_output
        if configured_output is None:
            raise ValueError("Fake A2A output is not configured.")
        output.resize_(configured_output.shape)
        output.copy_(configured_output)
        work = _FakeWork()
        self.works.append(work)
        return work

    def broadcast(self, tensor: torch.Tensor, src: int, group: Any, async_op: bool = False) -> None:
        """Record one fake tensor broadcast."""
        del tensor, src, group, async_op
        self.broadcast_count += 1


class TestMicroBatchSharding(unittest.TestCase):
    """Validate CP field paths recorded in ``BatchPlan``."""

    def test_shards_only_selected_nested_tensor(self) -> None:
        """Plan paths should leave unrelated batch fields unchanged."""
        micro_batch = {"input_ids": torch.tensor(range(8)), "labels": "replicated"}
        specs = (TensorShardSpec(("input_ids",), 0),)

        with patch("hyper_parallel.distributed_data.distributor.dist", _FakeDistributed()):
            result = shard_local_batch(micro_batch, specs, cp_rank=1, cp_size=2)

        self.assertEqual(result["input_ids"].tolist(), [4, 5, 6, 7])
        self.assertEqual(result["labels"], "replicated")

    def test_rejects_non_divisible_cp_dimension(self) -> None:
        """MVP CP slicing should fail before an uneven collective sequence."""
        micro_batch = {"input_ids": torch.tensor(range(7))}
        specs = (TensorShardSpec(("input_ids",), 0),)

        with patch("hyper_parallel.distributed_data.distributor.dist", _FakeDistributed()):
            with self.assertRaisesRegex(ValueError, "not divisible"):
                shard_local_batch(micro_batch, specs, cp_rank=0, cp_size=2)

    def test_metadata_all_gather_uses_data_owner_order(self) -> None:
        """Process-group order must not change deterministic candidate order."""
        owner_three = LocalBatchMeta(local_batch_id="three")
        owner_nine = LocalBatchMeta(local_batch_id="nine")
        fake_dist = _FakeDistributed(
            group_ranks=(9, 3),
            gathered_objects=((owner_nine,), (owner_three,)),
        )

        with patch("hyper_parallel.distributed_data.distributor.dist", fake_dist):
            synchronizer = TorchMetadataAllGather(group="metadata")
            result = synchronizer.gather((owner_nine,), data_owner_ranks=(3, 9))

        self.assertEqual([metadata.local_batch_id for metadata in result], ["three", "nine"])

    def test_metadata_all_gather_carries_online_tensor_specs(self) -> None:
        """Tensor transport descriptors should reuse the existing metadata collective."""
        owner_three = OnlineLocalBatchMetadata(
            LocalBatchMeta(local_batch_id="three"),
            TensorLocalBatchSpec((3,), "torch.float32", 3),
        )
        owner_nine = OnlineLocalBatchMetadata(
            LocalBatchMeta(local_batch_id="nine"),
            TensorLocalBatchSpec((2,), "torch.float32", 2),
        )
        fake_dist = _FakeDistributed(
            group_ranks=(9, 3),
            gathered_objects=((owner_nine,), (owner_three,)),
        )

        with patch("hyper_parallel.distributed_data.distributor.dist", fake_dist):
            synchronizer = TorchMetadataAllGather(group="metadata")
            result = synchronizer.gather((owner_nine,), data_owner_ranks=(3, 9))

        self.assertEqual(
            [metadata.local_batch_meta.local_batch_id for metadata in result],
            ["three", "nine"],
        )
        self.assertEqual([metadata.tensor_spec.shape for metadata in result], [(3,), (2,)])
        self.assertEqual(fake_dist.all_gather_count, 1)

    @staticmethod
    def _cross_owner_plan() -> tuple[Any, DataTopology]:
        """Build a two-owner plan in which both local batches change owner."""
        metadata = [
            LocalBatchMeta(
                local_batch_id="local",
                cost_hint=WorkloadCost(encoder=1.0),
            ),
            LocalBatchMeta(
                local_batch_id="peer",
                cost_hint=WorkloadCost(encoder=2.0),
            ),
        ]
        plan = DistributedBatchPlanner(2, 1).plan(
            metadata,
            step=0,
            local_batch_offset_start=0,
        )
        topology = DataTopology.from_layout(
            mesh_shape=(2,),
            mesh_dim_names=("dp_shard",),
            rank_list=(0, 1),
            global_rank=0,
        )
        return plan, topology

    @staticmethod
    def _online_tensor_metadata(
        plan: Any,
        tensor_specs: tuple[TensorLocalBatchSpec, ...],
    ) -> tuple[OnlineLocalBatchMetadata, ...]:
        """Build transport metadata in global source-position order."""
        planned_by_position = {
            local_batch.source_position: local_batch
            for local_batch in plan.local_batches
        }
        return tuple(
            OnlineLocalBatchMetadata(planned_by_position[position].meta, tensor_spec)
            for position, tensor_spec in enumerate(tensor_specs)
        )

    def test_packed_bytes_a2a_exchanges_nested_raw_payloads(self) -> None:
        """Packed-byte A2A should preserve nested JPEG/text records without object P2P."""
        plan, topology = self._cross_owner_plan()
        peer_payload = {"image": b"peer-jpeg", "text": "peer", "sizes": (10, 20)}
        peer_segment = _pack_payload_segment(((1, peer_payload),))
        fake_dist = _FakeDistributed(
            group_ranks=(1, 0),
            size_output=torch.tensor([len(peer_segment), 0], dtype=torch.int64),
            variable_output=torch.tensor(list(peer_segment), dtype=torch.uint8),
        )

        with patch("hyper_parallel.distributed_data.distributor.dist", fake_dist):
            redistributor = TorchPackedBytesLocalBatchRedistributor(group="metadata")
            result = redistributor.redistribute(
                ({"image": b"local-jpeg", "text": "local"},),
                plan,
                topology,
                (),
            )

        self.assertEqual(result, {1: peer_payload})
        self.assertEqual(fake_dist.variable_splits[1], [len(peer_segment), 0])
        self.assertTrue(all(work.waited for work in fake_dist.works))

    def test_packed_bytes_preserves_nested_dataloader_tensors(self) -> None:
        """Default online transport should accept a normal nested tensor batch."""
        payload = {
            "input_ids": torch.tensor([[1, 2], [3, 4]], dtype=torch.int64),
            "labels": (torch.tensor([5, 6], dtype=torch.int32),),
            "metadata": {"source": "megatron"},
        }

        with patch("hyper_parallel.distributed_data.distributor.dist", _FakeDistributed()):
            frame = _encode_binary_payload(payload)
            result = _decode_binary_payload(frame)

        self.assertEqual(result["input_ids"].shape, (2, 2))
        self.assertEqual(result["input_ids"].reshape(-1).tolist(), [1, 2, 3, 4])
        self.assertEqual(result["labels"][0].tolist(), [5, 6])
        self.assertEqual(result["metadata"], {"source": "megatron"})

    def test_direct_tensor_a2a_preserves_tensor_storage(self) -> None:
        """Direct mode should exchange uniform tensors without Host serialization."""
        plan, topology = self._cross_owner_plan()
        peer_tensor = torch.tensor([[9.0, 10.0]])
        fake_dist = _FakeDistributed(group_ranks=(1, 0), variable_output=peer_tensor)
        global_metadata = self._online_tensor_metadata(
            plan,
            (
                TensorLocalBatchSpec((2,), "torch.float32", 2),
                TensorLocalBatchSpec((2,), "torch.float32", 2),
            ),
        )

        with patch("hyper_parallel.distributed_data.distributor.dist", fake_dist):
            redistributor = TorchTensorLocalBatchRedistributor(group="metadata")
            result = redistributor.redistribute(
                (torch.tensor([1.0, 2.0]),),
                plan,
                topology,
                global_metadata,
            )

        self.assertEqual(result[1].reshape(-1).tolist(), [9.0, 10.0])
        self.assertEqual(fake_dist.variable_splits, ([1, 0], [1, 0]))
        self.assertEqual(fake_dist.all_gather_count, 0)
        self.assertTrue(all(work.waited for work in fake_dist.works))

    def test_direct_tensor_a2a_flattens_variable_shapes_from_global_metadata(self) -> None:
        """Variable tensor shapes should use element splits and reconstruct the target shape."""
        plan, topology = self._cross_owner_plan()
        fake_dist = _FakeDistributed(
            group_ranks=(1, 0),
            variable_output=torch.tensor([9.0, 10.0, 11.0]),
        )
        global_metadata = self._online_tensor_metadata(
            plan,
            (
                TensorLocalBatchSpec((2,), "torch.float32", 2),
                TensorLocalBatchSpec((3,), "torch.float32", 3),
            ),
        )

        with patch("hyper_parallel.distributed_data.distributor.dist", fake_dist):
            redistributor = TorchTensorLocalBatchRedistributor(group="metadata")
            result = redistributor.redistribute(
                (torch.tensor([1.0, 2.0]),),
                plan,
                topology,
                global_metadata,
            )

        self.assertEqual(result[1].shape, (3,))
        self.assertEqual(result[1].tolist(), [9.0, 10.0, 11.0])
        self.assertEqual(fake_dist.variable_splits, ([2, 0], [3, 0]))
        self.assertEqual(fake_dist.all_gather_count, 0)
        self.assertTrue(all(work.waited for work in fake_dist.works))

    def test_direct_tensor_a2a_rejects_mixed_global_dtypes_before_collective(self) -> None:
        """One A2AV buffer cannot represent multiple element dtypes."""
        plan, topology = self._cross_owner_plan()
        fake_dist = _FakeDistributed(group_ranks=(1, 0))
        global_metadata = self._online_tensor_metadata(
            plan,
            (
                TensorLocalBatchSpec((2,), "torch.float32", 2),
                TensorLocalBatchSpec((3,), "torch.float16", 3),
            ),
        )

        with patch("hyper_parallel.distributed_data.distributor.dist", fake_dist):
            redistributor = TorchTensorLocalBatchRedistributor(group="metadata")
            with self.assertRaisesRegex(ValueError, "one dtype"):
                redistributor.redistribute(
                    (torch.tensor([1.0, 2.0]),),
                    plan,
                    topology,
                    global_metadata,
                )

        self.assertIsNone(fake_dist.variable_splits)

    def test_direct_tensor_a2a_rejects_missing_tensor_descriptor_before_collective(self) -> None:
        """A non-tensor sample marker should fail consistently after metadata synchronization."""
        plan, topology = self._cross_owner_plan()
        fake_dist = _FakeDistributed(group_ranks=(1, 0))
        planned_by_position = {
            local_batch.source_position: local_batch
            for local_batch in plan.local_batches
        }
        global_metadata = (
            OnlineLocalBatchMetadata(planned_by_position[0].meta),
            OnlineLocalBatchMetadata(
                planned_by_position[1].meta,
                TensorLocalBatchSpec((3,), "torch.float32", 3),
            ),
        )

        with patch("hyper_parallel.distributed_data.distributor.dist", fake_dist):
            redistributor = TorchTensorLocalBatchRedistributor(group="metadata")
            with self.assertRaisesRegex(ValueError, "missing tensor metadata"):
                redistributor.redistribute(
                    (torch.tensor([1.0, 2.0]),),
                    plan,
                    topology,
                    global_metadata,
                )

        self.assertIsNone(fake_dist.variable_splits)

    def test_owner_broadcasts_tensor_micro_batch_without_object_serializing_storage(self) -> None:
        """Microbatch structure uses object gather while tensor storage uses broadcast."""
        metadata = [LocalBatchMeta(local_batch_id="0")]
        plan = DistributedBatchPlanner(1, 1).plan(
            metadata,
            step=0,
            local_batch_offset_start=0,
        )
        topology = DataTopology.from_layout(
            mesh_shape=(2,),
            mesh_dim_names=("tp",),
            rank_list=(0, 1),
            global_rank=0,
        )
        fake_dist = _FakeDistributed()
        micro_batch = {"input_ids": torch.tensor((1, 2))}

        with patch("hyper_parallel.distributed_data.distributor.dist", fake_dist):
            distributor = TorchModelParallelLocalBatchDistributor(group="consumer")
            received_plan, result = distributor.distribute(micro_batch, plan, topology)

        self.assertEqual(received_plan.plan_id, plan.plan_id)
        self.assertIs(result["input_ids"], micro_batch["input_ids"])
        self.assertEqual(fake_dist.broadcast_count, 1)
