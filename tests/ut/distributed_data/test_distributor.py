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
from types import SimpleNamespace
from typing import Any, Sequence
from unittest.mock import patch

import numpy as np

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
from hyper_parallel.platform.platform import PlatformType


class _FakeTensor:
    """Minimal tensor shape used to test backend-independent CP routing."""

    def __init__(self, values: Any, dtype: Any = np.float32) -> None:
        """Store values in a small NumPy-backed tensor."""
        self.array = np.asarray(values, dtype=dtype)
        self.shape = self.array.shape
        self.dtype = self.array.dtype
        self.device = "cpu"

    @property
    def values(self) -> tuple[Any, ...]:
        """Return flattened values for concise assertions."""
        return tuple(self.array.reshape(-1))

    def contiguous(self) -> "_FakeTensor":
        """Return an already-contiguous fake tensor."""
        return self

    def detach(self) -> "_FakeTensor":
        """Return a fake tensor without gradient state."""
        return self

    def cpu(self) -> "_FakeTensor":
        """Return the host-resident fake tensor."""
        return self

    def view(self, dtype: Any) -> "_FakeTensor":
        """Reinterpret fake tensor storage with a different dtype."""
        return _FakeTensor(self.array.view(dtype), dtype)

    def reshape(self, shape: tuple[int, ...]) -> "_FakeTensor":
        """Return one reshaped fake tensor."""
        return _FakeTensor(self.array.reshape(shape), self.dtype)

    def to(self, device: Any, non_blocking: bool = False) -> "_FakeTensor":
        """Record a logical device move without changing test storage."""
        del device, non_blocking
        return self

    def __getitem__(self, index: int | slice) -> "_FakeTensor":
        """Return one leading-dimension slice."""
        return _FakeTensor(self.array[index], self.dtype)


class _FakeWork:
    """Record explicit waits on asynchronous fake collectives."""

    def __init__(self) -> None:
        """Initialize an incomplete fake work handle."""
        self.waited = False

    def wait(self) -> None:
        """Record collective completion."""
        self.waited = True


class _FakePlatform:
    """Minimal platform operations used by ``shard_local_batch``."""

    platform_type = PlatformType.PYTORCH
    tensor_dtype = SimpleNamespace(int64=np.int64, uint8=np.uint8)

    def __init__(
        self,
        group_ranks: tuple[int, ...] = (0, 1),
        gathered_objects: tuple[Any, ...] | None = None,
        size_output: _FakeTensor | None = None,
        variable_output: _FakeTensor | None = None,
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

    @staticmethod
    def is_tensor(value: Any) -> bool:
        """Recognize fake tensors."""
        return isinstance(value, _FakeTensor)

    @staticmethod
    def chunk(value: _FakeTensor, split_dim: int, split_size: int, index: int) -> _FakeTensor:
        """Return one equal fake-tensor chunk."""
        if split_dim != 0:
            raise ValueError(f"Unexpected split_dim {split_dim}.")
        width = value.shape[0] // split_size
        return _FakeTensor(value.array[index * width:(index + 1) * width], value.dtype)

    @staticmethod
    def tensor(values: Any, dtype: Any = None, device: Any = None) -> _FakeTensor:
        """Create a fake tensor from Python values."""
        del device
        return _FakeTensor(values, dtype)

    @staticmethod
    def from_numpy(array: np.ndarray) -> _FakeTensor:
        """Create a fake tensor from a NumPy array."""
        return _FakeTensor(array, array.dtype)

    @staticmethod
    def str_to_dtype(dtype_name: str) -> np.dtype:
        """Resolve a serialized fake dtype name."""
        return np.dtype(dtype_name)

    @staticmethod
    def tensor_to_numpy(tensor: _FakeTensor) -> np.ndarray:
        """Expose fake tensor storage as a NumPy array."""
        return tensor.array

    @staticmethod
    def cat(tensors: Sequence[_FakeTensor], dim: int = 0) -> _FakeTensor:
        """Concatenate fake tensors along one dimension."""
        return _FakeTensor(np.concatenate([tensor.array for tensor in tensors], axis=dim), tensors[0].dtype)

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
        input_tensor: _FakeTensor,
        output_shape: list[int],
        group: Any,
        async_op: bool = False,
    ) -> tuple[_FakeTensor, _FakeWork]:
        """Return configured byte-split metadata."""
        del input_tensor, output_shape, group, async_op
        work = _FakeWork()
        self.works.append(work)
        return self.size_output, work

    def variable_all_to_all_single(
        self,
        input_tensor: _FakeTensor,
        input_splits: list[int],
        output_splits: list[int],
        group: Any,
        async_op: bool = False,
    ) -> tuple[_FakeTensor, _FakeWork]:
        """Return a configured packed-byte or direct-tensor receive buffer."""
        del input_tensor, group, async_op
        self.variable_splits = (input_splits, output_splits)
        work = _FakeWork()
        self.works.append(work)
        return self.variable_output, work

    def broadcast(self, tensor: _FakeTensor, src: int, group: Any, async_op: bool = False) -> None:
        """Record one fake tensor broadcast."""
        del tensor, src, group, async_op
        self.broadcast_count += 1

class TestMicroBatchSharding(unittest.TestCase):
    """Validate CP field paths recorded in ``BatchPlan``."""

    def test_shards_only_selected_nested_tensor(self) -> None:
        """Plan paths should leave unrelated batch fields unchanged."""
        micro_batch = {"input_ids": _FakeTensor(range(8)), "labels": "replicated"}
        specs = (TensorShardSpec(("input_ids",), 0),)

        with patch("hyper_parallel.distributed_data.distributor.platform", _FakePlatform()):
            result = shard_local_batch(micro_batch, specs, cp_rank=1, cp_size=2)

        self.assertEqual(result["input_ids"].values, (4, 5, 6, 7))
        self.assertEqual(result["labels"], "replicated")

    def test_rejects_non_divisible_cp_dimension(self) -> None:
        """MVP CP slicing should fail before an uneven collective sequence."""
        micro_batch = {"input_ids": _FakeTensor(range(7))}
        specs = (TensorShardSpec(("input_ids",), 0),)

        with patch("hyper_parallel.distributed_data.distributor.platform", _FakePlatform()):
            with self.assertRaisesRegex(ValueError, "not divisible"):
                shard_local_batch(micro_batch, specs, cp_rank=0, cp_size=2)

    def test_metadata_all_gather_uses_data_owner_order(self) -> None:
        """Process-group order must not change deterministic candidate order."""
        owner_three = LocalBatchMeta(local_batch_id="three")
        owner_nine = LocalBatchMeta(local_batch_id="nine")
        fake_platform = _FakePlatform(
            group_ranks=(9, 3),
            gathered_objects=((owner_nine,), (owner_three,)),
        )

        with patch("hyper_parallel.distributed_data.distributor.platform", fake_platform):
            synchronizer = TorchMetadataAllGather(group="metadata")
            result = synchronizer.gather((owner_nine,), data_owner_ranks=(3, 9))

        self.assertEqual([metadata.local_batch_id for metadata in result], ["three", "nine"])

    def test_metadata_all_gather_carries_online_tensor_specs(self) -> None:
        """Tensor transport descriptors should reuse the existing metadata collective."""
        owner_three = OnlineLocalBatchMetadata(
            LocalBatchMeta(local_batch_id="three"),
            TensorLocalBatchSpec((3,), "float32", 3),
        )
        owner_nine = OnlineLocalBatchMetadata(
            LocalBatchMeta(local_batch_id="nine"),
            TensorLocalBatchSpec((2,), "float32", 2),
        )
        fake_platform = _FakePlatform(
            group_ranks=(9, 3),
            gathered_objects=((owner_nine,), (owner_three,)),
        )

        with patch("hyper_parallel.distributed_data.distributor.platform", fake_platform):
            synchronizer = TorchMetadataAllGather(group="metadata")
            result = synchronizer.gather((owner_nine,), data_owner_ranks=(3, 9))

        self.assertEqual(
            [metadata.local_batch_meta.local_batch_id for metadata in result],
            ["three", "nine"],
        )
        self.assertEqual([metadata.tensor_spec.shape for metadata in result], [(3,), (2,)])
        self.assertEqual(fake_platform.all_gather_count, 1)

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
        fake_platform = _FakePlatform(
            group_ranks=(1, 0),
            size_output=_FakeTensor([len(peer_segment), 0], np.int64),
            variable_output=_FakeTensor(np.frombuffer(peer_segment, dtype=np.uint8), np.uint8),
        )

        with patch("hyper_parallel.distributed_data.distributor.platform", fake_platform):
            redistributor = TorchPackedBytesLocalBatchRedistributor(group="metadata")
            result = redistributor.redistribute(
                ({"image": b"local-jpeg", "text": "local"},),
                plan,
                topology,
                (),
            )

        self.assertEqual(result, {1: peer_payload})
        self.assertEqual(fake_platform.variable_splits[1], [len(peer_segment), 0])
        self.assertTrue(all(work.waited for work in fake_platform.works))

    def test_packed_bytes_preserves_nested_dataloader_tensors(self) -> None:
        """Default online transport should accept a normal nested tensor batch."""
        payload = {
            "input_ids": _FakeTensor([[1, 2], [3, 4]], np.int64),
            "labels": (_FakeTensor([5, 6], np.int32),),
            "metadata": {"source": "megatron"},
        }

        with patch("hyper_parallel.distributed_data.distributor.platform", _FakePlatform()):
            frame = _encode_binary_payload(payload)
            result = _decode_binary_payload(frame)

        self.assertEqual(result["input_ids"].shape, (2, 2))
        self.assertEqual(result["input_ids"].values, (1, 2, 3, 4))
        self.assertEqual(result["labels"][0].values, (5, 6))
        self.assertEqual(result["metadata"], {"source": "megatron"})

    def test_direct_tensor_a2a_preserves_tensor_storage(self) -> None:
        """Direct mode should exchange uniform tensors without Host serialization."""
        plan, topology = self._cross_owner_plan()
        peer_tensor = _FakeTensor([[9.0, 10.0]])
        fake_platform = _FakePlatform(group_ranks=(1, 0), variable_output=peer_tensor)
        global_metadata = self._online_tensor_metadata(
            plan,
            (
                TensorLocalBatchSpec((2,), "float32", 2),
                TensorLocalBatchSpec((2,), "float32", 2),
            ),
        )

        with patch("hyper_parallel.distributed_data.distributor.platform", fake_platform):
            redistributor = TorchTensorLocalBatchRedistributor(group="metadata")
            result = redistributor.redistribute(
                (_FakeTensor([1.0, 2.0]),),
                plan,
                topology,
                global_metadata,
            )

        self.assertEqual(result[1].values, (9.0, 10.0))
        self.assertEqual(fake_platform.variable_splits, ([1, 0], [1, 0]))
        self.assertEqual(fake_platform.all_gather_count, 0)
        self.assertTrue(all(work.waited for work in fake_platform.works))

    def test_direct_tensor_a2a_flattens_variable_shapes_from_global_metadata(self) -> None:
        """Variable tensor shapes should use element splits and reconstruct the target shape."""
        plan, topology = self._cross_owner_plan()
        fake_platform = _FakePlatform(
            group_ranks=(1, 0),
            variable_output=_FakeTensor([9.0, 10.0, 11.0]),
        )
        global_metadata = self._online_tensor_metadata(
            plan,
            (
                TensorLocalBatchSpec((2,), "float32", 2),
                TensorLocalBatchSpec((3,), "float32", 3),
            ),
        )

        with patch("hyper_parallel.distributed_data.distributor.platform", fake_platform):
            redistributor = TorchTensorLocalBatchRedistributor(group="metadata")
            result = redistributor.redistribute(
                (_FakeTensor([1.0, 2.0]),),
                plan,
                topology,
                global_metadata,
            )

        self.assertEqual(result[1].shape, (3,))
        self.assertEqual(result[1].values, (9.0, 10.0, 11.0))
        self.assertEqual(fake_platform.variable_splits, ([2, 0], [3, 0]))
        self.assertEqual(fake_platform.all_gather_count, 0)
        self.assertTrue(all(work.waited for work in fake_platform.works))

    def test_direct_tensor_a2a_rejects_mixed_global_dtypes_before_collective(self) -> None:
        """One A2AV buffer cannot represent multiple element dtypes."""
        plan, topology = self._cross_owner_plan()
        fake_platform = _FakePlatform(group_ranks=(1, 0))
        global_metadata = self._online_tensor_metadata(
            plan,
            (
                TensorLocalBatchSpec((2,), "float32", 2),
                TensorLocalBatchSpec((3,), "float16", 3),
            ),
        )

        with patch("hyper_parallel.distributed_data.distributor.platform", fake_platform):
            redistributor = TorchTensorLocalBatchRedistributor(group="metadata")
            with self.assertRaisesRegex(ValueError, "one dtype"):
                redistributor.redistribute(
                    (_FakeTensor([1.0, 2.0]),),
                    plan,
                    topology,
                    global_metadata,
                )

        self.assertIsNone(fake_platform.variable_splits)

    def test_direct_tensor_a2a_rejects_missing_tensor_descriptor_before_collective(self) -> None:
        """A non-tensor sample marker should fail consistently after metadata synchronization."""
        plan, topology = self._cross_owner_plan()
        fake_platform = _FakePlatform(group_ranks=(1, 0))
        planned_by_position = {
            local_batch.source_position: local_batch
            for local_batch in plan.local_batches
        }
        global_metadata = (
            OnlineLocalBatchMetadata(planned_by_position[0].meta),
            OnlineLocalBatchMetadata(
                planned_by_position[1].meta,
                TensorLocalBatchSpec((3,), "float32", 3),
            ),
        )

        with patch("hyper_parallel.distributed_data.distributor.platform", fake_platform):
            redistributor = TorchTensorLocalBatchRedistributor(group="metadata")
            with self.assertRaisesRegex(ValueError, "missing tensor metadata"):
                redistributor.redistribute(
                    (_FakeTensor([1.0, 2.0]),),
                    plan,
                    topology,
                    global_metadata,
                )

        self.assertIsNone(fake_platform.variable_splits)

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
        fake_platform = _FakePlatform()
        micro_batch = {"input_ids": _FakeTensor((1, 2))}

        with patch("hyper_parallel.distributed_data.distributor.platform", fake_platform):
            distributor = TorchModelParallelLocalBatchDistributor(group="consumer")
            received_plan, result = distributor.distribute(micro_batch, plan, topology)

        self.assertEqual(received_plan.plan_id, plan.plan_id)
        self.assertIs(result["input_ids"], micro_batch["input_ids"])
        self.assertEqual(fake_platform.broadcast_count, 1)
