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
"""UT for :mod:`hyper_parallel.core.distributed_checkpoint.standard_planner`."""
# pylint: disable=wrong-import-position
import importlib
import os
import pickle
import unittest
from unittest.mock import patch

import torch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
import hyper_parallel.platform.platform as _platform_mod

_platform_mod.platform = None

import hyper_parallel.core.distributed_checkpoint.standard_planner as planner_mod

importlib.reload(planner_mod)

from hyper_parallel.core.distributed_checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    TensorProperties,
    TensorStorageMetadata,
)
from hyper_parallel.core.distributed_checkpoint.planner import SavePlan, WriteItem, WriteItemType
from hyper_parallel.core.distributed_checkpoint.standard_planner import (
    StandardLoadPlanner,
    StandardSavePlanner,
    create_read_items_for_chunk_list,
)
from hyper_parallel.core.dtensor.device_mesh import _DEVICE_MESH_MAP
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.core.dtensor.placement_types import RaggedShard
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


class TestStandardPlanner(unittest.TestCase):
    """Tests for StandardSavePlanner and StandardLoadPlanner."""

    def setUp(self) -> None:
        os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
        _platform_mod.platform = None
        importlib.reload(planner_mod)
        StandardSavePlanner._cached_save_result.clear()

    @staticmethod
    def _ragged_tensor(local, local_units=(1, 3)):
        """Build a rank-zero RaggedShard DTensor without initializing a backend."""
        _DEVICE_MESH_MAP.clear()
        EXISTING_COMM_GROUPS.clear()
        with patch(
                "hyper_parallel.core.dtensor.device_mesh.platform.get_rank",
                return_value=0,
        ):
            mesh = Layout((2,), ("ragged",), init_backend=False).mesh
            return DTensor.from_local(
                local,
                mesh,
                (RaggedShard(dims=(0, 1), local_units=local_units),),
                shape=(6, 4, 8),
            )

    def test_save_planner_build_local_plan_for_tensors_and_bytes(self):
        """
        Feature: StandardSavePlanner.build_local_plan.
        Description: Configure planner with one torch tensor and one pickle-able object.
        Expectation: Plan contains one TENSOR WriteItem and one BYTE_IO WriteItem.
        """
        weight = torch.nn.Parameter(torch.randn(4, 8))
        state = {"weight": weight, "step": 42}
        planner = StandardSavePlanner(enable_plan_caching=False)
        planner.configure_planner(state, rank=0, use_collectives=False)
        plan = planner.build_local_plan()
        types = {item.type for item in plan.items}
        self.assertEqual(types, {WriteItemType.TENSOR, WriteItemType.BYTE_IO})
        tensor_items = [i for i in plan.items if i.type == WriteItemType.TENSOR]
        self.assertEqual(tensor_items[0].index.fqn, "weight")

    def test_save_planner_build_global_plan_assigns_chunk_indices(self):
        """
        Feature: StandardSavePlanner.build_global_plan.
        Description: Merge two local plans writing distinct tensor FQNs.
        Expectation: Metadata lists both tensors; chunk indices are assigned per FQN.
        """
        chunk = ChunkStorageMetadata(offsets=(0, 0), sizes=(2, 2))
        props = TensorProperties(dtype="float32")
        item_w = WriteItem(
            index=MetadataIndex(fqn="w"),
            type=WriteItemType.TENSOR,
            tensor_data={"chunk": chunk, "properties": props, "size": (2, 2)},
        )
        item_b = WriteItem(
            index=MetadataIndex(fqn="b"),
            type=WriteItemType.TENSOR,
            tensor_data={"chunk": chunk, "properties": props, "size": (2, 2)},
        )
        local_plans = [
            SavePlan(items=[item_w], planner_data={}),
            SavePlan(items=[item_b], planner_data={}),
        ]
        planner = StandardSavePlanner(enable_plan_caching=False, remove_redundancy=False)
        planner.configure_planner({"w": torch.zeros(2, 2), "b": torch.zeros(2, 2)}, use_collectives=False)
        global_plans, metadata = planner.build_global_plan(local_plans)
        self.assertEqual(len(global_plans), 2)
        self.assertIn("w", metadata.state_dict_metadata)
        self.assertIn("b", metadata.state_dict_metadata)
        w_md = metadata.state_dict_metadata["w"]
        self.assertIsInstance(w_md, TensorStorageMetadata)
        self.assertEqual(len(w_md.chunks), 1)

    def test_save_planner_get_data_returns_detached_cpu_tensor(self):
        """
        Feature: StandardSavePlanner.get_data for tensor items.
        Description: Resolve runtime tensor data for a TENSOR WriteItem.
        Expectation: Returned tensor is detached, on CPU, and numerically equal to source.
        """
        weight = torch.nn.Parameter(torch.ones(2, 2) * 5.0)
        planner = StandardSavePlanner(enable_plan_caching=False)
        planner.configure_planner({"weight": weight}, use_collectives=False)
        plan = planner.build_local_plan()
        tensor_item = next(i for i in plan.items if i.type == WriteItemType.TENSOR)
        data = planner.get_data(tensor_item)
        self.assertFalse(data.requires_grad)
        torch.testing.assert_close(data, weight.detach().cpu())

    def test_ragged_save_plan_emits_one_item_per_nd_box(self):
        """A flat RaggedShard interval is saved as ordered standard N-D chunks."""
        tensor = self._ragged_tensor(torch.arange(48))
        planner = StandardSavePlanner(enable_plan_caching=True)
        planner.configure_planner({"weight": tensor}, rank=0)

        plan = planner.build_local_plan()

        self.assertFalse(planner._enable_plan_caching)
        self.assertEqual(
            [item.index.offset for item in plan.items],
            [(0, 0, 0), (1, 0, 0)],
        )
        self.assertEqual(
            [tuple(planner.get_data(item).shape) for item in plan.items],
            [(1, 4, 8), (1, 2, 8)],
        )

    def test_save_planner_plan_cache_hit(self):
        """
        Feature: StandardSavePlanner plan caching.
        Description: cache_result then get_cached with same state_dict keys.
        Expectation: get_cached returns the stored CachedSaveResult.
        """
        planner = StandardSavePlanner(enable_plan_caching=True)
        planner.configure_planner({"w": torch.zeros(1)}, rank=0, use_collectives=True)
        plan = SavePlan(items=[])
        metadata = Metadata(state_dict_metadata={})
        planner.cache_result(plan, metadata)
        cached = planner.get_cached()
        self.assertIsNotNone(cached)
        self.assertIs(cached.final_plan, plan)
        self.assertIs(cached.metadata, metadata)

    def test_create_read_items_for_chunk_list_overlap(self):
        """
        Feature: create_read_items_for_chunk_list resharding overlap.
        Description: Local chunk is half of a saved full tensor chunk.
        Expectation: One ReadItem copies the overlapping region with correct offsets.
        """
        checkpoint_md = TensorStorageMetadata(
            properties=TensorProperties(dtype="float32"),
            size=(4, 4),
            chunks=[ChunkStorageMetadata(offsets=(0, 0), sizes=(4, 4))],
        )
        local_chunks = [ChunkStorageMetadata(offsets=(0, 0), sizes=(2, 4))]
        read_items = create_read_items_for_chunk_list("w", checkpoint_md, local_chunks)
        self.assertEqual(len(read_items), 1)
        self.assertEqual(read_items[0].lengths, (2, 4))

    def test_load_planner_build_local_plan_and_apply_bytes(self):
        """
        Feature: StandardLoadPlanner byte IO path.
        Description: Load planner configured with BYTE_IO metadata entry.
        Expectation: Local plan has BYTE_IO ReadItem; apply_bytes restores Python object.
        """
        from hyper_parallel.core.distributed_checkpoint.metadata import BytesStorageMetadata

        payload = {"lr": 0.01}
        state = {"opt_state": None}
        metadata = Metadata(state_dict_metadata={"opt_state": BytesStorageMetadata()})
        planner = StandardLoadPlanner()
        planner.configure_planner(state, metadata, use_collectives=False)
        plan = planner.build_local_plan()
        self.assertEqual(len(plan.items), 1)
        read_item = plan.items[0]
        planner.apply_bytes(read_item, pickle.dumps(payload))
        self.assertEqual(state["opt_state"], payload)

    def test_ragged_load_plan_reshards_saved_nd_chunks_into_flat_storage(self):
        """Load source Ragged boxes into a target with different local units."""
        target = self._ragged_tensor(torch.zeros(144, dtype=torch.int64), (3, 1))
        saved_chunks = [
            ChunkStorageMetadata((0, 0, 0), (1, 4, 8)),
            ChunkStorageMetadata((1, 0, 0), (1, 2, 8)),
            ChunkStorageMetadata((1, 2, 0), (1, 2, 8)),
            ChunkStorageMetadata((2, 0, 0), (4, 4, 8)),
        ]
        metadata = Metadata(
            state_dict_metadata={
                "weight": TensorStorageMetadata(
                    properties=TensorProperties(dtype="torch.int64"),
                    size=(6, 4, 8),
                    chunks=saved_chunks,
                )
            }
        )
        planner = StandardLoadPlanner()
        planner.configure_planner({"weight": target}, metadata, rank=0)
        global_tensor = torch.arange(192).reshape(6, 4, 8)

        with patch(
                "hyper_parallel.core.distributed_checkpoint.standard_planner.get_platform"
        ) as mock_platform:
            mock_platform.return_value.get_rank.return_value = 0
            read_items = planner.build_local_plan().items

        for item in read_items:
            storage_chunk = saved_chunks[item.storage_index.index]
            global_offsets = tuple(
                base + relative
                for base, relative in zip(storage_chunk.offsets, item.storage_offsets)
            )
            source_slices = tuple(
                slice(offset, offset + length)
                for offset, length in zip(global_offsets, item.lengths)
            )
            planner.acquire_tensor(item).copy_(global_tensor[source_slices])

        torch.testing.assert_close(target.to_local(), torch.arange(144))


if __name__ == "__main__":
    unittest.main()
