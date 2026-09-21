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
"""UT for :mod:`hyper_parallel.core.distributed_checkpoint.utils`."""
# pylint: disable=wrong-import-position
import importlib
import os
import unittest
from unittest.mock import patch

import torch



import hyper_parallel.core.distributed_checkpoint.utils as utils_mod

importlib.reload(utils_mod)

from hyper_parallel.core.distributed_checkpoint.metadata import (
    CHUNK_INFO,
    ChunkInfo,
    ChunkStorageMetadata,
    MetadataIndex,
    TensorProperties,
)
from hyper_parallel.core.distributed_checkpoint.planner import (
    SavePlan,
    WriteItem,
    WriteItemType,
)
from hyper_parallel.core.distributed_checkpoint.utils import (
    chunk_to_area,
    flatten_state_dict,
    infer_intersection,
    narrow_tensor_by_index,
    plan_ownership_masks,
    set_element,
    traverse_state_dict,
)
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.core.dtensor.placement_types import Shard


class TestUtil(unittest.TestCase):
    """Tests for distributed checkpoint utility helpers."""

    def setUp(self) -> None:
        """Rebuild util against the torch platform before every case."""
        importlib.reload(utils_mod)

    def test_infer_intersection_overlapping_1d(self):
        """
        Feature: infer_intersection for overlapping 1-D ranges.
        Description: Intersect [0, 4) with [2, 6).
        Expectation: Returns ((2, 4),).
        """
        area_a = ((0, 4),)
        area_b = ((2, 6),)
        self.assertEqual(infer_intersection(area_a, area_b), ((2, 4),))

    def test_infer_intersection_disjoint_returns_none(self):
        """
        Feature: infer_intersection for disjoint ranges.
        Description: Intersect [0, 2) with [3, 5).
        Expectation: Returns None when there is no overlap.
        """
        self.assertIsNone(infer_intersection(((0, 2),), ((3, 5),)))

    def test_infer_intersection_dimension_mismatch_raises(self):
        """
        Feature: infer_intersection dimension validation.
        Description: Pass areas with different numbers of dimensions.
        Expectation: ValueError is raised.
        """
        with self.assertRaises(ValueError):
            infer_intersection(((0, 2),), ((0, 2), (0, 2)))

    def test_narrow_tensor_by_index(self):
        """
        Feature: narrow_tensor_by_index slice extraction.
        Description: Narrow a 4x4 torch tensor to rows [1, 3) and cols [0, 2).
        Expectation: Result shape is (2, 2) and values match the source slice.
        """
        tensor = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        sliced = narrow_tensor_by_index(tensor, (1, 0), (2, 2))
        self.assertEqual(tuple(sliced.shape), (2, 2))
        torch.testing.assert_close(sliced, tensor[1:3, 0:2])

    def test_chunk_to_area(self):
        """
        Feature: chunk_to_area converts offsets/sizes to half-open ranges.
        Description: Chunk with offsets (2, 0) and sizes (4, 8).
        Expectation: Area is ((2, 6), (0, 8)).
        """
        chunk = ChunkStorageMetadata(offsets=(2, 0), sizes=(4, 8))
        self.assertEqual(chunk_to_area(chunk), ((2, 6), (0, 8)))

    def test_flatten_state_dict_nested(self):
        """
        Feature: flatten_state_dict dotted FQN keys.
        Description: Flatten a nested dict with model and optimizer subtrees.
        Expectation: Keys use dot notation; mappings preserve object paths.
        """
        nested = {"model": {"weight": torch.zeros(2), "bias": torch.zeros(2)}}
        flat, mappings = flatten_state_dict(nested)
        self.assertEqual(set(flat.keys()), {"model.weight", "model.bias"})
        self.assertEqual(mappings["model.weight"], ("model", "weight"))

    def test_flatten_state_dict_duplicate_fqn_raises(self):
        """
        Feature: flatten_state_dict duplicate key detection.
        Description: Two nested paths that flatten to the same FQN.
        Expectation: ValueError mentions duplicate flattened FQN.
        """
        nested = {"a": {"b.c": 1}, "a.b": {"c": 2}}
        with self.assertRaises(ValueError) as ctx:
            flatten_state_dict(nested)
        self.assertIn("Duplicate flattened FQN", str(ctx.exception))

    def test_set_element_nested_dict_and_list(self):
        """
        Feature: set_element rebuilds nested structure along a path.
        Description: Set values at dict and list paths in an empty root.
        Expectation: Root contains nested dict/list with assigned values.
        """
        root: dict = {}
        set_element(root, ("model", "layers", 0, "weight"), 1)
        set_element(root, ("model", "layers", 1, "weight"), 2)
        self.assertEqual(root["model"]["layers"][0]["weight"], 1)
        self.assertEqual(root["model"]["layers"][1]["weight"], 2)

    def test_set_element_empty_path_raises(self):
        """
        Feature: set_element path validation.
        Description: Call set_element with an empty path tuple.
        Expectation: ValueError is raised.
        """
        with self.assertRaises(ValueError):
            set_element({}, (), None)

    def test_traverse_state_dict_visits_tensor_leaves(self):
        """
        Feature: traverse_state_dict recursive visitor.
        Description: Traverse nested mappings and record tensor leaf paths.
        Expectation: Visitor receives dotted paths for each tensor leaf.
        """
        visited = []
        state = {"a": {"b": torch.zeros(1)}, "c": torch.zeros(1)}
        traverse_state_dict(state, lambda path, _: visited.append(".".join(path)))
        self.assertEqual(set(visited), {"a.b", "c"})

    def test_plan_ownership_masks_keeps_one_copy(self):
        """
        Feature: plan_ownership_masks deduplication.
        Description: Two plans both write the same MetadataIndex.
        Expectation: Exactly one plan is marked as the owner of the WriteItem.
        """
        chunk = ChunkStorageMetadata(offsets=(0,), sizes=(4,))
        props = TensorProperties(dtype="float32")
        index = MetadataIndex(fqn="w")
        item = WriteItem(
            index=index,
            type=WriteItemType.TENSOR,
            tensor_data={"chunk": chunk, "properties": props, "size": (4,)},
        )
        plans = [SavePlan(items=[item]), SavePlan(items=[item])]
        masks = plan_ownership_masks(plans)
        self.assertEqual([len(m) for m in masks], [1, 1])
        self.assertEqual(sum(sum(m) for m in masks), 1)

    def test_create_chunk_list_for_plain_tensor(self):
        """
        Feature: create_chunk_list_for_tensor full-tensor default chunk.
        Description: Plain torch tensor without CHUNK_INFO annotation.
        Expectation: Single chunk covers the full tensor from zero offsets.
        """
        tensor = torch.zeros(3, 5)
        chunks = utils_mod.create_chunk_list_for_tensor(tensor)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].offsets, (0, 0))
        self.assertEqual(chunks[0].sizes, (3, 5))

    def test_create_chunk_list_for_parameter_with_chunk_info(self):
        """
        Feature: create_chunk_list_for_tensor for Parameter with CHUNK_INFO.
        Description: nn.Parameter annotated with ChunkInfo shard metadata.
        Expectation: Returns a single ChunkStorageMetadata matching CHUNK_INFO.chunk.
        """
        chunk = ChunkStorageMetadata(offsets=(0, 4), sizes=(4, 4))
        info = ChunkInfo(chunk=chunk, global_shape=(8, 8))
        param = torch.nn.Parameter(torch.zeros(4, 4))
        object.__setattr__(param, CHUNK_INFO, info)
        chunks = utils_mod.create_chunk_list_for_tensor(param)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].offsets, (0, 4))
        self.assertEqual(chunks[0].sizes, (4, 4))

    def test_create_chunk_list_for_empty_uneven_shard(self):
        """DCP geometry should retain the logical offset of an empty trailing shard."""
        with patch("hyper_parallel.core.dtensor.device_mesh.dist.get_rank", return_value=3):
            mesh = DeviceMesh(
                "cpu",
                [0, 1, 2, 3],
                mesh_dim_names=("fsdp",),
                _init_backend=False,
            )
        layout = Layout.from_device_mesh(mesh)
        layout.set_placements((Shard(0, uneven_shard=True),))
        layout.placement_to_tensor_map(dim=2)
        layout.set_tensor_meta((6, 3), (3, 1), torch.float32)
        tensor = DTensor.from_local_with_layout(torch.empty(0, 3), layout)

        with patch.object(utils_mod.dist, "get_rank", return_value=3):
            chunks = utils_mod.create_chunk_list_for_tensor(tensor)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].offsets, (6, 0))
        self.assertEqual(chunks[0].sizes, (0, 3))

    def test_create_chunk_list_for_tensor_invalid_chunk_info_raises(self):
        """
        Feature: create_chunk_list_for_tensor CHUNK_INFO type check.
        Description: Parameter with CHUNK_INFO set to a non-ChunkInfo object.
        Expectation: ValueError is raised.
        """
        param = torch.nn.Parameter(torch.zeros(2, 2))
        object.__setattr__(param, CHUNK_INFO, "not_chunk_info")
        with self.assertRaises(ValueError) as ctx:
            utils_mod.create_chunk_list_for_tensor(param)
        self.assertIn("ChunkInfo", str(ctx.exception))

    def test_create_chunk_list_for_tensor_unsupported_type_raises(self):
        """
        Feature: create_chunk_list_for_tensor type validation.
        Description: Pass a plain Python int instead of a tensor.
        Expectation: ValueError mentions unsupported type.
        """
        with self.assertRaises(ValueError) as ctx:
            utils_mod.create_chunk_list_for_tensor(42)
        self.assertIn("Not support type", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
