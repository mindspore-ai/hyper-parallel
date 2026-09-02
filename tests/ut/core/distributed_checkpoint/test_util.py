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
"""UT for :mod:`hyper_parallel.core.distributed_checkpoint.util`."""
# pylint: disable=wrong-import-position
import importlib
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import torch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
import hyper_parallel.platform.platform as _platform_mod

_platform_mod.platform = None

import hyper_parallel.core.distributed_checkpoint.util as util_mod

importlib.reload(util_mod)

from hyper_parallel.core.distributed_checkpoint.metadata import (
    BroadcastInfo,
    CHUNK_INFO,
    ChunkInfo,
    ChunkStorageMetadata,
    MetadataIndex,
    TensorProperties,
)
from hyper_parallel.core.distributed_checkpoint.planner import SavePlan, WriteItem, WriteItemType
from hyper_parallel.core.distributed_checkpoint.util import (
    check_path,
    chunk_to_area,
    flatten_state_dict,
    has_valid_filename,
    narrow_tensor_by_index,
    remove_redundant_plans,
    set_element,
    traverse_state_dict,
)
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh, _DEVICE_MESH_MAP
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


class TestUtil(unittest.TestCase):
    """Tests for distributed checkpoint utility helpers."""

    def setUp(self) -> None:
        """Rebuild util against the torch platform before every case."""
        os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
        _platform_mod.platform = None
        importlib.reload(util_mod)

    def test_has_valid_filename(self):
        """
        Feature: has_valid_filename validation rules.
        Description: Check paths with and without valid stem/suffix letters.
        Expectation: Returns True for model.safetensors; False for invalid names.
        """
        self.assertTrue(has_valid_filename(Path("model.safetensors")))
        self.assertFalse(has_valid_filename(Path(".safetensors")))
        self.assertFalse(has_valid_filename(Path("123.456")))

    def test_check_path_creates_parent_for_file(self):
        """
        Feature: check_path creates parent directories for file paths.
        Description: Call check_path with a nested file path that does not exist.
        Expectation: Parent directory is created on disk.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "a" / "b" / "ckpt.bin"
            check_path(nested)
            self.assertTrue(nested.parent.is_dir())

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

    def test_remove_redundant_plans_keeps_one_copy(self):
        """
        Feature: remove_redundant_plans deduplication.
        Description: Two plans both write the same MetadataIndex.
        Expectation: Only one plan retains the WriteItem.
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
        deduped = remove_redundant_plans(plans)
        total_items = sum(len(p.items) for p in deduped)
        self.assertEqual(total_items, 1)

    def test_create_chunk_list_for_plain_tensor(self):
        """
        Feature: create_chunk_list_for_tensor full-tensor default chunk.
        Description: Plain torch tensor without CHUNK_INFO annotation.
        Expectation: Single chunk covers the full tensor from zero offsets.
        """
        tensor = torch.zeros(3, 5)
        chunks = util_mod.create_chunk_list_for_tensor(tensor)
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
        chunks = util_mod.create_chunk_list_for_tensor(param)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].offsets, (0, 4))
        self.assertEqual(chunks[0].sizes, (4, 4))

    def test_create_chunk_list_for_empty_uneven_shard(self):
        """DCP geometry should retain the logical offset of an empty trailing shard."""
        with patch("hyper_parallel.core.dtensor.device_mesh.platform.get_rank", return_value=3):
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

        with patch.object(util_mod.platform, "get_rank", return_value=3):
            chunks = util_mod.create_chunk_list_for_tensor(tensor)

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
            util_mod.create_chunk_list_for_tensor(param)
        self.assertIn("ChunkInfo", str(ctx.exception))

    def test_create_chunk_list_for_tensor_unsupported_type_raises(self):
        """
        Feature: create_chunk_list_for_tensor type validation.
        Description: Pass a plain Python int instead of a tensor.
        Expectation: ValueError mentions unsupported type.
        """
        with self.assertRaises(ValueError) as ctx:
            util_mod.create_chunk_list_for_tensor(42)
        self.assertIn("Not support type", str(ctx.exception))


class _RecordingPlatform:
    """Platform double recording the collectives ``util`` issues.

    ``broadcast``, ``new_group`` and ``destroy_process_group`` are the calls under test;
    ``all_gather_object`` stands in for the peer ranks, each of which reports
    ``peer_missing_groups`` as the groups it still needs.
    """

    def __init__(self, world_size: int = 2, peer_missing_groups: tuple = ()) -> None:
        """Build a double for a world of ``world_size`` ranks, this one being rank 0."""
        self.world_size = world_size
        self.peer_missing_groups = peer_missing_groups
        self.broadcasts: list = []
        self.created_groups: list = []
        self.destroyed_groups: list = []
        self.gathered: list = []

    def get_rank(self) -> int:
        """Return the rank the timing decorator logs."""
        return 0

    def get_world_size(self) -> int:
        """Return the world size that sizes the all-gather buffer."""
        return self.world_size

    def broadcast(self, tensor: Any, src_rank: int, group: Any) -> None:
        """Record one broadcast instead of reaching a backend."""
        self.broadcasts.append((tensor, src_rank, group))

    @staticmethod
    def detach(tensor: Any) -> Any:
        """Mirror the active Torch platform's data-only view."""
        return tensor.detach()

    def new_group(self, group_ranks: tuple) -> str:
        """Record one group creation and return a recognizable handle."""
        self.created_groups.append(group_ranks)
        return f"group{group_ranks}"

    def destroy_process_group(self, group: Any) -> None:
        """Record one group release instead of reaching a backend."""
        self.destroyed_groups.append(group)

    # pylint: disable=W0613
    def all_gather_object(self, object_list: list, obj: Any, group: Any = None) -> None:
        """Report ``obj`` for this rank and ``peer_missing_groups`` for every other one."""
        self.gathered.append(obj)
        object_list[0] = obj
        for index in range(1, len(object_list)):
            object_list[index] = self.peer_missing_groups


class TestBroadcastLoadedTensors(unittest.TestCase):
    """Tests for the broadcast path taken when only the minimum rank reads a shard."""

    def setUp(self) -> None:
        """Rebuild util against the torch platform before every case."""
        os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
        _platform_mod.platform = None
        importlib.reload(util_mod)

    @staticmethod
    def _dtensor(local: torch.Tensor) -> DTensor:
        """Build a rank-zero sharded DTensor without initializing a backend."""
        _DEVICE_MESH_MAP.clear()
        EXISTING_COMM_GROUPS.clear()
        with patch(
                "hyper_parallel.core.dtensor.device_mesh.platform.get_rank",
                return_value=0,
        ):
            mesh = Layout((2,), ("dp",), init_backend=False).mesh
            return DTensor.from_local(local, mesh, (Shard(0),))

    @staticmethod
    def _tag(obj: Any, group_ranks: tuple, src_rank: int) -> Any:
        """Attach BROADCAST_INFO the way StandardLoadPlanner.should_load_shard does."""
        setattr(obj, util_mod.BROADCAST_INFO, BroadcastInfo(group_ranks, src_rank))
        return obj

    def test_existing_group_broadcasts_local_shard_of_dtensor(self):
        """
        Feature: _broadcast_within_existing_groups DTensor path.
        Description: One tagged DTensor whose group the caller pre-built.
        Expectation: One broadcast, with the tagged src rank, of a buffer that still aliases
            the local shard so the in-place broadcast lands in the state dict entry.
        """
        fake = _RecordingPlatform()
        dtensor = self._tag(self._dtensor(torch.zeros(4)), (0, 1), 0)

        with patch.object(util_mod, "platform", fake):
            missing = util_mod._broadcast_within_existing_groups(
                {"w": dtensor}, {(0, 1): "pre_built"}
            )

        self.assertEqual(missing, {})
        self.assertEqual(len(fake.broadcasts), 1)
        sent, src_rank, group = fake.broadcasts[0]
        self.assertEqual((src_rank, group), (0, "pre_built"))
        self.assertEqual(sent.data_ptr(), dtensor.to_local().data_ptr())

    def test_existing_group_broadcasts_plain_tensor(self):
        """
        Feature: _broadcast_within_existing_groups plain tensor path.
        Description: A plain nn.Parameter is tagged too - should_load_shard tags every entry
            carrying CHUNK_INFO, not only DTensors - so the broadcast must not call to_local().
        Expectation: The parameter is broadcast detached, and the buffer still aliases it.
        """
        fake = _RecordingPlatform()
        param = self._tag(torch.nn.Parameter(torch.zeros(4)), (0, 1), 1)

        with patch.object(util_mod, "platform", fake):
            missing = util_mod._broadcast_within_existing_groups(
                {"w": param}, {(0, 1): "pre_built"}
            )

        self.assertEqual(missing, {})
        self.assertEqual(len(fake.broadcasts), 1)
        sent, src_rank, _ = fake.broadcasts[0]
        self.assertEqual(src_rank, 1)
        self.assertFalse(sent.requires_grad)
        self.assertEqual(sent.data_ptr(), param.data_ptr())

    def test_entries_without_broadcast_info_are_skipped(self):
        """
        Feature: broadcast_loaded_tensors entry filtering.
        Description: State dict mixes an untagged tensor and a non-tensor value.
        Expectation: Nothing is broadcast and no group is requested.
        """
        fake = _RecordingPlatform()

        with patch.object(util_mod, "platform", fake):
            util_mod.broadcast_loaded_tensors(
                {"w": torch.zeros(2), "step": 7}, {(0, 1): "pre_built"}
            )

        self.assertEqual(fake.broadcasts, [])
        self.assertEqual(fake.created_groups, [])

    def test_invalid_broadcast_info_type_raises(self):
        """
        Feature: broadcast info type validation.
        Description: An entry carries a string instead of a BroadcastInfo.
        Expectation: ValueError names the expected type and nothing is broadcast.
        """
        fake = _RecordingPlatform()
        tensor = torch.zeros(2)
        setattr(tensor, util_mod.BROADCAST_INFO, "not_broadcast_info")

        with patch.object(util_mod, "platform", fake):
            with self.assertRaises(ValueError) as ctx:
                util_mod.broadcast_loaded_tensors({"w": tensor}, {})

        self.assertIn("BroadcastInfo", str(ctx.exception))
        self.assertEqual(fake.broadcasts, [])

    def test_entries_without_a_pre_built_group_are_collected(self):
        """
        Feature: _broadcast_within_existing_groups missing-group bookkeeping.
        Description: Two entries need group (0, 1) but the caller only pre-built (2, 3).
        Expectation: Both entries come back keyed by (0, 1), and nothing is broadcast yet.
        """
        fake = _RecordingPlatform()
        first = self._tag(torch.zeros(2), (0, 1), 0)
        second = self._tag(torch.zeros(2), (0, 1), 0)

        with patch.object(util_mod, "platform", fake):
            missing = util_mod._broadcast_within_existing_groups(
                {"a": first, "b": second}, {(2, 3): "other"}
            )

        self.assertEqual(list(missing), [(0, 1)])
        self.assertEqual(len(missing[(0, 1)]), 2)
        self.assertIs(missing[(0, 1)][0], first)
        self.assertIs(missing[(0, 1)][1], second)
        self.assertEqual(fake.broadcasts, [])

    def test_create_groups_and_broadcast_covers_the_groups_of_every_rank(self):
        """
        Feature: _create_groups_and_broadcast collective group creation.
        Description: This rank misses group (0, 1) while a peer reports missing (2, 3).
        Expectation: Both groups are created - creating one is collective, so a rank takes part
            for groups it does not use itself - and the local entry is broadcast in group (0, 1).
        """
        fake = _RecordingPlatform(world_size=2, peer_missing_groups=((2, 3),))
        tensor = self._tag(torch.zeros(2), (0, 1), 0)

        with patch.object(util_mod, "platform", fake):
            util_mod._create_groups_and_broadcast({(0, 1): [tensor]})

        self.assertEqual(set(fake.created_groups), {(0, 1), (2, 3)})
        self.assertEqual(len(fake.broadcasts), 1)
        sent, src_rank, group = fake.broadcasts[0]
        self.assertEqual((src_rank, group), (0, "group(0, 1)"))
        self.assertEqual(sent.data_ptr(), tensor.data_ptr())

    def test_broadcast_loaded_tensors_creates_groups_when_none_given(self):
        """
        Feature: broadcast_loaded_tensors default group handling.
        Description: Call it without the optional groups argument.
        Expectation: The group is created on demand and the entry broadcast through it.
        """
        fake = _RecordingPlatform()
        tensor = self._tag(torch.zeros(2), (0, 1), 1)

        with patch.object(util_mod, "platform", fake):
            util_mod.broadcast_loaded_tensors({"w": tensor})

        self.assertEqual(fake.created_groups, [(0, 1)])
        self.assertEqual(len(fake.broadcasts), 1)
        self.assertEqual(fake.broadcasts[0][1:], (1, "group(0, 1)"))

    def test_broadcast_loaded_tensors_skips_group_creation_when_all_pre_built(self):
        """
        Feature: broadcast_loaded_tensors fast path.
        Description: Every tagged entry finds its group in the caller-supplied dict.
        Expectation: One broadcast, and neither an all-gather nor a group creation.
        """
        fake = _RecordingPlatform()
        tensor = self._tag(torch.zeros(2), (0, 1), 0)

        with patch.object(util_mod, "platform", fake):
            util_mod.broadcast_loaded_tensors({"w": tensor}, {(0, 1): "pre_built"})

        self.assertEqual(len(fake.broadcasts), 1)
        self.assertEqual(fake.created_groups, [])
        self.assertEqual(fake.gathered, [])

    def test_broadcast_consumes_the_mark_on_both_paths(self):
        """
        Feature: BROADCAST_INFO lifecycle.
        Description: Broadcast one entry whose group the caller pre-built and one whose group
            the load has to create on demand.
        Expectation: Both are broadcast and neither is still marked afterwards, so the next
            load plans over a state dict no stale source rank can be read from.
        """
        fake = _RecordingPlatform()
        pre_built = self._tag(torch.zeros(2), (0, 1), 0)
        on_demand = self._tag(torch.zeros(2), (0, 2), 0)

        with patch.object(util_mod, "platform", fake):
            util_mod.broadcast_loaded_tensors(
                {"pre_built": pre_built, "on_demand": on_demand}, {(0, 1): "pre_built"}
            )

        self.assertEqual(len(fake.broadcasts), 2)
        self.assertFalse(hasattr(pre_built, util_mod.BROADCAST_INFO))
        self.assertFalse(hasattr(on_demand, util_mod.BROADCAST_INFO))


    def test_on_demand_groups_are_released_after_the_broadcast(self):
        """
        Feature: _create_groups_and_broadcast group lifetime.
        Description: This rank needs group (0, 1) and takes part in creating a peer's (2, 3).
        Expectation: Only (0, 1) is released - the rank is not a member of (2, 3), so new_group
            handed it a non-member placeholder rather than a group to destroy. Keeping the
            group instead would leak one communicator set per load, and a loop that resumes
            repeatedly would run the backend out of them.
        """
        fake = _RecordingPlatform(world_size=2, peer_missing_groups=((2, 3),))
        tensor = self._tag(torch.zeros(2), (0, 1), 0)

        with patch.object(util_mod, "platform", fake):
            util_mod._create_groups_and_broadcast({(0, 1): [tensor]})

        self.assertEqual(fake.destroyed_groups, ["group(0, 1)"])

    def test_a_group_that_refuses_to_be_released_does_not_fail_the_load(self):
        """
        Feature: _destroy_groups best effort release.
        Description: The backend raises while releasing the group.
        Expectation: broadcast_loaded_tensors still returns. The tensors are already broadcast
            by then, so a leaked group is a warning, not a reason to fail the load.
        """
        fake = _RecordingPlatform()
        fake.destroy_process_group = Mock(side_effect=RuntimeError("backend is grumpy"))
        tensor = self._tag(torch.zeros(2), (0, 1), 0)

        with patch.object(util_mod, "platform", fake):
            util_mod.broadcast_loaded_tensors({"w": tensor})

        self.assertEqual(len(fake.broadcasts), 1)
        self.assertFalse(hasattr(tensor, util_mod.BROADCAST_INFO))

    def test_pre_built_groups_are_left_to_their_owner(self):
        """
        Feature: broadcast_loaded_tensors group ownership.
        Description: Every entry finds its group in the caller-supplied dict.
        Expectation: Nothing is released. Those groups belong to the caller, which reuses them
            across loads; destroying them here would break the next one.
        """
        fake = _RecordingPlatform()
        tensor = self._tag(torch.zeros(2), (0, 1), 0)

        with patch.object(util_mod, "platform", fake):
            util_mod.broadcast_loaded_tensors({"w": tensor}, {(0, 1): "pre_built"})

        self.assertEqual(fake.destroyed_groups, [])


if __name__ == "__main__":
    unittest.main()
