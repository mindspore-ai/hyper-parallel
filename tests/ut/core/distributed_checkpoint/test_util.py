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
from collections import deque
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
    CHUNK_INFO,
    ChunkInfo,
    ChunkStorageMetadata,
    MetadataIndex,
    TensorProperties,
)
from hyper_parallel.core.distributed_checkpoint.planner import (
    BroadcastSource,
    LoadItemType,
    ReadItem,
    SavePlan,
    WriteItem,
    WriteItemType,
)
from hyper_parallel.core.distributed_checkpoint.util import (
    check_path,
    chunk_to_area,
    flatten_state_dict,
    has_valid_filename,
    narrow_tensor_by_index,
    plan_ownership_masks,
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


class _RecordingHandle:
    """Work handle double, so a test can see when a broadcast was waited on."""

    def __init__(self, fake: "_RecordingPlatform", buffer: Any = None) -> None:
        """Register with the platform double that issued this broadcast."""
        self.fake = fake
        self.buffer = buffer
        self.waited = False
        fake.handles.append(self)

    def wait(self) -> None:
        """Mark this broadcast finished, standing in for what a sender would have sent."""
        self.waited = True
        self.fake.in_flight -= 1
        if self.fake.fill is not None and self.buffer is not None:
            self.buffer.fill_(self.fake.fill)


class _RecordingPlatform:
    """Platform double recording the collectives ``util`` issues.

    ``broadcast`` and ``create_group`` are the calls under test; ``all_gather_object``
    stands in for the peer ranks, each of which reports ``peer_missing_groups`` as the
    groups it still needs.
    """

    def __init__(self, world_size: int = 2, peer_missing_groups: tuple = (),
                 rank: int = 0, fill: float = None, cached_groups: dict = None) -> None:
        """Build a double for a world of ``world_size`` ranks, this one being ``rank``.

        ``fill`` stands in for what a broadcast brings back: the buffer it was given is
        filled with that value once the broadcast has been waited on, and not before, so
        a caller that read a buffer too early sees what was in it rather than what came.
        """
        self.world_size = world_size
        self.peer_missing_groups = peer_missing_groups
        self.cached_groups = dict(cached_groups or {})
        self.rank = rank
        self.fill = fill
        self.broadcasts: list = []
        self.created_groups: list = []
        self.destroyed_groups: list = []
        self.gathered: list = []
        # Ordered log of the calls whose relative order matters: a group must be drained
        # before it is released, or the broadcast still queued on it is torn down with it.
        self.events: list = []
        self.handles: list = []
        self.in_flight: int = 0
        self.peak_in_flight: int = 0

    def get_rank(self) -> int:
        """Return the rank the timing decorator logs and the batcher sends from."""
        return self.rank

    def get_world_size(self) -> int:
        """Return the world size that sizes the all-gather buffer."""
        return self.world_size

    def broadcast_async(self, tensor: Any, src_rank: int, group: Any) -> "_RecordingHandle":
        """Record one broadcast, count it as going, and hand back a handle to wait on."""
        self.broadcasts.append((tensor, src_rank, group))
        # Only a rank receiving has anything written into its buffer; the sender's is
        # read from and left as it was.
        handle = _RecordingHandle(self, None if src_rank == self.rank else tensor)
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        return handle

    @staticmethod
    def detach(tensor: Any) -> Any:
        """Mirror the active Torch platform's data-only view."""
        return tensor.detach()

    def new_group(self, group_ranks: tuple) -> str:
        """Record one group creation and return a recognizable handle."""
        self.created_groups.append(group_ranks)
        return f"group{group_ranks}"

    def synchronize(self) -> None:
        """Record one device drain instead of reaching a backend."""
        self.events.append("synchronize")

    def create_group(self, group_ranks: tuple) -> str:
        """The cached, template-expanding creator, which a load must not reach for."""
        raise AssertionError(
            f"create_group{group_ranks} was called: a load's groups live for one read "
            f"and must be made with new_group, which neither caches nor expands them"
        )

    def destroy_process_group(self, group: Any = None) -> None:
        """Record one group being torn down."""
        self.destroyed_groups.append(group)
        self.events.append(f"destroy {group}")

    def get_created_group(self, group_ranks: tuple) -> Any:
        """What the process-wide cache already holds for these ranks, if anything."""
        return self.cached_groups.get(tuple(group_ranks))

    @staticmethod
    def get_world_group() -> str:
        """The group of every rank, which is there from the start and outlives a load."""
        return "world_group"

    # pylint: disable=W0613
    def all_gather_object(self, object_list: list, obj: Any, group: Any = None) -> None:
        """Report ``obj`` for this rank, and for every other one a peer that needs
        ``peer_missing_groups`` and has none of them."""
        self.gathered.append(obj)
        object_list[0] = obj
        for index in range(1, len(object_list)):
            object_list[index] = (self.peer_missing_groups, self.peer_missing_groups)

    @staticmethod
    def new_tensor(size: tuple, dtype: Any = None, device: Any = None) -> torch.Tensor:
        """The staging buffer a batch is gathered into, filled with a value no shard
        carries so that a byte of it left unwritten shows up rather than passing."""
        return torch.full(size, -99.0, dtype=dtype, device=device)

    @staticmethod
    def copy_each(dests: list, srcs: list) -> None:
        """Copy every pair, as the platform does in one fused operation."""
        torch._foreach_copy_(dests, srcs)  # pylint: disable=protected-access

    @staticmethod
    def get_tensor_storage_size(tensor: Any) -> int:
        """How many bytes a shard takes, which is what decides whether it is batched."""
        return int(tensor.numel()) * int(tensor.element_size())


class TestBroadcastShard(unittest.TestCase):
    """Tests for sending on one shard that a rank read on behalf of a group."""

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
    def _item(fqn: str, source: Any = None, offset: tuple = (0,), index: int = 0) -> ReadItem:
        """A read item for one shard of ``fqn``, marked as read by ``source`` when given."""
        return ReadItem(
            type=LoadItemType.TENSOR,
            dest_index=MetadataIndex(fqn=fqn, offset=offset, index=index),
            dest_offsets=(0,),
            storage_index=MetadataIndex(fqn=fqn, offset=(0,), index=0),
            storage_offsets=(0,),
            lengths=(2,),
            source=source,
        )

    def test_a_marked_dtensor_is_sent_whole_from_its_source(self):
        """
        Feature: broadcast_shard DTensor path.
        Description: One marked shard whose group the caller pre-built.
        Expectation: One broadcast from the marked source, of a buffer still aliasing the
            local shard so that the in-place broadcast lands in the state dict entry rather
            than in a copy of it.
        """
        fake = _RecordingPlatform()
        dtensor = self._dtensor(torch.zeros(2))
        source = BroadcastSource(group_ranks=(0, 1), src_rank=1)
        in_flight = deque()

        with patch.object(util_mod, "platform", fake):
            util_mod.broadcast_shard(
                in_flight, {"w": dtensor}, self._item("w", source), {(0, 1): "pre_built"}
            )

        self.assertEqual(len(fake.broadcasts), 1)
        sent, src_rank, group = fake.broadcasts[0]
        self.assertEqual((src_rank, group), (1, "pre_built"))
        self.assertFalse(sent.requires_grad)
        self.assertEqual(sent.data_ptr(), dtensor.to_local().data_ptr())

    def test_a_shard_whose_group_was_not_built_is_a_loud_failure(self):
        """
        Feature: broadcast_shard against groups that do not cover the plan.
        Description: A marked shard reaching the send with its group missing, which is what a
            caller that skipped ensure_broadcast_groups would produce.
        Expectation: KeyError. Creating the group here instead would be a collective in the
            middle of a pipeline of them, entered by only the ranks that happened to be
            short one, so failing outright beats hanging the ranks that were not.
        """
        fake = _RecordingPlatform()
        source = BroadcastSource(group_ranks=(0, 1), src_rank=0)

        with patch.object(util_mod, "platform", fake):
            with self.assertRaises(KeyError):
                util_mod.broadcast_shard(deque(), {"w": torch.zeros(2)}, self._item("w", source), {})

        self.assertEqual(fake.broadcasts, [])

    def test_broadcasts_overlap_up_to_the_in_flight_limit(self):
        """
        Feature: broadcast_shard overlap.
        Description: Four times as many shards as the in-flight limit allows, sent one after
            another through a pre-built group.
        Expectation: Several are going at once, which is what lets the next read run over the
            send before it, but never more than the limit -- every one still going holds
            resources inside the backend.
        """
        limit = util_mod._MAX_BROADCASTS_IN_FLIGHT
        fake = _RecordingPlatform()
        source = BroadcastSource(group_ranks=(0, 1), src_rank=0)
        state = {f"w{i:02d}": torch.zeros(2) for i in range(4 * limit)}
        in_flight = deque()

        with patch.object(util_mod, "platform", fake):
            for name in state:
                util_mod.broadcast_shard(in_flight, state, self._item(name, source), {(0, 1): "g"})

        self.assertEqual(len(fake.broadcasts), 4 * limit)
        self.assertEqual(fake.peak_in_flight, limit)

    def test_waiting_leaves_nothing_going(self):
        """
        Feature: wait_broadcasts.
        Description: Three sends started, then waited on.
        Expectation: Every handle is waited and the queue is empty. The buffers are the state
            dict tensors themselves, so a load that carried on with one still in flight would
            be reading into memory a collective is still writing.
        """
        fake = _RecordingPlatform()
        source = BroadcastSource(group_ranks=(0, 1), src_rank=0)
        state = {name: torch.zeros(2) for name in ("a", "b", "c")}
        in_flight = deque()

        with patch.object(util_mod, "platform", fake):
            for name in state:
                util_mod.broadcast_shard(in_flight, state, self._item(name, source), {(0, 1): "g"})
            util_mod.wait_broadcasts(in_flight)

        self.assertEqual(len(fake.handles), 3)
        self.assertTrue(all(handle.waited for handle in fake.handles))
        self.assertEqual((fake.in_flight, len(in_flight)), (0, 0))


class TestEnsureBroadcastGroups(unittest.TestCase):
    """Tests for building the communication groups a broadcasting load sends through."""

    def setUp(self) -> None:
        """Rebuild util against the torch platform before every case."""
        os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
        _platform_mod.platform = None
        importlib.reload(util_mod)

    @staticmethod
    def _item(fqn: str, source: Any = None) -> ReadItem:
        """A read item for ``fqn``, marked as read by ``source`` when one is given."""
        return ReadItem(
            type=LoadItemType.TENSOR,
            dest_index=MetadataIndex(fqn=fqn, offset=(0,), index=0),
            dest_offsets=(0,),
            storage_index=MetadataIndex(fqn=fqn, offset=(0,), index=0),
            storage_offsets=(0,),
            lengths=(2,),
            source=source,
        )

    def test_a_group_the_caller_did_not_build_is_created(self):
        """
        Feature: ensure_broadcast_groups group creation.
        Description: A marked shard whose group the caller did not pre-build, with a peer
            rank reporting a group of its own.
        Expectation: The missing rank tuples are all-gathered and every rank creates the
            whole set, not only the group it needs, since creating one is itself collective.
            Only the one this rank asked for comes back to it.
        """
        fake = _RecordingPlatform(world_size=4, peer_missing_groups=((2, 3),))
        source = BroadcastSource(group_ranks=(0, 1), src_rank=0)

        with patch.object(util_mod, "platform", fake):
            groups = util_mod.ensure_broadcast_groups([self._item("w", source)])

        self.assertEqual(fake.created_groups, [(0, 1), (2, 3)])
        self.assertEqual(groups, {(0, 1): "group(0, 1)"})

    def test_groups_the_caller_built_are_kept_as_they_are(self):
        """
        Feature: ensure_broadcast_groups with everything already in hand.
        Description: Every group the plan needs was supplied by the caller.
        Expectation: They come back untouched and none is created, which is the point of
            pre-building them.
        """
        fake = _RecordingPlatform(world_size=4)
        source = BroadcastSource(group_ranks=(0, 1), src_rank=0)

        with patch.object(util_mod, "platform", fake):
            groups = util_mod.ensure_broadcast_groups([self._item("w", source)], {(0, 1): "pre_built"})

        self.assertEqual(groups, {(0, 1): "pre_built"})
        self.assertEqual(fake.created_groups, [])

    def test_a_rank_with_nothing_to_send_still_joins_the_all_gather(self):
        """
        Feature: ensure_broadcast_groups collective discipline.
        Description: A rank whose plan marked nothing, so it needs no group of its own, while
            a peer reports one it is short of.
        Expectation: It all-gathers and creates all the same. Whether a rank holds a shared
            shard is its own business, and a rank that stayed out on those grounds would hang
            the ranks that did not -- both calls in here are collective.
        """
        fake = _RecordingPlatform(world_size=4, peer_missing_groups=((2, 3),))

        with patch.object(util_mod, "platform", fake):
            groups = util_mod.ensure_broadcast_groups([self._item("w")])

        self.assertEqual(fake.gathered, [((), ())])
        self.assertEqual(fake.created_groups, [(2, 3)])
        self.assertEqual(groups, {})


class TestBroadcastGroupScope(unittest.TestCase):
    """The groups a load broadcasts through last only as long as the load does."""

    @staticmethod
    def _item(fqn: str, source: Any = None) -> ReadItem:
        """One read item for ``fqn``, marked to be broadcast when ``source`` is given."""
        return ReadItem(
            type=LoadItemType.TENSOR,
            dest_index=MetadataIndex(fqn=fqn, offset=(0,), index=0),
            dest_offsets=(0,),
            storage_index=MetadataIndex(fqn=fqn, offset=(0,), index=0),
            storage_offsets=(0,),
            lengths=(2,),
            source=source,
        )

    def test_a_group_the_load_built_is_destroyed_when_it_is_done(self):
        """
        Feature: broadcast_groups_for_load lifetime.
        Description: A shard whose group the caller did not pre-build, used inside the scope.
        Expectation: The group is there for the read and destroyed on the way out. It exists
            to carry this one load, and a communicator kept past that holds device memory for
            the rest of the job.
        """
        fake = _RecordingPlatform(world_size=4)
        source = BroadcastSource(group_ranks=(0, 1), src_rank=0)

        with patch.object(util_mod, "platform", fake):
            with util_mod.broadcast_groups_for_load([self._item("w", source)]) as groups:
                self.assertEqual(groups, {(0, 1): "group(0, 1)"})
                self.assertEqual(fake.destroyed_groups, [])

        self.assertEqual(fake.destroyed_groups, ["group(0, 1)"])

    def test_the_group_a_load_built_is_drained_before_it_is_released(self):
        """
        Feature: broadcast_groups_for_load drains the device stream before releasing.
        Description: A broadcast that has been waited on is only ordered against the device
            stream, so the transfer can still be queued when the call returns. Releasing the
            communicator at that point tears the transfer down with it, and every rank that
            was to receive the shard silently keeps whatever its buffer held.
        Expectation: The drain is issued, and it precedes every release.
        """
        fake = _RecordingPlatform(world_size=4)
        source = BroadcastSource(group_ranks=(0, 1), src_rank=0)

        with patch.object(util_mod, "platform", fake):
            with util_mod.broadcast_groups_for_load([self._item("w", source)]):
                pass

        self.assertEqual(fake.events, ["synchronize", "destroy group(0, 1)"])

    def test_a_group_that_refuses_to_be_released_does_not_fail_the_load(self):
        """
        Feature: broadcast_groups_for_load best effort release.
        Description: The backend raises while releasing the group.
        Expectation: The scope is left cleanly all the same. The shards have arrived by
            then, so a group that will not go away is a leak worth a warning rather than a
            reason to fail a load that already has its data.
        """
        fake = _RecordingPlatform(world_size=4)
        fake.destroy_process_group = Mock(side_effect=RuntimeError("backend is grumpy"))
        source = BroadcastSource(group_ranks=(0, 1), src_rank=0)

        with patch.object(util_mod, "platform", fake):
            with util_mod.broadcast_groups_for_load([self._item("w", source)]) as groups:
                self.assertEqual(groups, {(0, 1): "group(0, 1)"})

        fake.destroy_process_group.assert_called_once_with("group(0, 1)")

    def test_groups_the_caller_built_are_left_alone(self):
        """
        Feature: broadcast_groups_for_load with a pre-built group.
        Description: The one group the plan needs was supplied by the caller.
        Expectation: Nothing is created and nothing is destroyed. The caller owns what it
            passed in and may well use it again; tearing it down here would take a group out
            from under whoever built it.
        """
        fake = _RecordingPlatform(world_size=4)
        source = BroadcastSource(group_ranks=(0, 1), src_rank=0)

        with patch.object(util_mod, "platform", fake):
            with util_mod.broadcast_groups_for_load(
                    [self._item("w", source)], {(0, 1): "pre_built"}) as groups:
                self.assertEqual(groups, {(0, 1): "pre_built"})

        self.assertEqual(fake.created_groups, [])
        self.assertEqual(fake.destroyed_groups, [])

    def test_a_group_is_destroyed_even_when_the_read_fails(self):
        """
        Feature: broadcast_groups_for_load on the failing path.
        Description: A read that raises inside the scope.
        Expectation: The group is destroyed and the error propagates. A load that fails
            partway is exactly when a leaked communicator would go unnoticed.
        """
        fake = _RecordingPlatform(world_size=4)
        source = BroadcastSource(group_ranks=(0, 1), src_rank=0)

        with patch.object(util_mod, "platform", fake):
            with self.assertRaises(RuntimeError):
                with util_mod.broadcast_groups_for_load([self._item("w", source)]):
                    raise RuntimeError("read failed")

        self.assertEqual(fake.destroyed_groups, ["group(0, 1)"])


    def test_a_group_that_already_exists_is_reused_rather_than_made_again(self):
        """
        Feature: broadcast_groups_for_load reusing what the mesh already built.
        Description: The group a shard needs is already in the process-wide cache, as a tp
            column or a dp group is once a model has been sharded over its mesh.
        Expectation: It is taken from there rather than created a second time, and left
            standing on the way out. Two communicators over the same ranks would cost
            device memory for nothing, and destroying this one would take it away from the
            mesh that is still using it.
        """
        fake = _RecordingPlatform(world_size=4, cached_groups={(0, 1): "mesh_group"})
        source = BroadcastSource(group_ranks=(0, 1), src_rank=0)

        with patch.object(util_mod, "platform", fake):
            with util_mod.broadcast_groups_for_load([self._item("w", source)]) as groups:
                self.assertEqual(groups, {(0, 1): "mesh_group"})

        self.assertEqual(fake.created_groups, [])
        self.assertEqual(fake.destroyed_groups, [])

    def test_a_group_only_some_ranks_have_is_made_afresh_on_all_of_them(self):
        """
        Feature: broadcast_groups_for_load when the caches disagree.
        Description: This rank has the group it needs cached; a peer reports needing the
            same group and not having it.
        Expectation: Every rank makes it, this one included, and every rank ends up on the
            new one. Creating a group is collective over the world, so a rank that skipped
            it on the strength of its own cache while another went ahead would hang them
            both -- which is why the choice is taken from what was gathered rather than
            from what is in hand.
        """
        fake = _RecordingPlatform(
            world_size=4, peer_missing_groups=((0, 1),), cached_groups={(0, 1): "stale_group"}
        )
        source = BroadcastSource(group_ranks=(0, 1), src_rank=0)

        with patch.object(util_mod, "platform", fake):
            with util_mod.broadcast_groups_for_load([self._item("w", source)]) as groups:
                self.assertEqual(groups, {(0, 1): "group(0, 1)"})

        self.assertEqual(fake.created_groups, [(0, 1)])
        self.assertEqual(fake.destroyed_groups, ["group(0, 1)"])

    def test_a_rank_takes_part_in_making_a_group_it_is_not_in(self):
        """
        Feature: broadcast_groups_for_load collective discipline.
        Description: A rank whose plan marked nothing, while a peer reports a group of its
            own that nobody has yet.
        Expectation: It makes it too, keeps nothing, and destroys nothing. Creating a group
            is collective over the world, so a rank that sat this out because the group does
            not contain it would leave the ranks it does contain waiting.
        """
        fake = _RecordingPlatform(world_size=4, peer_missing_groups=((2, 3),))

        with patch.object(util_mod, "platform", fake):
            with util_mod.broadcast_groups_for_load([self._item("w")]) as groups:
                self.assertEqual(groups, {})

        self.assertEqual(fake.created_groups, [(2, 3)])
        self.assertEqual(fake.destroyed_groups, [])


    def test_a_parameter_the_whole_world_holds_reuses_the_group_the_job_runs_on(self):
        """
        Feature: broadcast_groups_for_load and the world group.
        Description: A shard replicated on every rank, so the group it needs holds all of
            them - what a plain data-parallel job asks for on every replicated parameter.
        Expectation: The group the job already runs on is used, nothing is created, and
            nothing is destroyed. Building a second communicator over every rank for one
            read is pure cost, and destroying this one would take the job's own group away.
        """
        fake = _RecordingPlatform(world_size=4)
        source = BroadcastSource(group_ranks=(0, 1, 2, 3), src_rank=0)

        with patch.object(util_mod, "platform", fake):
            with util_mod.broadcast_groups_for_load([self._item("w", source)]) as groups:
                self.assertEqual(groups, {(0, 1, 2, 3): "world_group"})

        self.assertEqual(fake.created_groups, [])
        self.assertEqual(fake.destroyed_groups, [])


    def test_the_cache_is_preferred_to_the_world_group_for_the_same_ranks(self):
        """
        Feature: broadcast_groups_for_load choosing between what already exists.
        Description: A shard replicated on every rank, where the mesh has already built a
            group spanning all of them and put it in the cache - what a one-dimensional
            mesh over the whole job leaves behind.
        Expectation: That group is used rather than the one the job runs on. Both would
            work, but staying on the communicator the rest of training is using beats
            reaching past it, and either way nothing is created and nothing destroyed.
        """
        fake = _RecordingPlatform(
            world_size=4, cached_groups={(0, 1, 2, 3): "mesh_world_group"}
        )
        source = BroadcastSource(group_ranks=(0, 1, 2, 3), src_rank=0)

        with patch.object(util_mod, "platform", fake):
            with util_mod.broadcast_groups_for_load([self._item("w", source)]) as groups:
                self.assertEqual(groups, {(0, 1, 2, 3): "mesh_world_group"})

        self.assertEqual(fake.created_groups, [])
        self.assertEqual(fake.destroyed_groups, [])

    def test_groups_are_destroyed_in_the_order_every_rank_agrees_on(self):
        """
        Feature: broadcast_groups_for_load teardown order.
        Description: A plan needing several groups, reported in no particular order.
        Expectation: They go down in rank-tuple order. Destroying a group is collective among
            its members, and two ranks that shared two groups but tore them down in opposite
            orders would be waiting on each other.
        """
        fake = _RecordingPlatform(world_size=6)
        items = [
            self._item("c", BroadcastSource(group_ranks=(4, 5), src_rank=4)),
            self._item("a", BroadcastSource(group_ranks=(0, 1), src_rank=0)),
            self._item("b", BroadcastSource(group_ranks=(2, 3), src_rank=2)),
        ]

        with patch.object(util_mod, "platform", fake):
            with util_mod.broadcast_groups_for_load(items):
                pass

        self.assertEqual(
            fake.destroyed_groups, ["group(0, 1)", "group(2, 3)", "group(4, 5)"]
        )


class TestBroadcastBatcher(unittest.TestCase):
    """Tests for gathering small shards into one broadcast instead of sending each alone."""

    _GROUP = (0, 1)
    _BATCH_BYTES = 4096

    def setUp(self) -> None:
        """Rebuild util against the torch platform before every case."""
        os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
        _platform_mod.platform = None
        importlib.reload(util_mod)

    @staticmethod
    def _item(fqn: str, src_rank: int = 0, group: tuple = (0, 1)) -> ReadItem:
        """A read item for ``fqn``, marked as read by ``src_rank`` on behalf of ``group``."""
        return ReadItem(
            type=LoadItemType.TENSOR,
            dest_index=MetadataIndex(fqn=fqn, offset=(0,), index=0),
            dest_offsets=(0,),
            storage_index=MetadataIndex(fqn=fqn, offset=(0,), index=0),
            storage_offsets=(0,),
            lengths=(2,),
            source=BroadcastSource(group_ranks=group, src_rank=src_rank),
        )

    def _run(self, state: dict, items: list, fake: Any, batch_bytes: int = None) -> Any:
        """Hand every item to a batcher, flush it, and wait on what it started."""
        batcher = util_mod.BroadcastBatcher(
            self._BATCH_BYTES if batch_bytes is None else batch_bytes, {self._GROUP: "g", (2, 3): "h"}
        )
        in_flight = deque()
        with patch.object(util_mod, "platform", fake):
            for item in items:
                batcher.add(in_flight, state, item)
            batcher.flush(in_flight)
            util_mod.wait_broadcasts(in_flight)
        return batcher

    def test_small_shards_of_one_group_go_in_a_single_broadcast(self):
        """
        Feature: BroadcastBatcher gathering.
        Description: Four shards well under the batch size, all held by the same group and
            read by the same rank.
        Expectation: One broadcast carrying all four rather than four carrying one each.
            A broadcast costs about the same whatever it carries until the shards are some
            megabytes, so four small ones cost four times what they need to.
        """
        fake = _RecordingPlatform(rank=1)
        state = {f"w{i}": torch.zeros(4) for i in range(4)}
        batcher = self._run(state, [self._item(name) for name in state], fake)

        self.assertEqual(len(fake.broadcasts), 1)
        self.assertEqual((batcher.sent, batcher.batched), (1, 4))

    def test_a_shard_at_the_batch_size_is_sent_on_its_own(self):
        """
        Feature: BroadcastBatcher threshold.
        Description: One shard as large as the batch size, among smaller ones.
        Expectation: It goes in a broadcast of its own and the small ones in another. Past
            that size the cost of starting a broadcast is already small against what it
            carries, and gathering it would only buy a copy in and a copy out.
        """
        fake = _RecordingPlatform(rank=1)
        state = {"big": torch.zeros(self._BATCH_BYTES // 4, dtype=torch.float32),
                 "small_a": torch.zeros(4), "small_b": torch.zeros(4)}
        batcher = self._run(state, [self._item(name) for name in state], fake)

        self.assertEqual(len(fake.broadcasts), 2)
        self.assertEqual(batcher.batched, 2)

    def test_shards_that_cannot_travel_together_are_kept_apart(self):
        """
        Feature: BroadcastBatcher keys.
        Description: Small shards differing in group, in sending rank, and in dtype.
        Expectation: One broadcast each. A broadcast reaches one group from one rank and
            carries one buffer, so shards disagreeing on any of the three cannot share it.
        """
        fake = _RecordingPlatform(rank=1)
        state = {"a": torch.zeros(4), "b": torch.zeros(4),
                 "c": torch.zeros(4), "d": torch.zeros(4, dtype=torch.float64)}
        items = [
            self._item("a"),
            self._item("b", src_rank=1),
            self._item("c", group=(2, 3)),
            self._item("d"),
        ]
        batcher = self._run(state, items, fake)

        self.assertEqual(len(fake.broadcasts), 4)
        self.assertEqual(batcher.batched, 0)

    def test_a_batch_fills_up_and_goes_before_it_is_over_size(self):
        """
        Feature: BroadcastBatcher flushing.
        Description: More small shards than one batch holds, added one after another.
        Expectation: Every batch is sent before it would pass the size it was given, so the
            buffer set aside for it is bounded by that size rather than by the checkpoint.
        """
        fake = _RecordingPlatform(rank=1)
        per_shard = 4 * 4  # four float32
        state = {f"w{i:02d}": torch.zeros(4) for i in range(3 * (self._BATCH_BYTES // per_shard))}
        batcher = self._run(state, [self._item(name) for name in state], fake)

        self.assertEqual(len(fake.broadcasts), 3)
        self.assertTrue(
            all(sent.numel() * sent.element_size() <= self._BATCH_BYTES for sent, src, group in fake.broadcasts),
            [sent.numel() * sent.element_size() for sent, src, group in fake.broadcasts],
        )
        self.assertEqual(batcher.batched, len(state))

    def test_what_a_batch_carried_reaches_the_shards_it_was_gathered_from(self):
        """
        Feature: BroadcastBatcher dealing a batch out again.
        Description: A receiving rank taking a batch whose broadcast fills the gathered
            buffer with a known value, with the shards it stands for starting empty.
        Expectation: Every shard holds the value once the broadcast has been waited on, and
            not before. The batch travels through a buffer of its own rather than the state
            dict entries, so a load that never dealt it out, or dealt it out early, would
            leave them as they were.
        """
        fake = _RecordingPlatform(rank=1, fill=7.0)
        state = {f"w{i}": torch.zeros(4) for i in range(3)}
        batcher = util_mod.BroadcastBatcher(self._BATCH_BYTES, {self._GROUP: "g"})
        in_flight = deque()

        with patch.object(util_mod, "platform", fake):
            for name in state:
                batcher.add(in_flight, state, self._item(name))
            batcher.flush(in_flight)
            still_waiting = [float(state[name].sum()) for name in state]
            util_mod.wait_broadcasts(in_flight)

        self.assertEqual(still_waiting, [0.0, 0.0, 0.0])
        for name in state:
            self.assertTrue(torch.equal(state[name], torch.full((4,), 7.0)), name)
        self.assertEqual(batcher.batched, 3)

    def test_the_sending_rank_gathers_the_shards_it_already_holds(self):
        """
        Feature: BroadcastBatcher on the rank that read the shards.
        Description: The sending rank batching shards whose buffers already hold their data.
        Expectation: What goes out carries those values, and the shards are left alone
            afterwards. The sender has nothing to receive, so dealing the batch back out to
            it would only copy what is already there.
        """
        fake = _RecordingPlatform(rank=0, fill=7.0)
        state = {f"w{i}": torch.full((4,), float(i + 1)) for i in range(3)}
        self._run(state, [self._item(name) for name in state], fake)

        self.assertEqual(len(fake.broadcasts), 1)
        sent, src_rank, group = fake.broadcasts[0]
        self.assertEqual((src_rank, group), (0, "g"))
        self.assertEqual([float(value) for value in sent], [1.0] * 4 + [2.0] * 4 + [3.0] * 4)
        for index, name in enumerate(state):
            self.assertTrue(torch.equal(state[name], torch.full((4,), float(index + 1))), name)

    def test_a_batch_size_of_zero_sends_every_shard_on_its_own(self):
        """
        Feature: BroadcastBatcher turned off.
        Description: Small shards with the batch size set to zero, which is what a platform
            that cannot gather them is given.
        Expectation: One broadcast each, exactly as before there was any batching.
        """
        fake = _RecordingPlatform(rank=1)
        state = {f"w{i}": torch.zeros(4) for i in range(3)}
        batcher = self._run(state, [self._item(name) for name in state], fake, batch_bytes=0)

        self.assertEqual(len(fake.broadcasts), 3)
        self.assertEqual(batcher.batched, 0)


if __name__ == "__main__":
    unittest.main()
