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
    CHUNK_INFO,
    ChunkInfo,
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    TensorProperties,
    TensorStorageMetadata,
)
from hyper_parallel.core.distributed_checkpoint.planner import (
    LoadItemType,
    LoadPlan,
    ReadItem,
    SavePlan,
    WriteItem,
    WriteItemType,
)
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
        StandardSavePlanner.cached_save_result.clear()

    @staticmethod
    def _ragged_tensor(local, local_units=(1, 3)):
        """Build a rank-zero RaggedShard DTensor without initializing a backend."""
        _DEVICE_MESH_MAP.clear()
        EXISTING_COMM_GROUPS.clear()
        with patch("hyper_parallel.core.dtensor.device_mesh.dist.get_rank",
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
        Expectation: Metadata lists both tensors; only this rank's plan is returned, with chunk
            indices assigned per FQN.
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
        planner.rank = 1
        own_plan, metadata = planner.build_global_plan(local_plans)
        # Only rank 1's plan comes back, and it carries the item rank 1 contributed.
        self.assertEqual([item.index.fqn for item in own_plan.items], ["b"])
        self.assertEqual(own_plan.items[0].index.index, 0)
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

        self.assertTrue(planner._enable_plan_caching)
        self.assertEqual(
            [item.index.offset for item in plan.items],
            [(0, 0, 0), (1, 0, 0)],
        )
        self.assertEqual(
            [tuple(planner.get_data(item).shape) for item in plan.items],
            [(1, 4, 8), (1, 2, 8)],
        )

    def test_ragged_save_plan_cache_reuses_geometry_and_reads_current_data(self):
        """Reuse a RaggedShard plan while resolving data from the current state dict."""
        first = StandardSavePlanner(enable_plan_caching=True)
        first.configure_planner({"weight": self._ragged_tensor(torch.arange(48))}, rank=0)
        first_plan = first.build_local_plan()
        first_final, first_metadata = first.build_global_plan([first_plan])
        first.cache_result(first.finalize_plan(first_final), first_metadata)

        second = StandardSavePlanner(enable_plan_caching=True)
        second.configure_planner({"weight": self._ragged_tensor(torch.arange(48, 96))}, rank=0)
        cached = second.get_cached()

        self.assertIsNotNone(cached)
        self.assertEqual(len(cached.final_plan.items), 2)
        pieces = [second.get_data(item) for item in cached.final_plan.items]
        self.assertEqual([tuple(piece.shape) for piece in pieces], [(1, 4, 8), (1, 2, 8)])
        torch.testing.assert_close(pieces[0].reshape(-1), torch.arange(48, 80))
        torch.testing.assert_close(pieces[1].reshape(-1), torch.arange(80, 96))

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

        # The rank lookup happens on the shared ``platform`` object imported from
        # util, so patch the method on it rather than a module-level getter.
        with patch(
                "hyper_parallel.core.distributed_checkpoint.util.platform.get_rank",
                return_value=0,
        ):
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

    @staticmethod
    def _chunk_tagged(tensor, replica_rank_list=None):
        """Tag a plain tensor the way an integration that does not use DTensor does."""
        shape = tuple(tensor.shape)
        setattr(
            tensor,
            CHUNK_INFO,
            ChunkInfo(
                chunk=ChunkStorageMetadata(offsets=(0,) * len(shape), sizes=shape),
                global_shape=shape,
                replica_rank_list=replica_rank_list,
            ),
        )
        return tensor

    @staticmethod
    def _read_item(fqn, offset, dest_offsets, lengths, index=0):
        """A tensor read item covering one region of the local chunk at ``index``."""
        return ReadItem(
            type=LoadItemType.TENSOR,
            dest_index=MetadataIndex(fqn=fqn, offset=offset, index=index),
            dest_offsets=dest_offsets,
            storage_index=MetadataIndex(fqn=fqn, offset=(0, 0), index=0),
            storage_offsets=dest_offsets,
            lengths=lengths,
        )

    @staticmethod
    def _single_chunk_metadata(fqn, shape):
        """Metadata for one tensor the checkpoint holds as a single full chunk."""
        return Metadata(
            state_dict_metadata={
                fqn: TensorStorageMetadata(
                    properties=TensorProperties(dtype="torch.float32"),
                    size=shape,
                    chunks=[ChunkStorageMetadata(offsets=(0,) * len(shape), sizes=shape)],
                )
            }
        )

    def _configured_load_planner(self, state, metadata, rank, **kwargs):
        """A load planner configured for one rank of a broadcasting load."""
        planner = StandardLoadPlanner()
        planner.configure_planner(
            state, metadata, rank=rank, broadcast_replicated_tensors=True, **kwargs
        )
        return planner

    @staticmethod
    def _local_plan_on_device(planner):
        """
        Build a local plan as though its buffers sat on the device.

        Unit tests run on the host, and a buffer left there is never handed to a broadcast,
        so a test about who reads for whom has to say the buffers are where a real load
        keeps them. What happens to one that really is on the host has its own test.
        """
        with patch.object(planner_mod, "_is_on_host", return_value=False):
            return planner.build_local_plan()

    def _plan_per_rank(self, state, metadata, all_plans):
        """
        What each rank ends up with, by running the global plan on each of them in turn.

        build_global_plan hands back only the plan of the rank running it, so a caller after
        every rank decision has to ask every rank, which is what the ranks themselves do.
        """
        return [
            self._configured_load_planner(state, metadata, rank=rank).build_global_plan(all_plans)
            for rank in range(len(all_plans))
        ]

    def test_local_plan_covers_every_read_the_rank_needs(self):
        """
        Feature: StandardLoadPlanner.build_local_plan.
        Description: Plan a load for a rank that shares its tensor with another one, with
            broadcasting turned on.
        Expectation: The rank still plans the read itself. Who ends up reading is settled
            globally afterwards, so a local plan that already dropped the read would leave
            the global one with nothing to choose between.
        """
        weight = self._chunk_tagged(torch.zeros(4, 4))
        planner = self._configured_load_planner(
            {"weight": weight}, self._single_chunk_metadata("weight", (4, 4)), rank=1
        )

        plan = planner.build_local_plan()

        self.assertEqual(len(plan.items), 1)
        self.assertIsNone(plan.items[0].source)

    def test_global_plan_hands_a_shared_tensor_to_one_rank(self):
        """
        Feature: StandardLoadPlanner.build_global_plan.
        Description: Two ranks whose plans ask for exactly the same region of a tensor.
        Expectation: Both plans keep the item and both name the same reader, one of the two.
            The reader still reads; the other one is told where its copy comes from. Neither
            plan loses the item, because the receiving rank has to join the broadcast.
        """
        state = {"weight": self._chunk_tagged(torch.zeros(4, 4))}
        metadata = self._single_chunk_metadata("weight", (4, 4))
        local = self._local_plan_on_device(self._configured_load_planner(state, metadata, rank=0))

        plans = self._plan_per_rank(state, metadata, [local, local])

        sources = [plan.items[0].source for plan in plans]
        self.assertEqual(len(plans), 2)
        self.assertTrue(all(len(plan.items) == 1 for plan in plans))
        self.assertEqual(sources[0], sources[1])
        self.assertEqual(sources[0].group_ranks, (0, 1))
        self.assertIn(sources[0].src_rank, (0, 1))

    def test_global_plan_leaves_a_uniquely_held_tensor_alone(self):
        """
        Feature: StandardLoadPlanner.build_global_plan.
        Description: Two ranks loading different halves of a tensor, so neither can read for
            the other.
        Expectation: No item is marked, and every rank reads its own half as before.
        """
        weight = self._chunk_tagged(torch.zeros(4, 4))
        planner = self._configured_load_planner(
            {"weight": weight}, self._single_chunk_metadata("weight", (4, 4)), rank=0
        )
        halves = [
            LoadPlan(items=[self._read_item("weight", (offset, 0), (0, 0), (2, 4))])
            for offset in (0, 2)
        ]

        global_plan = planner.build_global_plan(halves)

        self.assertTrue(all(item.source is None for item in global_plan.items))

    def test_global_plan_is_a_no_op_when_broadcasting_is_off(self):
        """
        Feature: StandardLoadPlanner.build_global_plan without broadcasting.
        Description: The same two identical plans, with broadcasting left off.
        Expectation: The rank's plan comes back untouched, so every rank reads for itself and
            the load issues no collective at all.
        """
        weight = self._chunk_tagged(torch.zeros(4, 4))
        planner = StandardLoadPlanner()
        planner.configure_planner(
            {"weight": weight}, self._single_chunk_metadata("weight", (4, 4)), rank=0
        )
        local = planner.build_local_plan()

        global_plan = planner.build_global_plan([local, local])

        self.assertEqual(global_plan, local)

    def test_global_plan_spreads_the_reads_across_the_group(self):
        """
        Feature: StandardLoadPlanner.build_global_plan reader balancing.
        Description: Four ranks all loading the same four tensors, so every tensor could be
            read by any of them.
        Expectation: The four reads land on four different ranks rather than all on rank 0,
            which is what keeps three ranks from idling at the broadcast while one reads.
        """
        state = {f"w{index}": self._chunk_tagged(torch.zeros(4, 4)) for index in range(4)}
        metadata = Metadata(state_dict_metadata={})
        for fqn in state:
            metadata.state_dict_metadata.update(
                self._single_chunk_metadata(fqn, (4, 4)).state_dict_metadata
            )
        planner = self._configured_load_planner(state, metadata, rank=0)
        local = self._local_plan_on_device(planner)

        global_plan = planner.build_global_plan([local] * 4)

        readers = {item.source.src_rank for item in global_plan.items}
        self.assertEqual(readers, {0, 1, 2, 3})


    def _multi_tensor_metadata(self, shapes):
        """Metadata for several tensors, each held as one full chunk."""
        metadata = Metadata(state_dict_metadata={})
        for fqn, shape in shapes.items():
            metadata.state_dict_metadata.update(
                self._single_chunk_metadata(fqn, shape).state_dict_metadata
            )
        return metadata

    def test_the_largest_tensor_is_placed_first(self):
        """
        Feature: StandardLoadPlanner.build_global_plan reader balancing.
        Description: Two ranks loading three identical tensors, one of them ten times the
            size of the other two.
        Expectation: The two small tensors end up on the rank that did not get the large
            one. Placing the small ones first would put one of them with the large tensor
            and overshoot, which is why the tensors are placed in descending size order.
        """
        shapes = {"big": (10,), "small_a": (1,), "small_b": (1,)}
        state = {fqn: self._chunk_tagged(torch.zeros(*shape)) for fqn, shape in shapes.items()}
        planner = self._configured_load_planner(state, self._multi_tensor_metadata(shapes), rank=0)
        local = self._local_plan_on_device(planner)

        global_plan = planner.build_global_plan([local, local])

        readers = {item.dest_index.fqn: item.source.src_rank for item in global_plan.items}
        self.assertEqual(readers["small_a"], readers["small_b"])
        self.assertNotEqual(readers["big"], readers["small_a"])

    def test_a_buffer_left_on_the_host_is_read_by_every_rank(self):
        """
        Feature: StandardLoadPlanner broadcast eligibility.
        Description: Two ranks holding the same tensor, whose destination stayed in host
            memory. An optimizer's step counter is the one that turns up in practice: it
            keeps to the host while the moments beside it follow the parameter onto the
            device.
        Expectation: The local plan says so, and no item is marked, so every rank reads its
            own copy. What stays on the host is little enough that reading it costs less
            than moving it, and the group a broadcast would go through, raised on the
            accelerator library, holds no backend for host memory in any case.
        """
        state = {"step": self._chunk_tagged(torch.zeros(4, 4))}
        metadata = self._single_chunk_metadata("step", (4, 4))
        local = self._configured_load_planner(state, metadata, rank=0).build_local_plan()

        plans = self._plan_per_rank(state, metadata, [local, local])

        self.assertTrue(local.items)
        self.assertTrue(all(item.broadcastable is False for item in local.items))
        self.assertTrue(all(item.source is None for plan in plans for item in plan.items))

    def test_one_rank_reporting_a_host_buffer_stops_the_whole_shard(self):
        """
        Feature: StandardLoadPlanner broadcast eligibility across ranks.
        Description: Two ranks holding the same tensor, one of them with its destination on
            the host and the other on the device.
        Expectation: Neither rank marks the shard. A broadcast needs every member to take
            part, so one rank that will not receive settles it for all of them -- and both
            ranks read the same report out of the gather, so they settle it the same way
            rather than one sending into a group the other never joined.
        """
        state = {"weight": self._chunk_tagged(torch.zeros(4, 4))}
        metadata = self._single_chunk_metadata("weight", (4, 4))
        on_device = self._local_plan_on_device(
            self._configured_load_planner(state, metadata, rank=0))
        on_host = self._configured_load_planner(state, metadata, rank=1).build_local_plan()

        plans = self._plan_per_rank(state, metadata, [on_device, on_host])

        self.assertTrue(all(item.broadcastable for item in on_device.items))
        self.assertTrue(all(not item.broadcastable for item in on_host.items))
        self.assertTrue(all(item.source is None for plan in plans for item in plan.items))

    def test_byte_io_items_are_never_handed_to_another_rank(self):
        """
        Feature: StandardLoadPlanner.build_global_plan item filtering.
        Description: Every rank loads the same pickled entry, whose read items are identical
            by construction.
        Expectation: The items stay unmarked. The entry is a Python object each rank rebuilds
            from bytes, not tensor storage a collective could write into, so grouping it
            would hand the others a broadcast that cannot happen.
        """
        from hyper_parallel.core.distributed_checkpoint.metadata import BytesStorageMetadata

        planner = self._configured_load_planner(
            {"opt_state": None},
            Metadata(state_dict_metadata={"opt_state": BytesStorageMetadata()}),
            rank=0,
        )
        local = planner.build_local_plan()

        global_plan = planner.build_global_plan([local, local])

        self.assertTrue(all(item.source is None for item in global_plan.items))

    def test_the_decision_does_not_depend_on_item_order_within_a_plan(self):
        """
        Feature: StandardLoadPlanner.build_global_plan determinism.
        Description: The same four tensors, listed in a different order in each of the two
            plans, as two ranks with differently ordered state dicts would produce.
        Expectation: Both plans name the same reader for every tensor. The ranks of a group
            never compare notes afterwards, so a decision that shifted with the order of the
            gathered items would leave them disagreeing on the src of their broadcast.
        """
        shapes = {name: (4,) for name in ("a", "b", "c", "d")}
        state = {fqn: self._chunk_tagged(torch.zeros(*shape)) for fqn, shape in shapes.items()}
        metadata = self._multi_tensor_metadata(shapes)
        forward = self._local_plan_on_device(self._configured_load_planner(state, metadata, rank=0))
        backward = LoadPlan(items=list(reversed(forward.items)))

        plans = self._plan_per_rank(state, metadata, [forward, backward])

        readers = [
            {item.dest_index.fqn: item.source.src_rank for item in plan.items}
            for plan in plans
        ]
        self.assertEqual(readers[0], readers[1])


    def test_each_shard_gets_the_ranks_that_hold_that_shard(self):
        """
        Feature: StandardLoadPlanner.build_global_plan grouping.
        Description: One tensor split over two shards, interleaved across four ranks: 0 and
            2 hold the top half, 1 and 3 hold the bottom half. Every item carries the same
            fqn, and neither group is a contiguous run of ranks.
        Expectation: Two groups, (0, 2) and (1, 3), each broadcasting its own half. What a
            rank loads is its shard, not the tensor the shard belongs to; grouping by name
            would put all four together and hand two of them the wrong half.
        """
        state = {"weight": self._chunk_tagged(torch.zeros(4, 4))}
        metadata = self._single_chunk_metadata("weight", (4, 4))
        top = LoadPlan(items=[self._read_item("weight", (0, 0), (0, 0), (2, 4))])
        bottom = LoadPlan(items=[self._read_item("weight", (2, 0), (0, 0), (2, 4))])

        plans = self._plan_per_rank(state, metadata, [top, bottom, top, bottom])

        sources = [plan.items[0].source for plan in plans]
        self.assertEqual([source.group_ranks for source in sources],
                         [(0, 2), (1, 3), (0, 2), (1, 3)])
        self.assertIn(sources[0].src_rank, (0, 2))
        self.assertIn(sources[1].src_rank, (1, 3))


    def test_two_shards_held_by_the_same_pair_are_handed_out_separately(self):
        """
        Feature: StandardLoadPlanner.build_global_plan shard independence.
        Description: Two ranks holding two shards each of one tensor, the same two shards.
            Grouping them says nothing about who reads them.
        Expectation: One shard to each rank, so the pair reads half the tensor apiece and
            sends the other half to its neighbour. Shards are placed one at a time and carry
            their own broadcast, so holding two of them together does not make them travel
            together -- that would leave one rank reading everything and the other idle.
        """
        state = {"weight": self._chunk_tagged(torch.zeros(8, 4))}
        metadata = self._single_chunk_metadata("weight", (8, 4))
        both = LoadPlan(items=[
            self._read_item("weight", (0, 0), (0, 0), (4, 4), index=0),
            self._read_item("weight", (4, 0), (0, 0), (4, 4), index=1),
        ])

        plans = self._plan_per_rank(state, metadata, [both, both])

        readers = {item.dest_index.offset: item.source.src_rank for item in plans[0].items}
        self.assertEqual(sorted(readers.values()), [0, 1])
        self.assertTrue(all(item.source.group_ranks == (0, 1)
                            for plan in plans for item in plan.items))

    def test_a_shard_assembled_from_several_reads_is_marked_on_every_item(self):
        """
        Feature: StandardLoadPlanner.build_global_plan over a resharded read.
        Description: One local shard put together out of two regions of the checkpoint,
            which is what a load that reshards produces: two read items share a dest_index,
            each covering part of the destination. Two ranks hold that shard.
        Expectation: Both items come back carrying the same source. A broadcast sends the
            whole local buffer once every piece is in place, so an item left unmarked would
            send its rank to storage for a region the reader is already sending it.
        """
        state = {"weight": self._chunk_tagged(torch.zeros(4, 4))}
        metadata = self._single_chunk_metadata("weight", (4, 4))
        two_pieces = LoadPlan(items=[
            self._read_item("weight", (0, 0), (0, 0), (2, 4)),
            self._read_item("weight", (0, 0), (2, 0), (2, 4)),
        ])

        plans = self._plan_per_rank(state, metadata, [two_pieces, two_pieces])

        for rank, marked in enumerate(plans):
            sources = [item.source for item in marked.items]
            self.assertEqual(len(sources), 2, f"rank {rank} came back with {len(sources)} items")
            self.assertIsNotNone(sources[0], f"rank {rank} left the first piece unmarked")
            self.assertEqual(sources[0], sources[1],
                             f"rank {rank} gave the two pieces of one shard different sources")
            self.assertEqual(sources[0].group_ranks, (0, 1))

    def test_a_shard_is_weighed_by_all_of_its_reads_not_one_of_them(self):
        """
        Feature: StandardLoadPlanner.build_global_plan balancing resharded reads.
        Description: Two shards held by the same pair of ranks. One is assembled from two
            reads of 8 elements each, the other from a single read of 12. Weighed whole the
            first is the heavier of the two; weighed by one of its pieces it is the lighter.
        Expectation: The two-piece shard is placed first and so goes to rank 0, the other to
            rank 1. Weighing a resharded shard by one piece would understate what reading it
            costs and pile the real work onto one rank.
        """
        state = {
            "split": self._chunk_tagged(torch.zeros(4, 4)),
            "whole": self._chunk_tagged(torch.zeros(3, 4)),
        }
        metadata = self._multi_tensor_metadata({"split": (4, 4), "whole": (3, 4)})
        plan = LoadPlan(items=[
            self._read_item("split", (0, 0), (0, 0), (2, 4)),
            self._read_item("split", (0, 0), (2, 0), (2, 4)),
            self._read_item("whole", (0, 0), (0, 0), (3, 4)),
        ])

        plans = self._plan_per_rank(state, metadata, [plan, plan])

        readers = {item.dest_index.fqn: item.source.src_rank for item in plans[0].items}
        self.assertEqual(readers["split"], 0)
        self.assertEqual(readers["whole"], 1)


if __name__ == "__main__":
    unittest.main()
