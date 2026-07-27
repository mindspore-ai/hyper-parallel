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
from hyper_parallel.core.distributed_checkpoint.topology_mapper import TopologyMapper


class TestStandardPlanner(unittest.TestCase):
    """Tests for StandardSavePlanner and StandardLoadPlanner."""

    def setUp(self) -> None:
        os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
        _platform_mod.platform = None
        importlib.reload(planner_mod)
        StandardSavePlanner._cached_save_result.clear()

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

    def test_global_plan_pp_prefix_isolation_dp_dedup_tp_chunks(self):
        """
        Feature: StandardSavePlanner.build_global_plan with PP stage prefix.
        Description: Two PP stages, each with 2 DP replicas and 2 TP chunks (8 ranks total).
            Each rank submits its own SavePlan. Stage-0 and stage-1 share the same local
            parameter name "layers.0.weight". The nested state_dict uses pp_stage_{pp_rank}
            as key so that flatten produces distinct FQNs like
            "model.pp_stage_0.layers.0.weight" and "model.pp_stage_1.layers.0.weight".
            DP replicas within the same stage submit identical MetadataIndex entries
            (same FQN + same offset), so remove_redundant_plans deduplicates them.
        Expectation:
            - Metadata contains two independent TensorStorageMetadata entries (one per stage).
            - Each stage retains exactly 2 TP chunks (DP replicas are deduplicated).
            - DP dedup does NOT cross stage boundaries.
            - Chunk offsets, sizes and global shape are preserved correctly.
        """
        props = TensorProperties(dtype="float32")
        global_shape = (4, 8)
        tp_chunk_size = (4, 4)

        chunk_tp0 = ChunkStorageMetadata(offsets=(0, 0), sizes=tp_chunk_size)
        chunk_tp1 = ChunkStorageMetadata(offsets=(0, 4), sizes=tp_chunk_size)

        pp_size, dp_size, tp_size = 2, 2, 2
        local_plans = []
        for pp_rank in range(pp_size):
            for dp_rank in range(dp_size):
                for tp_rank in range(tp_size):
                    chunk = chunk_tp0 if tp_rank == 0 else chunk_tp1
                    fqn = f"model.pp_stage_{pp_rank}.layers.0.weight"
                    item = WriteItem(
                        index=MetadataIndex(fqn=fqn, offset=chunk.offsets),
                        type=WriteItemType.TENSOR,
                        tensor_data={"chunk": chunk, "properties": props, "size": global_shape},
                    )
                    local_plans.append(SavePlan(items=[item], planner_data={}))

        planner = StandardSavePlanner(enable_plan_caching=False, remove_redundancy=True)
        flat_state = {
            "model.pp_stage_0.layers.0.weight": torch.zeros(*global_shape),
            "model.pp_stage_1.layers.0.weight": torch.zeros(*global_shape),
        }
        planner.configure_planner(flat_state, use_collectives=True)

        global_plans, metadata = planner.build_global_plan(local_plans)

        fqn0 = "model.pp_stage_0.layers.0.weight"
        fqn1 = "model.pp_stage_1.layers.0.weight"
        self.assertIn(fqn0, metadata.state_dict_metadata)
        self.assertIn(fqn1, metadata.state_dict_metadata)

        md0 = metadata.state_dict_metadata[fqn0]
        md1 = metadata.state_dict_metadata[fqn1]
        self.assertIsInstance(md0, TensorStorageMetadata)
        self.assertIsInstance(md1, TensorStorageMetadata)

        self.assertEqual(len(md0.chunks), 2, "Stage 0 should have exactly 2 TP chunks after DP dedup")
        self.assertEqual(len(md1.chunks), 2, "Stage 1 should have exactly 2 TP chunks after DP dedup")

        self.assertEqual(md0.size, global_shape)
        self.assertEqual(md1.size, global_shape)

        all_offsets = {c.offsets for c in md0.chunks}
        self.assertIn((0, 0), all_offsets)
        self.assertIn((0, 4), all_offsets)

        all_offsets_1 = {c.offsets for c in md1.chunks}
        self.assertIn((0, 0), all_offsets_1)
        self.assertIn((0, 4), all_offsets_1)


    def test_load_planner_delegates_to_topology_mapper(self):
        """
        Feature: StandardLoadPlanner delegates to TopologyMapper.
        Description: Planner with explicit topology_mapper maps target FQN to checkpoint FQN.
        Expectation: ReadItem dest_index.fqn is target, storage_index.fqn is checkpoint FQN.
        """
        mapping = {"model.pp_stage_1.layers.0.weight": "model.pp_stage_0.layers.0.weight"}
        mapper = TopologyMapper(target_to_checkpoint_fqn=mapping)
        checkpoint_md = TensorStorageMetadata(
            properties=TensorProperties(dtype="float32"),
            size=(4, 4),
            chunks=[ChunkStorageMetadata(offsets=(0, 0), sizes=(4, 4))],
        )
        metadata = Metadata(
            state_dict_metadata={"model.pp_stage_0.layers.0.weight": checkpoint_md},
        )
        weight = torch.zeros(4, 4)
        state = {"model.pp_stage_1.layers.0.weight": weight}
        planner = StandardLoadPlanner(topology_mapper=mapper)
        planner.configure_planner(state, metadata, use_collectives=False)
        plan = planner.build_local_plan()
        self.assertEqual(len(plan.items), 1)
        self.assertEqual(plan.items[0].dest_index.fqn, "model.pp_stage_1.layers.0.weight")
        self.assertEqual(plan.items[0].storage_index.fqn, "model.pp_stage_0.layers.0.weight")

    def test_load_planner_missing_key_reports_both_fqns(self):
        """
        Feature: StandardLoadPlanner with topology_mapper reports both FQNs on missing key.
        Description: Target FQN maps to a checkpoint FQN that does not exist in metadata.
        Expectation: RuntimeError mentions both target and checkpoint FQN.
        """
        mapping = {"target.weight": "nonexistent.weight"}
        mapper = TopologyMapper(target_to_checkpoint_fqn=mapping)
        metadata = Metadata(state_dict_metadata={})
        state = {"target.weight": torch.zeros(4, 4)}
        planner = StandardLoadPlanner(topology_mapper=mapper)
        planner.configure_planner(state, metadata, use_collectives=False)
        with self.assertRaises(RuntimeError) as ctx:
            planner.build_local_plan()
        msg = str(ctx.exception)
        self.assertIn("target.weight", msg)
        self.assertIn("nonexistent.weight", msg)

    def test_load_planner_shape_mismatch_reports_both_fqns(self):
        """
        Feature: StandardLoadPlanner with topology_mapper reports both FQNs on shape mismatch.
        Description: Target tensor has different shape from checkpoint metadata.
        Expectation: ValueError mentions both target and checkpoint FQN.
        """
        mapping = {"target.weight": "source.weight"}
        mapper = TopologyMapper(target_to_checkpoint_fqn=mapping)
        checkpoint_md = TensorStorageMetadata(
            properties=TensorProperties(dtype="float32"),
            size=(4, 4),
            chunks=[ChunkStorageMetadata(offsets=(0, 0), sizes=(4, 4))],
        )
        metadata = Metadata(state_dict_metadata={"source.weight": checkpoint_md})
        state = {"target.weight": torch.zeros(2, 2)}
        planner = StandardLoadPlanner(topology_mapper=mapper)
        planner.configure_planner(state, metadata, use_collectives=False)
        with self.assertRaises(ValueError) as ctx:
            planner.build_local_plan()
        msg = str(ctx.exception)
        self.assertIn("target.weight", msg)
        self.assertIn("source.weight", msg)

    def test_load_planner_default_mapper_is_identity(self):
        """
        Feature: StandardLoadPlanner without topology_mapper uses identity.
        Description: No mapper provided; planner should behave as before.
        Expectation: ReadItem dest_index.fqn == storage_index.fqn == target_fqn.
        """
        checkpoint_md = TensorStorageMetadata(
            properties=TensorProperties(dtype="float32"),
            size=(4, 4),
            chunks=[ChunkStorageMetadata(offsets=(0, 0), sizes=(4, 4))],
        )
        metadata = Metadata(state_dict_metadata={"w": checkpoint_md})
        state = {"w": torch.zeros(4, 4)}
        planner = StandardLoadPlanner()
        planner.configure_planner(state, metadata, use_collectives=False)
        plan = planner.build_local_plan()
        self.assertEqual(len(plan.items), 1)
        self.assertEqual(plan.items[0].dest_index.fqn, "w")
        self.assertEqual(plan.items[0].storage_index.fqn, "w")

    def test_hsdp_ep_expert_weight_dedup(self):
        """
        Feature: StandardSavePlanner.build_global_plan deduplicates HSDP replica chunks.
        Description: 8 physical ranks in HSDP(replicate=2, shard=2) × EP=2 topology.
            Expert weight global shape (8, 16) is sharded along dim-0 by both
            HSDP shard (2 slices) and EP (2 slices), producing 4 unique logical
            chunks.  Each HSDP replicate group holds identical copies, so 8 ranks
            submit 8 WriteItems but only 4 unique chunks should remain after
            ``remove_redundant_plans``.
        Expectation:
            - Metadata for the expert weight has exactly 4 chunks.
            - Each chunk has distinct (offsets, sizes).
            - Global shape, dtype, and chunk offsets/sizes are preserved.
        """
        props = TensorProperties(dtype="float32")
        global_shape = (8, 16)

        chunk_0_0 = ChunkStorageMetadata(offsets=(0, 0), sizes=(2, 16))
        chunk_0_1 = ChunkStorageMetadata(offsets=(2, 0), sizes=(2, 16))
        chunk_1_0 = ChunkStorageMetadata(offsets=(4, 0), sizes=(2, 16))
        chunk_1_1 = ChunkStorageMetadata(offsets=(6, 0), sizes=(2, 16))
        unique_chunks = [chunk_0_0, chunk_0_1, chunk_1_0, chunk_1_1]

        hsdp_replicate = 2
        hsdp_shard = 2
        ep_size = 2
        fqn = "model.moe.experts.w1"

        local_plans = []
        for rep in range(hsdp_replicate):
            for shard_idx in range(hsdp_shard):
                for ep_idx in range(ep_size):
                    chunk_idx = shard_idx * ep_size + ep_idx
                    chunk = unique_chunks[chunk_idx]
                    item = WriteItem(
                        index=MetadataIndex(fqn=fqn, offset=chunk.offsets),
                        type=WriteItemType.TENSOR,
                        tensor_data={
                            "chunk": chunk,
                            "properties": props,
                            "size": global_shape,
                        },
                    )
                    local_plans.append(SavePlan(items=[item], planner_data={}))

        self.assertEqual(len(local_plans), 8, "Should have 8 physical rank plans")

        planner = StandardSavePlanner(enable_plan_caching=False, remove_redundancy=True)
        flat_state = {fqn: torch.zeros(*global_shape)}
        planner.configure_planner(flat_state, use_collectives=True)

        global_plans, metadata = planner.build_global_plan(local_plans)

        md = metadata.state_dict_metadata[fqn]
        self.assertIsInstance(md, TensorStorageMetadata)
        self.assertEqual(
            len(md.chunks), 4,
            f"Expected 4 unique expert chunks after HSDP dedup, got {len(md.chunks)}",
        )
        self.assertEqual(md.size, global_shape)

        all_chunk_offsets = {c.offsets for c in md.chunks}
        expected_offsets = {c.offsets for c in unique_chunks}
        self.assertEqual(all_chunk_offsets, expected_offsets)

    def test_hsdp_ep_replicated_router_dedup(self):
        """
        Feature: Replicated router/buffer tensors are deduplicated across HSDP and EP groups.
        Description: 8 ranks each submit a full-tensor WriteItem for a replicated router
            weight (shape (16,)). All 8 items share the same FQN and offset (0,).
            ``remove_redundant_plans`` must collapse them to a single chunk.
        Expectation: Metadata has exactly 1 chunk for the router weight.
        """
        props = TensorProperties(dtype="float32")
        global_shape = (16,)

        chunk_full = ChunkStorageMetadata(offsets=(0,), sizes=global_shape)
        fqn = "model.moe.router.gate.weight"

        local_plans = []
        for _ in range(8):
            item = WriteItem(
                index=MetadataIndex(fqn=fqn, offset=chunk_full.offsets),
                type=WriteItemType.TENSOR,
                tensor_data={
                    "chunk": chunk_full,
                    "properties": props,
                    "size": global_shape,
                },
            )
            local_plans.append(SavePlan(items=[item], planner_data={}))

        planner = StandardSavePlanner(enable_plan_caching=False, remove_redundancy=True)
        flat_state = {fqn: torch.zeros(*global_shape)}
        planner.configure_planner(flat_state, use_collectives=True)

        _, metadata = planner.build_global_plan(local_plans)

        md = metadata.state_dict_metadata[fqn]
        self.assertIsInstance(md, TensorStorageMetadata)
        self.assertEqual(
            len(md.chunks), 1,
            f"Replicated router should have 1 chunk after dedup, got {len(md.chunks)}",
        )
        self.assertEqual(md.chunks[0].offsets, (0,))
        self.assertEqual(md.chunks[0].sizes, global_shape)

    def test_hsdp_ep_expert_and_router_combined_dedup(self):
        """
        Feature: Combined HSDP+EP checkpoint deduplicates both sharded experts and replicated router.
        Description: 8 ranks each submit plans for:
            - experts.w1: 4 unique chunks (2 HSDP-shard × 2 EP), each duplicated by HSDP replicate
            - router.gate.weight: fully replicated, 8 identical copies
        Expectation:
            - experts.w1 has 4 unique chunks in metadata.
            - router.gate.weight has 1 unique chunk in metadata.
        """
        props = TensorProperties(dtype="float32")
        expert_global = (8, 16)
        router_global = (16,)

        e_chunks = [
            ChunkStorageMetadata(offsets=(0, 0), sizes=(2, 16)),
            ChunkStorageMetadata(offsets=(2, 0), sizes=(2, 16)),
            ChunkStorageMetadata(offsets=(4, 0), sizes=(2, 16)),
            ChunkStorageMetadata(offsets=(6, 0), sizes=(2, 16)),
        ]
        r_chunk = ChunkStorageMetadata(offsets=(0,), sizes=router_global)

        hsdp_replicate, hsdp_shard, ep_size = 2, 2, 2

        local_plans = []
        for rep in range(hsdp_replicate):
            for shard_idx in range(hsdp_shard):
                for ep_idx in range(ep_size):
                    expert_chunk = e_chunks[shard_idx * ep_size + ep_idx]
                    expert_item = WriteItem(
                        index=MetadataIndex(fqn="experts.w1", offset=expert_chunk.offsets),
                        type=WriteItemType.TENSOR,
                        tensor_data={"chunk": expert_chunk, "properties": props, "size": expert_global},
                    )
                    router_item = WriteItem(
                        index=MetadataIndex(fqn="router.gate.weight", offset=r_chunk.offsets),
                        type=WriteItemType.TENSOR,
                        tensor_data={"chunk": r_chunk, "properties": props, "size": router_global},
                    )
                    local_plans.append(SavePlan(items=[expert_item, router_item], planner_data={}))

        planner = StandardSavePlanner(enable_plan_caching=False, remove_redundancy=True)
        flat_state = {"experts.w1": torch.zeros(*expert_global), "router.gate.weight": torch.zeros(*router_global)}
        planner.configure_planner(flat_state, use_collectives=True)

        _, metadata = planner.build_global_plan(local_plans)

        e_md = metadata.state_dict_metadata["experts.w1"]
        r_md = metadata.state_dict_metadata["router.gate.weight"]
        self.assertEqual(len(e_md.chunks), 4, "experts.w1 should have 4 unique chunks")
        self.assertEqual(len(r_md.chunks), 1, "router should have 1 unique chunk")

    def test_load_planner_bytes_item_uses_mapped_fqn(self):
        """
        Feature: StandardLoadPlanner with topology_mapper on bytes items.
        Description: Bytes object with PP FQN mapping.
        Expectation: dest_index.fqn is target, storage_index.fqn is checkpoint FQN.
        """
        from hyper_parallel.core.distributed_checkpoint.metadata import BytesStorageMetadata

        mapping = {"target.state": "source.state"}
        mapper = TopologyMapper(target_to_checkpoint_fqn=mapping)
        metadata = Metadata(
            state_dict_metadata={"source.state": BytesStorageMetadata()},
        )
        state = {"target.state": None}
        planner = StandardLoadPlanner(topology_mapper=mapper)
        planner.configure_planner(state, metadata, use_collectives=False)
        plan = planner.build_local_plan()
        self.assertEqual(len(plan.items), 1)
        self.assertEqual(plan.items[0].dest_index.fqn, "target.state")
        self.assertEqual(plan.items[0].storage_index.fqn, "source.state")

    def test_incremental_disables_plan_cache(self):
        """
        Feature: StandardSavePlanner incremental=True disables plan cache.
        Description: Configure planner with incremental=True.
        Expectation: get_cached returns None even after caching.
        """
        planner = StandardSavePlanner(enable_plan_caching=True)
        planner.configure_planner(
            {"w": torch.zeros(1)}, rank=0, use_collectives=True, incremental=True,
        )
        plan = SavePlan(items=[])
        metadata = Metadata(state_dict_metadata={})
        planner.cache_result(plan, metadata)
        cached = planner.get_cached()
        self.assertIsNone(cached)

    def test_full_save_plan_cache_not_regressed(self):
        """
        Feature: StandardSavePlanner full save still uses plan cache.
        Description: Configure planner without incremental flag.
        Expectation: Cache works as before (hit after caching).
        """
        planner = StandardSavePlanner(enable_plan_caching=True)
        planner.configure_planner(
            {"w": torch.zeros(1)}, rank=0, use_collectives=True,
        )
        plan = SavePlan(items=[])
        metadata = Metadata(state_dict_metadata={})
        planner.cache_result(plan, metadata)
        cached = planner.get_cached()
        self.assertIsNotNone(cached)
        self.assertIs(cached.final_plan, plan)


if __name__ == "__main__":
    unittest.main()
