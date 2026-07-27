# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""UT for :mod:`hyper_parallel.core.distributed_checkpoint.topology_mapper`."""
import unittest

from hyper_parallel.core.distributed_checkpoint.metadata import (
    ChunkStorageMetadata,
    MetadataIndex,
    TensorProperties,
    TensorStorageMetadata,
)
from hyper_parallel.core.distributed_checkpoint.planner import LoadItemType
from hyper_parallel.core.distributed_checkpoint.topology_mapper import TopologyMapper


class TestTopologyMapperInit(unittest.TestCase):
    """Tests for TopologyMapper construction and validation."""

    def test_default_mapper_is_identity(self):
        """
        Feature: TopologyMapper with no mapping.
        Description: map_fqn returns the input FQN unchanged.
        Expectation: Identity mapping for any FQN.
        """
        mapper = TopologyMapper()
        self.assertEqual(mapper.map_fqn("model.weight"), "model.weight")

    def test_explicit_mapping(self):
        """
        Feature: TopologyMapper with explicit mapping.
        Description: map_fqn returns checkpoint FQN for mapped keys.
        Expectation: Mapped FQNs are translated; unmapped FQNs are identity.
        """
        mapping = {
            "model.pp_stage_1.layers.0.weight": "model.pp_stage_0.layers.1.weight",
        }
        mapper = TopologyMapper(target_to_checkpoint_fqn=mapping)
        self.assertEqual(
            mapper.map_fqn("model.pp_stage_1.layers.0.weight"),
            "model.pp_stage_0.layers.1.weight",
        )
        self.assertEqual(
            mapper.map_fqn("model.pp_stage_0.layers.0.weight"),
            "model.pp_stage_0.layers.0.weight",
        )

    def test_mapping_is_deep_copied(self):
        """
        Feature: TopologyMapper copies the mapping dict.
        Description: Mutating the original dict after construction does not affect the mapper.
        Expectation: Mapper retains the original mapping.
        """
        mapping = {"a": "b"}
        mapper = TopologyMapper(target_to_checkpoint_fqn=mapping)
        mapping["a"] = "c"
        self.assertEqual(mapper.map_fqn("a"), "b")

    def test_invalid_key_raises(self):
        """
        Feature: TopologyMapper validates keys.
        Description: Non-string or empty key in mapping.
        Expectation: ValueError is raised.
        """
        with self.assertRaises(ValueError):
            TopologyMapper(target_to_checkpoint_fqn={"": "b"})
        with self.assertRaises(ValueError):
            TopologyMapper(target_to_checkpoint_fqn={123: "b"})

    def test_invalid_value_raises(self):
        """
        Feature: TopologyMapper validates values.
        Description: Non-string or empty value in mapping.
        Expectation: ValueError is raised.
        """
        with self.assertRaises(ValueError):
            TopologyMapper(target_to_checkpoint_fqn={"a": ""})
        with self.assertRaises(ValueError):
            TopologyMapper(target_to_checkpoint_fqn={"a": 123})

    def test_map_fqn_invalid_target_raises(self):
        """
        Feature: TopologyMapper.map_fqn validates input.
        Description: Non-string or empty target_fqn.
        Expectation: ValueError is raised.
        """
        mapper = TopologyMapper()
        with self.assertRaises(ValueError):
            mapper.map_fqn("")
        with self.assertRaises(ValueError):
            mapper.map_fqn(42)

    def test_multiple_targets_same_checkpoint_fqn(self):
        """
        Feature: TopologyMapper allows multiple target FQNs to map to the same checkpoint FQN.
        Description: Two target FQNs map to the same checkpoint FQN (shared parameters).
        Expectation: Both target FQNs resolve to the same checkpoint FQN.
        """
        mapping = {
            "model.pp_stage_1.layers.0.weight": "model.pp_stage_0.layers.0.weight",
            "model.pp_stage_2.layers.0.weight": "model.pp_stage_0.layers.0.weight",
        }
        mapper = TopologyMapper(target_to_checkpoint_fqn=mapping)
        self.assertEqual(
            mapper.map_fqn("model.pp_stage_1.layers.0.weight"),
            "model.pp_stage_0.layers.0.weight",
        )
        self.assertEqual(
            mapper.map_fqn("model.pp_stage_2.layers.0.weight"),
            "model.pp_stage_0.layers.0.weight",
        )


class TestTopologyMapperComputeRequiredShards(unittest.TestCase):
    """Tests for TopologyMapper.compute_required_shards."""

    def _make_md(
        self,
        global_shape: tuple[int, ...],
        chunks: list[tuple[tuple[int, ...], tuple[int, ...]]],
    ) -> TensorStorageMetadata:
        """Build TensorStorageMetadata from global shape and chunk specifications."""
        return TensorStorageMetadata(
            properties=TensorProperties(dtype="float32"),
            size=global_shape,
            chunks=[
                ChunkStorageMetadata(offsets=offsets, sizes=sizes)
                for offsets, sizes in chunks
            ],
        )

    def test_identity_mapping_preserves_fqn(self):
        """
        Feature: compute_required_shards with identity mapper.
        Description: Full tensor checkpoint, local chunk is first half.
        Expectation: ReadItem dest_index.fqn == storage_index.fqn == target_fqn.
        """
        mapper = TopologyMapper()
        checkpoint_md = self._make_md((8, 8), [((0, 0), (8, 8))])
        local_chunks = [ChunkStorageMetadata(offsets=(0, 0), sizes=(4, 8))]
        items = mapper.compute_required_shards("w", checkpoint_md, local_chunks)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].dest_index.fqn, "w")
        self.assertEqual(items[0].storage_index.fqn, "w")
        self.assertEqual(items[0].lengths, (4, 8))

    def test_pp_fqn_mapping_separates_dest_and_storage(self):
        """
        Feature: compute_required_shards with PP FQN mapping.
        Description: Target FQN differs from checkpoint FQN.
        Expectation: dest_index.fqn == target_fqn, storage_index.fqn == checkpoint_fqn.
        """
        mapping = {"target.weight": "source.weight"}
        mapper = TopologyMapper(target_to_checkpoint_fqn=mapping)
        checkpoint_md = self._make_md((4, 4), [((0, 0), (4, 4))])
        local_chunks = [ChunkStorageMetadata(offsets=(0, 0), sizes=(4, 4))]
        items = mapper.compute_required_shards("target.weight", checkpoint_md, local_chunks)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].dest_index.fqn, "target.weight")
        self.assertEqual(items[0].storage_index.fqn, "source.weight")

    def test_tp4_to_tp2_shrink(self):
        """
        Feature: compute_required_shards with TP4→TP2 (shrink).
        Description: Checkpoint has 4 TP shards; target needs 2 TP shards.
            Global shape (8,), saved chunks: [0,2), [2,4), [4,6), [6,8).
            Target local chunk for TP shard-0: [0,4).
        Expectation: Two ReadItems, one from each of the first two saved chunks.
        """
        mapper = TopologyMapper()
        checkpoint_md = self._make_md(
            (8,),
            [((0,), (2,)), ((2,), (2,)), ((4,), (2,)), ((6,), (2,))],
        )
        local_chunks = [ChunkStorageMetadata(offsets=(0,), sizes=(4,))]
        items = mapper.compute_required_shards("w", checkpoint_md, local_chunks)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].lengths, (2,))
        self.assertEqual(items[0].dest_offsets, (0,))
        self.assertEqual(items[0].storage_offsets, (0,))
        self.assertEqual(items[1].lengths, (2,))
        self.assertEqual(items[1].dest_offsets, (2,))
        self.assertEqual(items[1].storage_offsets, (0,))

    def test_tp2_to_tp4_expand(self):
        """
        Feature: compute_required_shards with TP2→TP4 (expand).
        Description: Checkpoint has 2 TP shards; target needs 4 TP shards.
            Global shape (8,), saved chunks: [0,4), [4,8).
            Target local chunk for TP shard-1: [2,4).
        Expectation: One ReadItem reading [2,4) from first saved chunk.
        """
        mapper = TopologyMapper()
        checkpoint_md = self._make_md(
            (8,),
            [((0,), (4,)), ((4,), (4,))],
        )
        local_chunks = [ChunkStorageMetadata(offsets=(2,), sizes=(2,))]
        items = mapper.compute_required_shards("w", checkpoint_md, local_chunks)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].lengths, (2,))
        self.assertEqual(items[0].dest_offsets, (0,))
        self.assertEqual(items[0].storage_offsets, (2,))

    def test_2d_tp4_to_tp2(self):
        """
        Feature: compute_required_shards with 2-D TP4→TP2.
        Description: Global shape (8, 8), checkpoint has 4 chunks along dim-1.
            Saved chunks: [0,0,8,2), [0,2,8,2), [0,4,8,2), [0,6,8,2).
            Target local chunk: [0,0,8,4).
        Expectation: Two ReadItems from the first two saved chunks.
        """
        mapper = TopologyMapper()
        checkpoint_md = self._make_md(
            (8, 8),
            [((0, 0), (8, 2)), ((0, 2), (8, 2)), ((0, 4), (8, 2)), ((0, 6), (8, 2))],
        )
        local_chunks = [ChunkStorageMetadata(offsets=(0, 0), sizes=(8, 4))]
        items = mapper.compute_required_shards("w", checkpoint_md, local_chunks)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].lengths, (8, 2))
        self.assertEqual(items[0].dest_offsets, (0, 0))
        self.assertEqual(items[1].lengths, (8, 2))
        self.assertEqual(items[1].dest_offsets, (0, 2))

    def test_no_intersection_raises(self):
        """
        Feature: compute_required_shards detects no overlap.
        Description: Local chunk is entirely outside saved chunks.
        Expectation: ValueError is raised indicating no intersection.
        """
        mapper = TopologyMapper()
        checkpoint_md = self._make_md((8,), [((0,), (4,))])
        local_chunks = [ChunkStorageMetadata(offsets=(4,), sizes=(4,))]
        with self.assertRaises(ValueError) as ctx:
            mapper.compute_required_shards("w", checkpoint_md, local_chunks)
        self.assertIn("no intersection", str(ctx.exception))

    def test_partial_coverage_raises(self):
        """
        Feature: compute_required_shards detects incomplete coverage.
        Description: Saved chunks only cover half of the local chunk.
        Expectation: ValueError is raised indicating not fully covered.
        """
        mapper = TopologyMapper()
        checkpoint_md = self._make_md((8,), [((0,), (2,))])
        local_chunks = [ChunkStorageMetadata(offsets=(0,), sizes=(4,))]
        with self.assertRaises(ValueError) as ctx:
            mapper.compute_required_shards("w", checkpoint_md, local_chunks)
        self.assertIn("not fully covered", str(ctx.exception))

    def test_empty_local_chunks_returns_empty(self):
        """
        Feature: compute_required_shards with empty local_chunks.
        Description: No local chunks needed.
        Expectation: Returns empty list.
        """
        mapper = TopologyMapper()
        checkpoint_md = self._make_md((4,), [((0,), (4,))])
        self.assertEqual(mapper.compute_required_shards("w", checkpoint_md, []), [])

    def test_empty_saved_chunks_returns_empty(self):
        """
        Feature: compute_required_shards with no saved chunks.
        Description: Checkpoint metadata has no chunks.
        Expectation: Returns empty list.
        """
        mapper = TopologyMapper()
        checkpoint_md = self._make_md((4,), [])
        local_chunks = [ChunkStorageMetadata(offsets=(0,), sizes=(4,))]
        self.assertEqual(mapper.compute_required_shards("w", checkpoint_md, local_chunks), [])

    def test_invalid_target_fqn_raises(self):
        """
        Feature: compute_required_shards validates target_fqn.
        Description: Empty or non-string target_fqn.
        Expectation: ValueError is raised.
        """
        mapper = TopologyMapper()
        checkpoint_md = self._make_md((4,), [((0,), (4,))])
        local_chunks = [ChunkStorageMetadata(offsets=(0,), sizes=(4,))]
        with self.assertRaises(ValueError):
            mapper.compute_required_shards("", checkpoint_md, local_chunks)
        with self.assertRaises(ValueError):
            mapper.compute_required_shards(123, checkpoint_md, local_chunks)

    def test_pp_mapping_with_tp_resharding(self):
        """
        Feature: compute_required_shards with PP FQN mapping + TP resharding.
        Description: Target FQN maps to a different checkpoint FQN, and TP size
            changes from 4 to 2. Global shape (8, 8).
            Saved chunks under "src.layers.0.weight": 4 TP shards along dim-1.
            Target local chunk for "dst.layers.0.weight": 2 TP shards along dim-1.
        Expectation: ReadItems have dest_index.fqn="dst.layers.0.weight" and
            storage_index.fqn="src.layers.0.weight" with correct offsets.
        """
        mapping = {"dst.layers.0.weight": "src.layers.0.weight"}
        mapper = TopologyMapper(target_to_checkpoint_fqn=mapping)
        checkpoint_md = self._make_md(
            (8, 8),
            [((0, 0), (8, 2)), ((0, 2), (8, 2)), ((0, 4), (8, 2)), ((0, 6), (8, 2))],
        )
        local_chunks = [ChunkStorageMetadata(offsets=(0, 0), sizes=(8, 4))]
        items = mapper.compute_required_shards("dst.layers.0.weight", checkpoint_md, local_chunks)
        self.assertEqual(len(items), 2)
        for item in items:
            self.assertEqual(item.dest_index.fqn, "dst.layers.0.weight")
            self.assertEqual(item.storage_index.fqn, "src.layers.0.weight")
        self.assertEqual(items[0].lengths, (8, 2))
        self.assertEqual(items[1].lengths, (8, 2))

    def test_replicated_checkpoint_to_tp_shard(self):
        """
        Feature: compute_required_shards with replicated checkpoint loaded into TP sharded.
        Description: Checkpoint has one full-sized chunk; target needs a TP shard.
        Expectation: One ReadItem with correct dest and storage offsets.
        """
        mapper = TopologyMapper()
        checkpoint_md = self._make_md((8, 8), [((0, 0), (8, 8))])
        local_chunks = [ChunkStorageMetadata(offsets=(0, 4), sizes=(8, 4))]
        items = mapper.compute_required_shards("w", checkpoint_md, local_chunks)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].dest_index.fqn, "w")
        self.assertEqual(items[0].storage_index.fqn, "w")
        self.assertEqual(items[0].dest_offsets, (0, 0))
        self.assertEqual(items[0].storage_offsets, (0, 4))
        self.assertEqual(items[0].lengths, (8, 4))

    def test_all_items_are_tensor_type(self):
        """
        Feature: compute_required_shards produces TENSOR ReadItems.
        Description: Normal resharding scenario.
        Expectation: All returned ReadItems have type TENSOR.
        """
        mapper = TopologyMapper()
        checkpoint_md = self._make_md((8,), [((0,), (8,))])
        local_chunks = [ChunkStorageMetadata(offsets=(0,), sizes=(4,))]
        items = mapper.compute_required_shards("w", checkpoint_md, local_chunks)
        for item in items:
            self.assertEqual(item.type, LoadItemType.TENSOR)

    def test_ep4_to_ep2_shrink(self):
        """
        Feature: compute_required_shards with EP4→EP2 (shrink).
        Description: Global shape (8, 16). Checkpoint saved with EP=4 has
            4 chunks along dim-0: [0,2), [2,4), [4,6), [6,8).
            Target with EP=2 needs chunk [0,4) for rank-0.
        Expectation: Two ReadItems from the first two saved chunks.
        """
        mapper = TopologyMapper()
        checkpoint_md = self._make_md(
            (8, 16),
            [((0, 0), (2, 16)), ((2, 0), (2, 16)), ((4, 0), (2, 16)), ((6, 0), (2, 16))],
        )
        local_chunks = [ChunkStorageMetadata(offsets=(0, 0), sizes=(4, 16))]
        items = mapper.compute_required_shards(
            "experts.w1", checkpoint_md, local_chunks
        )
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].lengths, (2, 16))
        self.assertEqual(items[0].dest_offsets, (0, 0))
        self.assertEqual(items[0].storage_offsets, (0, 0))
        self.assertEqual(items[1].lengths, (2, 16))
        self.assertEqual(items[1].dest_offsets, (2, 0))
        self.assertEqual(items[1].storage_offsets, (0, 0))
        for item in items:
            self.assertEqual(item.dest_index.fqn, "experts.w1")
            self.assertEqual(item.storage_index.fqn, "experts.w1")

    def test_ep4_to_ep2_shrink_rank1(self):
        """
        Feature: compute_required_shards with EP4→EP2, rank-1.
        Description: Global shape (8, 16). Checkpoint has 4 EP chunks.
            Target rank-1 with EP=2 needs chunk [4,8).
        Expectation: Two ReadItems from the last two saved chunks.
        """
        mapper = TopologyMapper()
        checkpoint_md = self._make_md(
            (8, 16),
            [((0, 0), (2, 16)), ((2, 0), (2, 16)), ((4, 0), (2, 16)), ((6, 0), (2, 16))],
        )
        local_chunks = [ChunkStorageMetadata(offsets=(4, 0), sizes=(4, 16))]
        items = mapper.compute_required_shards(
            "experts.w1", checkpoint_md, local_chunks
        )
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].lengths, (2, 16))
        self.assertEqual(items[0].dest_offsets, (0, 0))
        self.assertEqual(items[0].storage_offsets, (0, 0))
        self.assertEqual(items[1].lengths, (2, 16))
        self.assertEqual(items[1].dest_offsets, (2, 0))
        self.assertEqual(items[1].storage_offsets, (0, 0))

    def test_ep2_to_ep4_expand(self):
        """
        Feature: compute_required_shards with EP2→EP4 (expand).
        Description: Global shape (8, 16). Checkpoint saved with EP=2 has
            2 chunks along dim-0: [0,4), [4,8).
            Target with EP=4 needs chunk [2,4) for rank-1.
        Expectation: One ReadItem reading [2,4) from the first saved chunk.
        """
        mapper = TopologyMapper()
        checkpoint_md = self._make_md(
            (8, 16),
            [((0, 0), (4, 16)), ((4, 0), (4, 16))],
        )
        local_chunks = [ChunkStorageMetadata(offsets=(2, 0), sizes=(2, 16))]
        items = mapper.compute_required_shards(
            "experts.w1", checkpoint_md, local_chunks
        )
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].lengths, (2, 16))
        self.assertEqual(items[0].dest_offsets, (0, 0))
        self.assertEqual(items[0].storage_offsets, (2, 0))
        self.assertEqual(items[0].dest_index.fqn, "experts.w1")
        self.assertEqual(items[0].storage_index.fqn, "experts.w1")

    def test_ep2_to_ep4_expand_rank3(self):
        """
        Feature: compute_required_shards with EP2→EP4, rank-3.
        Description: Global shape (8, 16). Checkpoint has 2 EP chunks.
            Target rank-3 with EP=4 needs chunk [6,8).
        Expectation: One ReadItem reading [6,8) from the second saved chunk.
        """
        mapper = TopologyMapper()
        checkpoint_md = self._make_md(
            (8, 16),
            [((0, 0), (4, 16)), ((4, 0), (4, 16))],
        )
        local_chunks = [ChunkStorageMetadata(offsets=(6, 0), sizes=(2, 16))]
        items = mapper.compute_required_shards(
            "experts.w1", checkpoint_md, local_chunks
        )
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].lengths, (2, 16))
        self.assertEqual(items[0].dest_offsets, (0, 0))
        self.assertEqual(items[0].storage_offsets, (2, 0))

    def test_hsdp_shard_and_ep_combined_overlap(self):
        """
        Feature: compute_required_shards with HSDP shard + EP combined.
        Description: Global shape (8, 16). Checkpoint saved with HSDP(shard=2)
            x EP=2 on dim-0 produces 4 chunks:
            [0,2), [2,4), [4,6), [6,8) each of size (2,16).
            Target with EP=4 needs chunk [2,4) — one of the saved chunks exactly.
        Expectation: One ReadItem matching the exact saved chunk.
        """
        mapper = TopologyMapper()
        checkpoint_md = self._make_md(
            (8, 16),
            [((0, 0), (2, 16)), ((2, 0), (2, 16)), ((4, 0), (2, 16)), ((6, 0), (2, 16))],
        )
        local_chunks = [ChunkStorageMetadata(offsets=(2, 0), sizes=(2, 16))]
        items = mapper.compute_required_shards(
            "experts.w1", checkpoint_md, local_chunks
        )
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].lengths, (2, 16))
        self.assertEqual(items[0].dest_offsets, (0, 0))
        self.assertEqual(items[0].storage_offsets, (0, 0))

    def test_ep_with_tp_2d_overlap(self):
        """
        Feature: compute_required_shards with EP + TP combined.
        Description: Global shape (8, 16). Checkpoint saved with EP=2 (dim-0)
            x TP=2 (dim-1) produces 4 chunks:
            [(0,0),(4,8)], [(0,8),(4,8)], [(4,0),(4,8)], [(4,8),(4,8)].
            Target with EP=4 x TP=1 needs chunk [2,4) x [0,16) for EP-rank-1.
        Expectation: Two ReadItems from the first-row TP shards.
        """
        mapper = TopologyMapper()
        checkpoint_md = self._make_md(
            (8, 16),
            [((0, 0), (4, 8)), ((0, 8), (4, 8)), ((4, 0), (4, 8)), ((4, 8), (4, 8))],
        )
        local_chunks = [ChunkStorageMetadata(offsets=(2, 0), sizes=(2, 16))]
        items = mapper.compute_required_shards(
            "experts.w1", checkpoint_md, local_chunks
        )
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].lengths, (2, 8))
        self.assertEqual(items[0].dest_offsets, (0, 0))
        self.assertEqual(items[0].storage_offsets, (2, 0))
        self.assertEqual(items[1].lengths, (2, 8))
        self.assertEqual(items[1].dest_offsets, (0, 8))
        self.assertEqual(items[1].storage_offsets, (2, 0))

    def test_ep_resize_with_pp_fqn_mapping(self):
        """
        Feature: compute_required_shards with EP resize + PP FQN mapping.
        Description: EP2→EP4 resharding combined with target FQN mapping
            from "model.pp_stage_1.experts.w1" to "model.pp_stage_0.experts.w1".
        Expectation: ReadItems have dest_index.fqn = target FQN and
            storage_index.fqn = checkpoint FQN, with correct EP resize offsets.
        """
        mapping = {
            "model.pp_stage_1.experts.w1": "model.pp_stage_0.experts.w1",
        }
        mapper = TopologyMapper(target_to_checkpoint_fqn=mapping)
        checkpoint_md = self._make_md(
            (8, 16),
            [((0, 0), (4, 16)), ((4, 0), (4, 16))],
        )
        local_chunks = [ChunkStorageMetadata(offsets=(2, 0), sizes=(2, 16))]
        items = mapper.compute_required_shards(
            "model.pp_stage_1.experts.w1", checkpoint_md, local_chunks
        )
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].dest_index.fqn, "model.pp_stage_1.experts.w1")
        self.assertEqual(items[0].storage_index.fqn, "model.pp_stage_0.experts.w1")
        self.assertEqual(items[0].lengths, (2, 16))
        self.assertEqual(items[0].dest_offsets, (0, 0))
        self.assertEqual(items[0].storage_offsets, (2, 0))


if __name__ == "__main__":
    unittest.main()
