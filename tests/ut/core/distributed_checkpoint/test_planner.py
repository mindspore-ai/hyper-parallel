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
"""UT for :mod:`hyper_parallel.core.distributed_checkpoint.planner`."""
import pickle
import unittest

from hyper_parallel.core.distributed_checkpoint.metadata import (
    ChunkStorageMetadata,
    MetadataIndex,
    TensorProperties,
)
from hyper_parallel.core.distributed_checkpoint.planner import (
    LoadItemType,
    LoadPlan,
    ReadItem,
    SavePlan,
    WriteItem,
    WriteItemType,
)


class TestPlanner(unittest.TestCase):
    """Tests for planner dataclasses and helpers."""

    def test_write_item_type_values(self):
        """
        Feature: WriteItemType enum values.
        Description: Compare enum members to expected string values.
        Expectation: TENSOR and BYTE_IO match torch-aligned identifiers.
        """
        self.assertEqual(WriteItemType.TENSOR.value, "tensor")
        self.assertEqual(WriteItemType.BYTE_IO.value, "byte_io")

    def test_write_item_tensor_storage_size_non_tensor(self):
        """
        Feature: WriteItem.tensor_storage_size for non-tensor items.
        Description: Call tensor_storage_size on a BYTE_IO WriteItem.
        Expectation: Returns None because the item is not a tensor write.
        """
        item = WriteItem(
            index=MetadataIndex(fqn="opt"),
            type=WriteItemType.BYTE_IO,
        )
        self.assertIsNone(item.tensor_storage_size())

    def test_write_item_tensor_storage_size_missing_tensor_data(self):
        """
        Feature: WriteItem.tensor_storage_size without tensor_data.
        Description: TENSOR WriteItem with tensor_data left as None.
        Expectation: Returns None when chunk/properties are unavailable.
        """
        item = WriteItem(
            index=MetadataIndex(fqn="w"),
            type=WriteItemType.TENSOR,
        )
        self.assertIsNone(item.tensor_storage_size())

    def test_write_item_tensor_storage_size_float32(self):
        """
        Feature: WriteItem.tensor_storage_size estimation.
        Description: TENSOR item with chunk sizes (2, 4) and float32 dtype.
        Expectation: Estimated bytes equal 2 * 4 * 4 = 32.
        """
        chunk = ChunkStorageMetadata(offsets=(0, 0), sizes=(2, 4))
        props = TensorProperties(dtype="float32")
        item = WriteItem(
            index=MetadataIndex(fqn="w"),
            type=WriteItemType.TENSOR,
            tensor_data={"chunk": chunk, "properties": props, "size": (2, 4)},
        )
        self.assertEqual(item.tensor_storage_size(), 32)

    def test_write_item_tensor_storage_size_unknown_dtype(self):
        """
        Feature: WriteItem.tensor_storage_size fallback element size.
        Description: TENSOR item with unrecognized dtype string.
        Expectation: Uses default 4-byte element size (numel * 4).
        """
        chunk = ChunkStorageMetadata(offsets=(0,), sizes=(3,))
        props = TensorProperties(dtype="custom_dtype")
        item = WriteItem(
            index=MetadataIndex(fqn="w"),
            type=WriteItemType.TENSOR,
            tensor_data={"chunk": chunk, "properties": props, "size": (3,)},
        )
        self.assertEqual(item.tensor_storage_size(), 12)

    def test_save_plan_and_load_plan_defaults(self):
        """
        Feature: SavePlan and LoadPlan default containers.
        Description: Instantiate empty plans.
        Expectation: items default to empty lists; optional fields are None.
        """
        save_plan = SavePlan()
        load_plan = LoadPlan()
        self.assertEqual(save_plan.items, [])
        self.assertIsNone(save_plan.storage_data)
        self.assertIsNone(save_plan.planner_data)
        self.assertEqual(load_plan.items, [])

    def test_read_item_fields(self):
        """
        Feature: ReadItem records resharding copy geometry.
        Description: Build a TENSOR ReadItem with offsets and lengths.
        Expectation: All index/offset/length fields are preserved.
        """
        read_item = ReadItem(
            type=LoadItemType.TENSOR,
            dest_index=MetadataIndex(fqn="w", offset=(0, 0), index=0),
            dest_offsets=(0, 0),
            storage_index=MetadataIndex(fqn="w", offset=(0, 0), index=0),
            storage_offsets=(0, 0),
            lengths=(2, 4),
        )
        self.assertEqual(read_item.lengths, (2, 4))
        self.assertEqual(read_item.type, LoadItemType.TENSOR)


class TestLoadPlanIdentity(unittest.TestCase):
    """Tests for the stripped plan that travels through the all-gather."""

    @staticmethod
    def _plan():
        """A load plan with one tensor item and one byte-io item."""
        return LoadPlan(items=[
            ReadItem(
                type=LoadItemType.TENSOR,
                dest_index=MetadataIndex(fqn="w", offset=(4, 0), index=1),
                dest_offsets=(0, 2),
                storage_index=MetadataIndex(fqn="w", offset=(0, 0), index=3),
                storage_offsets=(4, 2),
                lengths=(2, 4),
            ),
            ReadItem(
                type=LoadItemType.BYTE_IO,
                dest_index=MetadataIndex(fqn="opt"),
                dest_offsets=(0,),
                storage_index=MetadataIndex(fqn="opt"),
                storage_offsets=(0,),
                lengths=(0,),
            ),
        ])

    def test_identity_keeps_what_recognizes_a_read_and_drops_the_rest(self):
        """
        Feature: LoadPlan.identity.
        Description: Strip a plan holding a tensor item and a byte-io item.
        Expectation: Type, destination and lengths survive untouched, so the global plan can
            still tell which shard a read lands in and how much of it it moves, while the
            checkpoint location is elided.
        """
        original = self._plan()

        stripped = original.identity()

        for before, after in zip(original.items, stripped.items):
            self.assertEqual(after.type, before.type)
            self.assertEqual(after.dest_index, before.dest_index)
            self.assertEqual(after.lengths, before.lengths)
            self.assertEqual(after.storage_offsets, ())
            self.assertNotEqual(after.storage_index, before.storage_index)

    def test_identity_leaves_the_original_plan_alone(self):
        """
        Feature: LoadPlan.identity.
        Description: Strip a plan and then look at the plan it was taken from.
        Expectation: Unchanged. The rank that sends the stripped copy goes on to execute the
            original, which still has to say where in the checkpoint to read from.
        """
        original = self._plan()

        original.identity()

        self.assertEqual(original.items[0].storage_offsets, (4, 2))
        self.assertEqual(original.items[0].storage_index, MetadataIndex(fqn="w", offset=(0, 0), index=3))

    def test_identity_is_smaller_on_the_wire(self):
        """
        Feature: LoadPlan.identity.
        Description: Pickle a plan of a hundred tensor items both ways, the way the
            all-gather does.
        Expectation: The stripped plan is markedly smaller. Every rank sends its plan to
            every other one, so this is the cost that grows with the world size.
        """
        items = [
            ReadItem(
                type=LoadItemType.TENSOR,
                dest_index=MetadataIndex(fqn=f"layers.{index}.weight", offset=(index, 0), index=0),
                dest_offsets=(0, 0),
                storage_index=MetadataIndex(fqn=f"layers.{index}.weight", offset=(0, 0), index=index),
                storage_offsets=(index, 0),
                lengths=(2, 4),
            )
            for index in range(100)
        ]
        plan = LoadPlan(items=items)

        full = len(pickle.dumps(plan))
        stripped = len(pickle.dumps(plan.identity()))

        self.assertLess(stripped, full * 0.75)


if __name__ == "__main__":

    unittest.main()
