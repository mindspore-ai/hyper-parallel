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
"""UT for :mod:`hyper_parallel.core.distributed_checkpoint.metadata`."""
import unittest

from hyper_parallel.core.distributed_checkpoint.metadata import (
    BytesStorageMetadata,
    ChunkInfo,
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    TensorProperties,
    TensorStorageMetadata,
)


class TestMetadata(unittest.TestCase):
    """Tests for checkpoint metadata dataclasses."""

    def test_metadata_index_defaults(self):
        """
        Feature: MetadataIndex construction.
        Description: Create MetadataIndex with FQN only.
        Expectation: offset is empty tuple and index is None.
        """
        idx = MetadataIndex(fqn="layer.weight")
        self.assertEqual(idx.fqn, "layer.weight")
        self.assertEqual(idx.offset, ())
        self.assertIsNone(idx.index)

    def test_chunk_storage_metadata_fields(self):
        """
        Feature: ChunkStorageMetadata records shard geometry.
        Description: Build chunk with offsets and sizes for a 2-D shard.
        Expectation: Fields match the supplied offsets and sizes tuples.
        """
        chunk = ChunkStorageMetadata(offsets=(0, 4), sizes=(8, 4))
        self.assertEqual(chunk.offsets, (0, 4))
        self.assertEqual(chunk.sizes, (8, 4))

    def test_chunk_info_wraps_chunk_and_global_shape(self):
        """
        Feature: ChunkInfo links local chunk to global tensor shape.
        Description: Wrap a ChunkStorageMetadata with a global shape tuple.
        Expectation: chunk and global_shape are stored on the dataclass.
        """
        chunk = ChunkStorageMetadata(offsets=(0,), sizes=(4,))
        info = ChunkInfo(chunk=chunk, global_shape=(8,))
        self.assertIs(info.chunk, chunk)
        self.assertEqual(info.global_shape, (8,))

    def test_tensor_properties_defaults(self):
        """
        Feature: TensorProperties optional fields.
        Description: Instantiate with dtype only.
        Expectation: requires_grad is False and memory_format is None.
        """
        props = TensorProperties(dtype="float32")
        self.assertEqual(props.dtype, "float32")
        self.assertFalse(props.requires_grad)
        self.assertIsNone(props.memory_format)

    def test_tensor_storage_metadata_default_chunks(self):
        """
        Feature: TensorStorageMetadata chunk list default.
        Description: Create metadata without explicit chunks.
        Expectation: chunks defaults to an empty list.
        """
        props = TensorProperties(dtype="float16")
        md = TensorStorageMetadata(properties=props, size=(2, 2))
        self.assertEqual(md.chunks, [])
        self.assertEqual(md.size, (2, 2))

    def test_metadata_version_and_optional_fields(self):
        """
        Feature: Global Metadata container defaults.
        Description: Build Metadata with one tensor entry.
        Expectation: version is 1.0; planner_data and storage_data default to None.
        """
        props = TensorProperties(dtype="float32")
        tensor_md = TensorStorageMetadata(properties=props, size=(4,))
        md = Metadata(state_dict_metadata={"w": tensor_md})
        self.assertEqual(md.version, "1.0")
        self.assertIsNone(md.planner_data)
        self.assertIsNone(md.storage_data)
        self.assertIn("w", md.state_dict_metadata)
        md.state_dict_metadata["meta"] = BytesStorageMetadata()
        self.assertIsInstance(md.state_dict_metadata["meta"], BytesStorageMetadata)


if __name__ == "__main__":
    unittest.main()
