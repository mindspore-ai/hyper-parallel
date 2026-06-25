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
"""UT for :mod:`hyper_parallel.core.distributed_checkpoint.storage`."""
import unittest

from hyper_parallel.core.distributed_checkpoint.metadata import MetadataIndex
from hyper_parallel.core.distributed_checkpoint.storage import (
    METADATA_FILE_NAME,
    StorageInfo,
    WriteResult,
)


class TestStorage(unittest.TestCase):
    """Tests for storage dataclasses and constants."""

    def test_metadata_file_name_constant(self):
        """
        Feature: METADATA_FILE_NAME constant.
        Description: Read the module-level metadata filename constant.
        Expectation: Matches torch DCP convention (.metadata).
        """
        self.assertEqual(METADATA_FILE_NAME, ".metadata")

    def test_storage_info_fields(self):
        """
        Feature: StorageInfo records file location and byte range.
        Description: Create StorageInfo for a safetensors shard entry.
        Expectation: relative_path, offset, and length are stored.
        """
        info = StorageInfo(relative_path="_rank0_.safetensors", offset=0, length=-1)
        self.assertEqual(info.relative_path, "_rank0_.safetensors")
        self.assertEqual(info.offset, 0)
        self.assertEqual(info.length, -1)

    def test_write_result_links_index_and_storage(self):
        """
        Feature: WriteResult pairs MetadataIndex with StorageInfo.
        Description: Build WriteResult after a hypothetical write.
        Expectation: index and storage_data reference the same logical item.
        """
        index = MetadataIndex(fqn="weight", offset=(0, 0), index=0)
        storage = StorageInfo(relative_path="_rank0_.safetensors", offset=0, length=-1)
        result = WriteResult(index=index, storage_data=storage)
        self.assertEqual(result.index.fqn, "weight")
        self.assertEqual(result.storage_data.relative_path, "_rank0_.safetensors")


if __name__ == "__main__":
    unittest.main()
