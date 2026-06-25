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
"""UT for :mod:`hyper_parallel.core.distributed_checkpoint.filesystem_storage`."""
# pylint: disable=wrong-import-position
import importlib
import os
import pickle
import tempfile
import unittest
from pathlib import Path

import torch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
import hyper_parallel.platform.platform as _platform_mod

_platform_mod.platform = None

import hyper_parallel.core.distributed_checkpoint.filesystem_storage as fs_mod
import hyper_parallel.core.distributed_checkpoint.standard_planner as planner_mod

importlib.reload(planner_mod)
importlib.reload(fs_mod)

from hyper_parallel.core.distributed_checkpoint.filesystem_storage import (
    FileSystemReader,
    FileSystemWriter,
    _get_tensor_size,
)
from hyper_parallel.core.distributed_checkpoint.metadata import Metadata, MetadataIndex
from hyper_parallel.core.distributed_checkpoint.storage import METADATA_FILE_NAME, StorageInfo


class TestFilesystemStorage(unittest.TestCase):
    """Tests for filesystem checkpoint storage reader/writer."""

    def setUp(self) -> None:
        os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
        _platform_mod.platform = None
        importlib.reload(planner_mod)
        importlib.reload(fs_mod)
        planner_mod.StandardSavePlanner._cached_save_result.clear()

    def test_get_tensor_size_torch_tensor(self):
        """
        Feature: _get_tensor_size helper.
        Description: Pass a torch tensor with shape attribute.
        Expectation: Returns tuple shape.
        """
        tensor = torch.zeros(3, 5)
        self.assertEqual(_get_tensor_size(tensor), (3, 5))

    def test_filesystem_writer_reader_tensor_roundtrip(self):
        """
        Feature: FileSystemWriter and FileSystemReader tensor I/O.
        Description: Write one rank's tensor shard then read back via execute_read.
        Expectation: Loaded state_dict tensor matches saved values.
        """
        from hyper_parallel.core.distributed_checkpoint.standard_planner import StandardLoadPlanner, StandardSavePlanner

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir)
            weight = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(3, 4))
            save_state = {"weight": weight}
            save_planner = StandardSavePlanner(enable_plan_caching=False)
            save_planner.configure_planner(save_state, rank=0, use_collectives=False)
            save_plan = save_planner.build_local_plan()
            global_plans, metadata = save_planner.build_global_plan([save_plan])
            final_plan = save_planner.finalize_plan(global_plans[0])

            writer = FileSystemWriter(ckpt_dir)
            writer.configure_writer(is_coordinator=True, rank=0, use_collectives=False)
            write_results = writer.execute_write(final_plan, save_planner)
            writer.finalize_checkpoint(metadata, [write_results])

            metadata_path = ckpt_dir / f"0{METADATA_FILE_NAME}"
            self.assertTrue(metadata_path.exists())

            load_state = {"weight": torch.zeros(3, 4)}
            load_planner = StandardLoadPlanner()
            loaded_md = pickle.loads(metadata_path.read_bytes())
            load_planner.configure_planner(load_state, loaded_md, rank=0, use_collectives=False)
            load_plan = load_planner.build_local_plan()

            reader = FileSystemReader(ckpt_dir)
            reader.configure_reader(loaded_md, is_coordinator=True, rank=0, use_collectives=False)
            reader.execute_read(load_plan, load_planner)

            torch.testing.assert_close(load_state["weight"], weight)

    def test_filesystem_reader_load_metadata_rank_local(self):
        """
        Feature: FileSystemReader.load_metadata rank-local path.
        Description: Write pickled metadata to .rank{rank}_metadata filename pattern.
        Expectation: load_metadata(rank=0) returns the same Metadata object.
        """
        from hyper_parallel.core.distributed_checkpoint.metadata import TensorProperties, TensorStorageMetadata

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir)
            md = Metadata(
                state_dict_metadata={
                    "w": TensorStorageMetadata(
                        properties=TensorProperties(dtype="float32"),
                        size=(2, 2),
                    )
                }
            )
            md_path = ckpt_dir / f"0{METADATA_FILE_NAME}"
            with open(md_path, "wb") as f:
                pickle.dump(md, f)

            reader = FileSystemReader(ckpt_dir)
            loaded = reader.load_metadata(rank=0)
            self.assertEqual(loaded.state_dict_metadata["w"].size, (2, 2))

    def test_filesystem_reader_missing_metadata_raises(self):
        """
        Feature: FileSystemReader.load_metadata error handling.
        Description: Reader points at empty directory without metadata files.
        Expectation: FileNotFoundError mentions the expected metadata path.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            reader = FileSystemReader(tmpdir)
            with self.assertRaises(FileNotFoundError) as ctx:
                reader.load_metadata()
            self.assertIn(METADATA_FILE_NAME, str(ctx.exception))

    def test_filesystem_reader_group_items_by_storage_path(self):
        """
        Feature: FileSystemReader._group_items_by_file.
        Description: Load plan with ReadItems referencing the same safetensors file.
        Expectation: Items are grouped under one absolute file path key.
        """
        from hyper_parallel.core.distributed_checkpoint.planner import LoadItemType, LoadPlan, ReadItem

        storage_index = MetadataIndex(fqn="w", offset=(0, 0), index=0)
        storage_info = StorageInfo(relative_path="_rank0_.safetensors", offset=0, length=-1)
        md = Metadata(state_dict_metadata={}, storage_data={storage_index: storage_info})
        reader = FileSystemReader("/tmp/unused")
        reader.storage_data = md.storage_data

        read_item = ReadItem(
            type=LoadItemType.TENSOR,
            dest_index=storage_index,
            dest_offsets=(0, 0),
            storage_index=storage_index,
            storage_offsets=(0, 0),
            lengths=(2, 2),
        )
        grouped = reader._group_items_by_file(LoadPlan(items=[read_item]))
        self.assertEqual(len(grouped), 1)
        self.assertEqual(len(next(iter(grouped.values()))), 1)


if __name__ == "__main__":
    unittest.main()
