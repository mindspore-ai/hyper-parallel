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
from hyper_parallel.core.distributed_checkpoint.planner import SavePlan
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

    def test_incremental_base_delta_roundtrip(self):
        """
        Feature: Incremental save writes only changed FQNs; delta loads completely.
        Description: Save a base checkpoint, then do an incremental save where
            only ``step`` changed. The delta directory should only contain a
            bytes file. Loading the delta should restore both weight and step.
        Expectation: Delta only writes changed FQN data; loaded state matches.
        """
        from hyper_parallel.core.distributed_checkpoint.standard_planner import StandardLoadPlanner, StandardSavePlanner

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "base"
            delta_dir = Path(tmpdir) / "delta"

            # --- Save base ---
            weight = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(3, 4))
            step = 7
            base_state = {"weight": weight, "step": step}
            save_planner = StandardSavePlanner(enable_plan_caching=False)
            save_planner.configure_planner(base_state, rank=0, use_collectives=False)
            save_plan = save_planner.build_local_plan()
            global_plans, metadata = save_planner.build_global_plan([save_plan])
            final_plan = save_planner.finalize_plan(global_plans[0])

            writer = FileSystemWriter(base_dir)
            writer.configure_writer(is_coordinator=True, rank=0, use_collectives=False)
            write_results = writer.execute_write(final_plan, save_planner)
            writer.finalize_checkpoint(metadata, [write_results])

            # --- Save delta (only step changed) ---
            new_step = 42
            delta_state = {"weight": weight, "step": new_step}
            delta_planner = StandardSavePlanner(enable_plan_caching=False)
            delta_planner.configure_planner(
                delta_state, rank=0, use_collectives=False, incremental=True,
            )
            delta_plan = delta_planner.build_local_plan()
            delta_global, delta_metadata = delta_planner.build_global_plan([delta_plan])
            delta_final = delta_planner.finalize_plan(delta_global[0])

            delta_writer = FileSystemWriter(
                delta_dir, incremental_from=base_dir, changed_fqns={"step"},
            )
            delta_writer.configure_writer(is_coordinator=True, rank=0, use_collectives=False)
            delta_opt_plan = delta_writer.optimize_local_plan(delta_final)
            delta_opt_plans = delta_writer.optimize_global_plan([delta_opt_plan])
            delta_write_results = delta_writer.execute_write(delta_opt_plans[0], delta_planner)
            delta_writer.finalize_checkpoint(delta_metadata, [delta_write_results])

            # Delta should not contain safetensors (weight is unchanged)
            safetensor_files = list(delta_dir.glob("*.safetensors"))
            self.assertEqual(len(safetensor_files), 0, "Delta should not write safetensors for unchanged tensors")

            # --- Load from delta ---
            load_state = {"weight": torch.zeros(3, 4), "step": None}
            load_planner = StandardLoadPlanner()
            loaded_md = pickle.loads((delta_dir / f"0{METADATA_FILE_NAME}").read_bytes())
            load_planner.configure_planner(load_state, loaded_md, rank=0, use_collectives=False)
            load_plan = load_planner.build_local_plan()

            reader = FileSystemReader(delta_dir)
            reader.configure_reader(loaded_md, is_coordinator=True, rank=0, use_collectives=False)
            reader.execute_read(load_plan, load_planner)

            torch.testing.assert_close(load_state["weight"], weight)
            self.assertEqual(load_state["step"], new_step)

    def test_incremental_unchanged_index_relocated(self):
        """
        Feature: Incremental save relocates baseline relative paths.
        Description: Save base then delta; check delta metadata has paths
            like ``../base/...`` for unchanged items.
        Expectation: Unchanged item StorageInfo.relative_path points to base dir.
        """
        from hyper_parallel.core.distributed_checkpoint.standard_planner import StandardSavePlanner

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "base"
            delta_dir = Path(tmpdir) / "delta"

            weight = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(3, 4))
            step = 7
            base_state = {"weight": weight, "step": step}
            save_planner = StandardSavePlanner(enable_plan_caching=False)
            save_planner.configure_planner(base_state, rank=0, use_collectives=False)
            save_plan = save_planner.build_local_plan()
            global_plans, metadata = save_planner.build_global_plan([save_plan])
            final_plan = save_planner.finalize_plan(global_plans[0])

            writer = FileSystemWriter(base_dir)
            writer.configure_writer(is_coordinator=True, rank=0, use_collectives=False)
            write_results = writer.execute_write(final_plan, save_planner)
            writer.finalize_checkpoint(metadata, [write_results])

            delta_state = {"weight": weight, "step": 42}
            delta_planner = StandardSavePlanner(enable_plan_caching=False)
            delta_planner.configure_planner(
                delta_state, rank=0, use_collectives=False, incremental=True,
            )
            delta_plan = delta_planner.build_local_plan()
            delta_global, delta_metadata = delta_planner.build_global_plan([delta_plan])
            delta_final = delta_planner.finalize_plan(delta_global[0])

            delta_writer = FileSystemWriter(
                delta_dir, incremental_from=base_dir, changed_fqns={"step"},
            )
            delta_writer.configure_writer(is_coordinator=True, rank=0, use_collectives=False)
            delta_opt_plan = delta_writer.optimize_local_plan(delta_final)
            delta_opt_plans = delta_writer.optimize_global_plan([delta_opt_plan])
            delta_write_results = delta_writer.execute_write(delta_opt_plans[0], delta_planner)
            delta_writer.finalize_checkpoint(delta_metadata, [delta_write_results])

            delta_md = pickle.loads((delta_dir / f"0{METADATA_FILE_NAME}").read_bytes())
            weight_indices = [idx for idx in delta_md.storage_data if idx.fqn == "weight"]
            self.assertTrue(len(weight_indices) > 0)
            for idx in weight_indices:
                rel_path = delta_md.storage_data[idx].relative_path
                self.assertIn("..", rel_path, "Unchanged weight path should reference baseline via ..")

    def test_incremental_empty_changed_fqns(self):
        """
        Feature: Incremental save with empty changed_fqns.
        Description: No FQNs changed, only metadata snapshot.
        Expectation: No tensor/bytes files in delta; delta metadata has complete index.
        """
        from hyper_parallel.core.distributed_checkpoint.standard_planner import StandardSavePlanner

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "base"
            delta_dir = Path(tmpdir) / "delta"

            weight = torch.nn.Parameter(torch.zeros(2, 2))
            step = 1
            base_state = {"weight": weight, "step": step}
            save_planner = StandardSavePlanner(enable_plan_caching=False)
            save_planner.configure_planner(base_state, rank=0, use_collectives=False)
            save_plan = save_planner.build_local_plan()
            global_plans, metadata = save_planner.build_global_plan([save_plan])
            final_plan = save_planner.finalize_plan(global_plans[0])

            writer = FileSystemWriter(base_dir)
            writer.configure_writer(is_coordinator=True, rank=0, use_collectives=False)
            write_results = writer.execute_write(final_plan, save_planner)
            writer.finalize_checkpoint(metadata, [write_results])

            delta_state = {"weight": weight, "step": step}
            delta_planner = StandardSavePlanner(enable_plan_caching=False)
            delta_planner.configure_planner(
                delta_state, rank=0, use_collectives=False, incremental=True,
            )
            delta_plan = delta_planner.build_local_plan()
            delta_global, delta_metadata = delta_planner.build_global_plan([delta_plan])
            delta_final = delta_planner.finalize_plan(delta_global[0])

            delta_writer = FileSystemWriter(
                delta_dir, incremental_from=base_dir, changed_fqns=set(),
            )
            delta_writer.configure_writer(is_coordinator=True, rank=0, use_collectives=False)
            delta_opt_plan = delta_writer.optimize_local_plan(delta_final)
            delta_opt_plans = delta_writer.optimize_global_plan([delta_opt_plan])
            delta_write_results = delta_writer.execute_write(delta_opt_plans[0], delta_planner)
            delta_writer.finalize_checkpoint(delta_metadata, [delta_write_results])

            safetensor_files = list(delta_dir.glob("*.safetensors"))
            bytes_files = list(delta_dir.glob("*.bytes"))
            self.assertEqual(len(safetensor_files), 0)
            self.assertEqual(len(bytes_files), 0)

            delta_md = pickle.loads((delta_dir / f"0{METADATA_FILE_NAME}").read_bytes())
            all_fqns = {idx.fqn for idx in delta_md.storage_data}
            self.assertIn("weight", all_fqns)
            self.assertIn("step", all_fqns)

    def test_incremental_new_fqn_must_be_changed(self):
        """
        Feature: Incremental save fails when new FQN is not in changed_fqns.
        Description: Delta state has a new FQN not in baseline, but not marked changed.
        Expectation: ValueError mentioning missing FQN.
        """
        from hyper_parallel.core.distributed_checkpoint.standard_planner import StandardSavePlanner

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "base"
            delta_dir = Path(tmpdir) / "delta"

            weight = torch.nn.Parameter(torch.zeros(2, 2))
            base_state = {"weight": weight}
            save_planner = StandardSavePlanner(enable_plan_caching=False)
            save_planner.configure_planner(base_state, rank=0, use_collectives=False)
            save_plan = save_planner.build_local_plan()
            global_plans, metadata = save_planner.build_global_plan([save_plan])
            final_plan = save_planner.finalize_plan(global_plans[0])

            writer = FileSystemWriter(base_dir)
            writer.configure_writer(is_coordinator=True, rank=0, use_collectives=False)
            write_results = writer.execute_write(final_plan, save_planner)
            writer.finalize_checkpoint(metadata, [write_results])

            bias = torch.zeros(2)
            delta_state = {"weight": weight, "bias": bias}
            delta_planner = StandardSavePlanner(enable_plan_caching=False)
            delta_planner.configure_planner(
                delta_state, rank=0, use_collectives=False, incremental=True,
            )
            delta_plan = delta_planner.build_local_plan()
            delta_global, delta_metadata = delta_planner.build_global_plan([delta_plan])
            delta_final = delta_planner.finalize_plan(delta_global[0])

            delta_writer = FileSystemWriter(
                delta_dir, incremental_from=base_dir, changed_fqns=set(),
            )
            delta_writer.configure_writer(is_coordinator=True, rank=0, use_collectives=False)
            delta_opt_plan = delta_writer.optimize_local_plan(delta_final)
            delta_opt_plans = delta_writer.optimize_global_plan([delta_opt_plan])
            delta_write_results = delta_writer.execute_write(delta_opt_plans[0], delta_planner)
            with self.assertRaises(ValueError) as ctx:
                delta_writer.finalize_checkpoint(delta_metadata, [delta_write_results])
            self.assertIn("bias", str(ctx.exception))

    def test_incremental_deleted_fqn_not_inherited(self):
        """
        Feature: Incremental save does not inherit deleted FQNs.
        Description: Delta state removes an FQN present in baseline.
        Expectation: Delta metadata does not contain the deleted FQN.
        """
        from hyper_parallel.core.distributed_checkpoint.standard_planner import StandardSavePlanner

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "base"
            delta_dir = Path(tmpdir) / "delta"

            weight = torch.nn.Parameter(torch.zeros(2, 2))
            bias = torch.zeros(2)
            base_state = {"weight": weight, "bias": bias}
            save_planner = StandardSavePlanner(enable_plan_caching=False)
            save_planner.configure_planner(base_state, rank=0, use_collectives=False)
            save_plan = save_planner.build_local_plan()
            global_plans, metadata = save_planner.build_global_plan([save_plan])
            final_plan = save_planner.finalize_plan(global_plans[0])

            writer = FileSystemWriter(base_dir)
            writer.configure_writer(is_coordinator=True, rank=0, use_collectives=False)
            write_results = writer.execute_write(final_plan, save_planner)
            writer.finalize_checkpoint(metadata, [write_results])

            delta_state = {"weight": weight}
            delta_planner = StandardSavePlanner(enable_plan_caching=False)
            delta_planner.configure_planner(
                delta_state, rank=0, use_collectives=False, incremental=True,
            )
            delta_plan = delta_planner.build_local_plan()
            delta_global, delta_metadata = delta_planner.build_global_plan([delta_plan])
            delta_final = delta_planner.finalize_plan(delta_global[0])

            delta_writer = FileSystemWriter(
                delta_dir, incremental_from=base_dir, changed_fqns=set(),
            )
            delta_writer.configure_writer(is_coordinator=True, rank=0, use_collectives=False)
            delta_opt_plan = delta_writer.optimize_local_plan(delta_final)
            delta_opt_plans = delta_writer.optimize_global_plan([delta_opt_plan])
            delta_write_results = delta_writer.execute_write(delta_opt_plans[0], delta_planner)
            delta_writer.finalize_checkpoint(delta_metadata, [delta_write_results])

            delta_md = pickle.loads((delta_dir / f"0{METADATA_FILE_NAME}").read_bytes())
            self.assertNotIn("bias", delta_md.state_dict_metadata)
            all_fqns = {idx.fqn for idx in delta_md.storage_data}
            self.assertNotIn("bias", all_fqns)

    def test_incremental_shape_mismatch_fails(self):
        """
        Feature: Incremental save validates unchanged tensor metadata.
        Description: Unchanged FQN has different shape in current vs baseline.
        Expectation: ValueError mentioning size mismatch.
        """
        from hyper_parallel.core.distributed_checkpoint.standard_planner import StandardSavePlanner

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "base"
            delta_dir = Path(tmpdir) / "delta"

            weight = torch.nn.Parameter(torch.zeros(2, 2))
            base_state = {"weight": weight, "step": 1}
            save_planner = StandardSavePlanner(enable_plan_caching=False)
            save_planner.configure_planner(base_state, rank=0, use_collectives=False)
            save_plan = save_planner.build_local_plan()
            global_plans, metadata = save_planner.build_global_plan([save_plan])
            final_plan = save_planner.finalize_plan(global_plans[0])

            writer = FileSystemWriter(base_dir)
            writer.configure_writer(is_coordinator=True, rank=0, use_collectives=False)
            write_results = writer.execute_write(final_plan, save_planner)
            writer.finalize_checkpoint(metadata, [write_results])

            new_weight = torch.nn.Parameter(torch.zeros(4, 4))
            delta_state = {"weight": new_weight, "step": 2}
            delta_planner = StandardSavePlanner(enable_plan_caching=False)
            delta_planner.configure_planner(
                delta_state, rank=0, use_collectives=False, incremental=True,
            )
            delta_plan = delta_planner.build_local_plan()
            delta_global, delta_metadata = delta_planner.build_global_plan([delta_plan])
            delta_final = delta_planner.finalize_plan(delta_global[0])

            delta_writer = FileSystemWriter(
                delta_dir, incremental_from=base_dir, changed_fqns={"step"},
            )
            delta_writer.configure_writer(is_coordinator=True, rank=0, use_collectives=False)
            delta_opt_plan = delta_writer.optimize_local_plan(delta_final)
            delta_opt_plans = delta_writer.optimize_global_plan([delta_opt_plan])
            delta_write_results = delta_writer.execute_write(delta_opt_plans[0], delta_planner)
            with self.assertRaises(ValueError) as ctx:
                delta_writer.finalize_checkpoint(delta_metadata, [delta_write_results])
            self.assertIn("size mismatch", str(ctx.exception))

    def test_incremental_baseline_missing_raises(self):
        """
        Feature: Incremental save with missing baseline directory.
        Description: incremental_from points to non-existent directory.
        Expectation: FileNotFoundError when loading baseline metadata.
        """
        from hyper_parallel.core.distributed_checkpoint.standard_planner import StandardSavePlanner

        with tempfile.TemporaryDirectory() as tmpdir:
            delta_dir = Path(tmpdir) / "delta"
            fake_base = Path(tmpdir) / "nonexistent"

            weight = torch.nn.Parameter(torch.zeros(2, 2))
            delta_state = {"weight": weight}
            delta_planner = StandardSavePlanner(enable_plan_caching=False)
            delta_planner.configure_planner(
                delta_state, rank=0, use_collectives=False, incremental=True,
            )
            delta_plan = delta_planner.build_local_plan()
            delta_global, delta_metadata = delta_planner.build_global_plan([delta_plan])
            delta_final = delta_planner.finalize_plan(delta_global[0])

            delta_writer = FileSystemWriter(
                delta_dir, incremental_from=fake_base, changed_fqns=set(),
            )
            delta_writer.configure_writer(is_coordinator=True, rank=0, use_collectives=False)
            delta_opt_plan = delta_writer.optimize_local_plan(delta_final)
            with self.assertRaises(FileNotFoundError):
                delta_writer.optimize_global_plan([delta_opt_plan])

    def test_incremental_ranks_disagree_on_changed_fqns(self):
        """
        Feature: Incremental save validates cross-rank changed_fqns consistency.
        Description: Two plans with different changed_fqns sets.
        Expectation: ValueError mentioning distinct sets.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            delta_dir = Path(tmpdir) / "delta"
            delta_writer = FileSystemWriter(
                delta_dir, incremental_from=Path("/fake"), changed_fqns={"a"},
            )
            delta_writer.configure_writer(is_coordinator=True, rank=0, use_collectives=False)

            plan_a = SavePlan(items=[], storage_data={"changed_fqns": frozenset({"a"})})
            plan_b = SavePlan(items=[], storage_data={"changed_fqns": frozenset({"b"})})
            with self.assertRaises(ValueError) as ctx:
                delta_writer.optimize_global_plan([plan_a, plan_b])
            self.assertIn("distinct sets", str(ctx.exception))

    def test_incremental_metadata_version_is_2_0(self):
        """
        Feature: Incremental save produces metadata with version 2.0.
        Description: Base and delta are saved; check delta metadata version.
        Expectation: Delta metadata version is "2.0".
        """
        from hyper_parallel.core.distributed_checkpoint.standard_planner import StandardSavePlanner

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "base"
            delta_dir = Path(tmpdir) / "delta"

            weight = torch.nn.Parameter(torch.zeros(2, 2))
            step = 1
            base_state = {"weight": weight, "step": step}
            save_planner = StandardSavePlanner(enable_plan_caching=False)
            save_planner.configure_planner(base_state, rank=0, use_collectives=False)
            save_plan = save_planner.build_local_plan()
            global_plans, metadata = save_planner.build_global_plan([save_plan])
            final_plan = save_planner.finalize_plan(global_plans[0])

            writer = FileSystemWriter(base_dir)
            writer.configure_writer(is_coordinator=True, rank=0, use_collectives=False)
            write_results = writer.execute_write(final_plan, save_planner)
            writer.finalize_checkpoint(metadata, [write_results])

            delta_state = {"weight": weight, "step": 2}
            delta_planner = StandardSavePlanner(enable_plan_caching=False)
            delta_planner.configure_planner(
                delta_state, rank=0, use_collectives=False, incremental=True,
            )
            delta_plan = delta_planner.build_local_plan()
            delta_global, delta_metadata = delta_planner.build_global_plan([delta_plan])
            delta_final = delta_planner.finalize_plan(delta_global[0])

            delta_writer = FileSystemWriter(
                delta_dir, incremental_from=base_dir, changed_fqns={"step"},
            )
            delta_writer.configure_writer(is_coordinator=True, rank=0, use_collectives=False)
            delta_opt_plan = delta_writer.optimize_local_plan(delta_final)
            delta_opt_plans = delta_writer.optimize_global_plan([delta_opt_plan])
            delta_write_results = delta_writer.execute_write(delta_opt_plans[0], delta_planner)
            delta_writer.finalize_checkpoint(delta_metadata, [delta_write_results])

            delta_md = pickle.loads((delta_dir / f"0{METADATA_FILE_NAME}").read_bytes())
            self.assertEqual(delta_md.version, "2.0")


if __name__ == "__main__":
    unittest.main()
