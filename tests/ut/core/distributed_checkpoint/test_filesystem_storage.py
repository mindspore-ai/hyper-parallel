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
from typing import Any
from unittest.mock import Mock, patch

import torch
from safetensors import safe_open

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
from hyper_parallel.core.distributed_checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
)
from hyper_parallel.core.distributed_checkpoint.planner import (
    BroadcastSource,
    SavePlan,
    WriteItem,
    WriteItemType,
)
from hyper_parallel.core.distributed_checkpoint.storage import METADATA_FILE_NAME, StorageInfo
from hyper_parallel.core.dtensor.device_mesh import _DEVICE_MESH_MAP
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.core.dtensor.placement_types import RaggedShard
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


class _FakeBatcher:
    """Stands in for the batcher, noting the shards handed to it instead of sending any."""

    def __init__(self, events: list = None) -> None:
        """Note shards into ``events`` when given one, and count them either way."""
        self.events = events
        self.sent = 0
        self.batched = 0

    def add(self, in_flight: Any, state_dict: dict, item: Any) -> None:
        """Take one shard, as the real batcher does, without reaching a backend."""
        self.sent += 1
        if self.events is not None:
            self.events.append(f"send {item.dest_index.fqn}")

    def flush(self, in_flight: Any) -> None:
        """Send whatever is waiting, which for a stand-in is nothing."""


def _pairs(reader: Any, reqs: list, storage_data: dict, keys: Any = None) -> list:
    """What a read hands back, without a checkpoint file behind it: an item, and nothing read."""
    return [(req, None) for req in reqs]


class TestFilesystemStorage(unittest.TestCase):
    """Tests for filesystem checkpoint storage reader/writer."""

    def setUp(self) -> None:
        os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
        _platform_mod.platform = None
        importlib.reload(planner_mod)
        importlib.reload(fs_mod)
        planner_mod.StandardSavePlanner.cached_save_result.clear()

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
            global_plan, metadata = save_planner.build_global_plan([save_plan])
            final_plan = save_planner.finalize_plan(global_plan)

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

    def test_writer_assigns_unique_physical_keys_to_same_fqn_chunks(self):
        """Multiple logical chunks with one FQN remain distinct in safetensors."""
        def make_item(fqn, offset):
            return WriteItem(
                index=MetadataIndex(fqn=fqn, offset=offset),
                type=WriteItemType.TENSOR,
                tensor_data={
                    "chunk": ChunkStorageMetadata(offsets=offset, sizes=(1, 2)),
                },
            )

        items = [
            make_item("weight", (0, 0)),
            make_item("weight", (1, 0)),
            make_item("weight.__dcp_chunk_0", (0, 0)),
        ]
        planner = Mock()
        planner.get_data.side_effect = [
            torch.tensor([[1.0, 2.0]]),
            torch.tensor([[3.0, 4.0]]),
            torch.tensor([[5.0, 6.0]]),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = FileSystemWriter(tmpdir)
            writer.configure_writer(is_coordinator=True, rank=0)
            results = writer.execute_write(SavePlan(items=items), planner)

            physical_keys = [result.storage_data.tensor_key for result in results]
            self.assertEqual(len(set(physical_keys)), 3)
            self.assertEqual(physical_keys[2], "weight.__dcp_chunk_0")
            with safe_open(
                    str(Path(tmpdir) / "_rank0_.safetensors"),
                    framework="pt",
                    device="cpu",
            ) as tensor_file:
                self.assertEqual(set(tensor_file.keys()), set(physical_keys))

    def test_ragged_tensor_roundtrip_uses_nd_box_storage(self):
        """Save and load a rank-local RaggedShard through filesystem storage."""
        _DEVICE_MESH_MAP.clear()
        EXISTING_COMM_GROUPS.clear()
        with patch("hyper_parallel.core.dtensor.device_mesh.dist.get_rank",
                return_value=0,
        ):
            mesh = Layout((2,), ("ragged",), init_backend=False).mesh
            source = DTensor.from_local(
                torch.arange(48),
                mesh,
                (RaggedShard(dims=(0, 1), local_units=(1, 3)),),
                shape=(6, 4, 8),
            )
            with tempfile.TemporaryDirectory() as tmpdir:
                ckpt_dir = Path(tmpdir)
                from hyper_parallel.core.distributed_checkpoint.standard_planner import (
                    StandardLoadPlanner,
                    StandardSavePlanner,
                )

                save_planner = StandardSavePlanner(enable_plan_caching=False)
                save_planner.configure_planner(
                    {"weight": source}, rank=0, use_collectives=False
                )
                save_plan = save_planner.build_local_plan()
                global_plan, metadata = save_planner.build_global_plan([save_plan])
                writer = FileSystemWriter(ckpt_dir)
                writer.configure_writer(is_coordinator=True, rank=0, use_collectives=False)
                results = writer.execute_write(global_plan, save_planner)
                writer.finalize_checkpoint(metadata, [results])

                target = DTensor.from_local(
                    torch.zeros(48, dtype=torch.int64),
                    mesh,
                    (RaggedShard(dims=(0, 1), local_units=(1, 3)),),
                    shape=(6, 4, 8),
                )
                loaded_md = pickle.loads((ckpt_dir / f"0{METADATA_FILE_NAME}").read_bytes())
                load_planner = StandardLoadPlanner()
                load_planner.configure_planner(
                    {"weight": target}, loaded_md, rank=0, use_collectives=False
                )
                reader = FileSystemReader(ckpt_dir)
                reader.configure_reader(loaded_md, is_coordinator=True, rank=0)
                # The rank lookup happens on the shared ``platform`` object
                # imported from util; patch the method on it.
                with patch(
                    "hyper_parallel.core.distributed_checkpoint.util.platform.get_rank",
                    return_value=0,
                ):
                    load_plan = load_planner.build_local_plan()
                reader.execute_read(load_plan, load_planner)

                torch.testing.assert_close(target.to_local(), source.to_local())

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
        from hyper_parallel.core.distributed_checkpoint.planner import LoadItemType, ReadItem

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
        grouped = reader._group_items_by_file([read_item])
        self.assertEqual(len(grouped), 1)
        self.assertEqual(len(next(iter(grouped.values()))), 1)


    @staticmethod
    def _pipeline_reader(tmpdir, shards, files=1):
        """
        A reader over a checkpoint, with a plan holding the given shards.

        Args:
            tmpdir (str): Directory the checkpoint files are made in.
            shards (list): ``(fqn, source)`` per shard, source None when this rank alone
                wants it.
            files (int): Over how many files the shards are spread, one to a shard and round
                again. More than one is what the dedup leaves behind for the entries a load
                does not broadcast: the rank that wrote one is whichever was carrying the
                least at the time, so they sit wherever that put them.

        Returns:
            tuple: The reader and the load plan to hand it.
        """
        from hyper_parallel.core.distributed_checkpoint.planner import (
            LoadItemType, LoadPlan, ReadItem,
        )

        storage_data, items = {}, []
        for index, (fqn, source) in enumerate(shards):
            relative = f"_rank{index % files}_.safetensors"
            Path(tmpdir, relative).touch()
            metadata_index = MetadataIndex(fqn=fqn, offset=(index,), index=index)
            storage_data[metadata_index] = StorageInfo(relative_path=relative, offset=0, length=-1)
            items.append(ReadItem(
                type=LoadItemType.TENSOR,
                dest_index=metadata_index,
                dest_offsets=(0,),
                storage_index=metadata_index,
                storage_offsets=(0,),
                lengths=(2,),
                source=source,
            ))
        reader = FileSystemReader(tmpdir)
        reader.storage_data = storage_data
        return reader, LoadPlan(items=items)

    def test_execute_read_sends_each_shard_as_soon_as_it_is_in_place(self):
        """
        Feature: FileSystemReader.execute_read pipelining.
        Description: Two shards this rank reads on behalf of its group, one it receives from
            elsewhere, and one nobody else wants. What is put in place, and what is sent, are
            recorded as they happen.
        Expectation: Every shared shard is sent right after it lands and before the next one
            does, and the shard nobody shares comes last. That interleaving is what leaves a
            send in flight while the shard after it is being read; putting everything in
            place first would keep the group waiting through all of it, and taking the
            private shard early would delay every send behind it.
        """
        mine = BroadcastSource(group_ranks=(0, 1), src_rank=0)
        theirs = BroadcastSource(group_ranks=(0, 1), src_rank=1)
        events = []

        def record_apply(fetched: list, planner: Any) -> None:
            """Note which shards were put in place instead of copying anything."""
            events.extend(f"apply {req.dest_index.fqn}" for req, payload in fetched)

        with tempfile.TemporaryDirectory() as tmpdir:
            reader, plan = self._pipeline_reader(
                tmpdir, [("alone", None), ("a", mine), ("b", theirs), ("c", mine)]
            )
            files = fs_mod._OpenFiles(8, lambda path: path, lambda _reader: None)
            with patch.object(fs_mod, "_open_checkpoint_files", lambda: files), \
                    patch.object(fs_mod, "_fetch_tensor_file", _pairs), \
                    patch.object(fs_mod, "_apply_fetched", record_apply), \
                    patch.object(fs_mod, "BroadcastBatcher",
                                 lambda *_args: _FakeBatcher(events)), \
                    patch.object(fs_mod, "wait_broadcasts", lambda _in_flight: None):
                reader.execute_read(plan, Mock(), {(0, 1): "pre_built"})

        self.assertEqual(
            events, ["apply a", "send a", "send b", "apply c", "send c", "apply alone"]
        )

    def test_the_shards_nobody_shares_are_read_a_file_at_a_time(self):
        """
        Feature: execute_read gathering the shards of one file into a single read.
        Description: Thirty-six shards nobody else wants, spread over twelve files and
            landing in the plan a file apart, with the handle cache cut to one file so that
            nothing is held from one shard to the next.
        Expectation: Twelve reads and twelve opens, each read carrying the three shards of
            one file. Nothing waits on these shards, so nothing holds them to the order the
            ranks agreed on and they are gathered by file instead, which leaves the read
            independent of how much the cache happens to be holding. It is the pickled
            entries this matters to most: a load never broadcasts one, so every rank reads
            every entry it wants, out of whatever file the dedup put it in, and nothing
            holds a bytes file open between reads - eighty entries over three files cost
            eighty opens where three would do.
        """
        opened, reads = [], []

        def open_one(path: str) -> Any:
            """Stand in for opening a file, and count the opening."""
            opened.append(path)
            return path

        def record_read(reader: Any, reqs: list, storage_data: dict) -> list:
            """Note which shards one read asked its file for."""
            reads.append([req.dest_index.fqn for req in reqs])
            return _pairs(reader, reqs, storage_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            reader, plan = self._pipeline_reader(
                tmpdir, [(f"w{shard:02d}", None) for shard in range(36)], files=12)
            held = fs_mod._OpenFiles(1, open_one, lambda _reader: None)
            with patch.object(fs_mod, "_open_checkpoint_files", lambda: held), \
                    patch.object(fs_mod, "_fetch_tensor_file", record_read), \
                    patch.object(fs_mod, "_apply_fetched", lambda *_args: None), \
                    patch.object(fs_mod, "wait_broadcasts", lambda _in_flight: None):
                reader.execute_read(plan, Mock(), None)

        self.assertEqual([len(read) for read in reads], [3] * 12,
                         "a file was asked for its shards one at a time")
        self.assertEqual(len(opened), 12, "a file was opened again for a later shard of it")
        self.assertEqual(sorted(fqn for read in reads for fqn in read),
                         sorted(f"w{shard:02d}" for shard in range(36)),
                         "gathering by file left a shard unread")

    def test_a_read_of_a_file_that_is_not_there_says_which_one(self):
        """
        Feature: FileSystemReader.execute_read on a checkpoint file that is missing.
        Description: A plan naming a file the checkpoint directory does not hold, which is
            what a checkpoint half copied or half written leaves behind.
        Expectation: FileNotFoundError naming the file. Nothing looks for the file before
            opening it: safetensors raises that itself, and so does the builtin open behind
            a bytes file, so a check ahead of them said no more than they do and said it
            with a stat per file - a round trip of its own on shared storage.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            reader, plan = self._pipeline_reader(tmpdir, [("gone", None)])
            Path(tmpdir, "_rank0_.safetensors").unlink()
            with patch.object(fs_mod, "BroadcastBatcher", lambda *_args: _FakeBatcher()), \
                    patch.object(fs_mod, "wait_broadcasts", lambda _in_flight: None):
                with self.assertRaises(FileNotFoundError) as caught:
                    reader.execute_read(plan, Mock(), None)

        self.assertIn("_rank0_.safetensors", str(caught.exception))

    def test_a_read_that_fails_fails_the_load(self):
        """
        Feature: FileSystemReader.execute_read read failures.
        Description: A read that raises, as an unreadable checkpoint file would.
        Expectation: The load raises it rather than carrying on. Reads are pulled one shard
            at a time as the copies ask for them, so a failure has to come out of the pull
            that asked for it; one swallowed there would leave the load putting shards in
            place that were never read.
        """
        mine = BroadcastSource(group_ranks=(0, 1), src_rank=0)

        def fail(reader: Any, reqs: list, storage_data: dict, keys: Any = None) -> list:
            """A read that goes wrong, as an unreadable checkpoint file would."""
            raise RuntimeError("checkpoint file is not readable")

        with tempfile.TemporaryDirectory() as tmpdir:
            reader, plan = self._pipeline_reader(tmpdir, [(name, mine) for name in "ab"])
            files = fs_mod._OpenFiles(8, lambda path: path, lambda _reader: None)
            with patch.object(fs_mod, "_open_checkpoint_files", lambda: files), \
                    patch.object(fs_mod, "_fetch_tensor_file", fail), \
                    patch.object(fs_mod, "BroadcastBatcher", lambda *_args: _FakeBatcher()), \
                    patch.object(fs_mod, "wait_broadcasts", lambda _in_flight: None):
                with self.assertRaises(RuntimeError):
                    reader.execute_read(plan, Mock(), {(0, 1): "pre_built"})

    def test_execute_read_keeps_a_checkpoint_file_open_across_shards(self):
        """
        Feature: FileSystemReader.execute_read file handling.
        Description: Three shards of one checkpoint file, read one at a time by the pipeline.
        Expectation: The file is opened once. Going through the shards in the order every
            rank agrees on returns to the same file over and over, and opening one costs far
            more than the slice read it wraps, so reopening per shard would pay that over
            and over for nothing.
        """
        mine = BroadcastSource(group_ranks=(0, 1), src_rank=0)
        opened = []

        with tempfile.TemporaryDirectory() as tmpdir:
            reader, plan = self._pipeline_reader(tmpdir, [(name, mine) for name in ("a", "b", "c")])
            files = fs_mod._OpenFiles(8, lambda path: opened.append(path) or path, lambda _reader: None)
            with patch.object(fs_mod, "_open_checkpoint_files", lambda: files), \
                    patch.object(fs_mod, "_fetch_tensor_file", _pairs), \
                    patch.object(fs_mod, "_apply_fetched", lambda *_args: None), \
                    patch.object(fs_mod, "BroadcastBatcher", lambda *_args: _FakeBatcher()), \
                    patch.object(fs_mod, "wait_broadcasts", lambda _in_flight: None):
                reader.execute_read(plan, Mock(), {(0, 1): "pre_built"})

        self.assertEqual(len(opened), 1)


if __name__ == "__main__":

    unittest.main()
