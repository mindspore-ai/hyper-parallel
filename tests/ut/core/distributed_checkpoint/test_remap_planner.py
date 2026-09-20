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
"""UT for :mod:`hyper_parallel.core.distributed_checkpoint.remap_planner`."""
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from hyper_parallel.core.distributed_checkpoint.api import load
from hyper_parallel.core.distributed_checkpoint.hf_storage import HuggingFaceStorageReader
from hyper_parallel.core.distributed_checkpoint.metadata import (
    CHUNK_INFO,
    ChunkInfo,
    ChunkStorageMetadata,
    MetadataIndex,
)
from hyper_parallel.core.distributed_checkpoint.remap_planner import (
    DeferredRead,
    RemapBlock,
    RemapLoadPlanner,
)


def _temporary_directory(test_case: unittest.TestCase) -> Path:
    """A fresh directory, removed once ``test_case`` is done."""
    path = Path(tempfile.mkdtemp(prefix="test_remap_planner_"))
    test_case.addCleanup(shutil.rmtree, path, ignore_errors=True)
    return path


def _checkpoint(test_case: unittest.TestCase, tensors: dict[str, torch.Tensor]) -> Path:
    """A Hugging Face checkpoint directory holding ``tensors`` in model.safetensors."""
    checkpoint_dir = _temporary_directory(test_case)
    save_file(tensors, str(checkpoint_dir / "model.safetensors"))
    return checkpoint_dir


def _read_into(checkpoint_dir: Path, planner: RemapLoadPlanner, state_dict: dict[str, Any]) -> None:
    """Plan and read ``state_dict`` through ``planner`` as a single rank, with no process group."""
    reader = HuggingFaceStorageReader(checkpoint_dir)
    metadata = reader.load_metadata()
    planner.configure_planner(state_dict, metadata, rank=0)
    reader.configure_reader(metadata, is_coordinator=True, rank=0)
    reader.execute_read(planner.build_local_plan(), planner)


def _rows(source: str, start_row: int, rows: int, columns: int, dest_row: int = 0) -> RemapBlock:
    """A block copying ``rows`` whole rows of a matrix from ``start_row`` to ``dest_row``."""
    return RemapBlock(
        offsets=(dest_row, 0), lengths=(rows, columns), source=source, base=(start_row, 0), coeff=((1, 0), (0, 1))
    )


class TestRemapLoadPlanner(unittest.TestCase):
    """Tests for loading checkpoint tensors laid out differently from the state dict."""

    def test_blocks_fill_one_tensor_from_two_checkpoint_tensors(self):
        """
        Feature: RemapLoadPlanner with blocks read straight into place.
        Description: Fill a 6x3 tensor with the rows of a 4x3 "q" followed by those of a 2x3 "k".
        Expectation: The tensor equals torch.cat of the two, converted to the dtype of the state dict.
        """
        q = torch.arange(12, dtype=torch.bfloat16).reshape(4, 3)
        k = torch.arange(100, 106, dtype=torch.bfloat16).reshape(2, 3)
        checkpoint_dir = _checkpoint(self, {"q": q, "k": k})
        fused = torch.zeros(6, 3)
        planner = RemapLoadPlanner({"qk": [_rows("q", 0, 4, 3), _rows("k", 0, 2, 3, dest_row=4)]})

        _read_into(checkpoint_dir, planner, {"qk": fused})

        expected = torch.cat([q, k]).float()
        self.assertTrue(torch.equal(fused, expected), f"fused mismatch: expected={expected}, got={fused}")

    def test_transposed_block_is_copied_straight_into_place(self):
        """
        Feature: RemapLoadPlanner with a block that reorders dimensions.
        Description: Fill a 3x2 tensor from a 2x3 checkpoint tensor, destination rows walking checkpoint
            columns, and ask the planner where the one read lands.
        Expectation: The tensor equals the transpose of the checkpoint tensor, and the read lands in a
            view of the destination rather than in a buffer of its own.
        """
        weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        checkpoint_dir = _checkpoint(self, {"w": weight})
        target = torch.zeros(3, 2)
        block = RemapBlock(offsets=(0, 0), lengths=(3, 2), source="w", base=(0, 0), coeff=((0, 1), (1, 0)))
        planner = RemapLoadPlanner({"t": [block]})

        _read_into(checkpoint_dir, planner, {"t": target})

        self.assertTrue(torch.equal(target, weight.t()), f"transpose mismatch: expected={weight.t()}, got={target}")
        item = planner.build_local_plan().items[0]
        acquired = planner.acquire_tensor(item)
        self.assertIsInstance(acquired, torch.Tensor, f"expected a view of the destination, got {type(acquired)}")
        self.assertEqual(acquired.data_ptr(), target.data_ptr(), "the view does not write into the destination")

    def test_gathered_block_applies_post_operations_in_checkpoint_dtype(self):
        """
        Feature: RemapLoadPlanner with a block that skips elements and has post operations.
        Description: Fill a float32 vector with every other element of a bfloat16 checkpoint vector
            plus one, the addition given as a post operation.
        Expectation: The vector equals the elements added to in bfloat16 and then converted, as a
            conversion run on the checkpoint tensor would compute them.
        """
        source = torch.linspace(-3, 3, 10).to(torch.bfloat16)
        checkpoint_dir = _checkpoint(self, {"v": source})
        target = torch.zeros(5)
        block = RemapBlock(offsets=(0,), lengths=(5,), source="v", base=(0,), coeff=((2,),),
                           post=(lambda tensor: tensor + 1.0,))
        planner = RemapLoadPlanner({"v": [block]})

        _read_into(checkpoint_dir, planner, {"v": target})

        expected = (source[::2] + 1.0).float()
        self.assertTrue(torch.equal(target, expected), f"gather mismatch: expected={expected}, got={target}")

    def test_blocks_are_cut_to_the_shard_the_state_dict_holds(self):
        """
        Feature: RemapLoadPlanner on a tensor holding one shard of the whole.
        Description: Hold rows 3-5 of the 6x3 concatenation of a 4x3 "q" and a 2x3 "k", which straddle
            both blocks.
        Expectation: The shard equals those rows of the concatenation, and each block is read only
            where it overlaps the shard.
        """
        q = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        k = torch.arange(100, 106, dtype=torch.float32).reshape(2, 3)
        checkpoint_dir = _checkpoint(self, {"q": q, "k": k})
        shard = torch.zeros(3, 3)
        chunk = ChunkStorageMetadata(offsets=(3, 0), sizes=(3, 3))
        setattr(shard, CHUNK_INFO, ChunkInfo(chunk=chunk, global_shape=(6, 3)))
        planner = RemapLoadPlanner({"qk": [_rows("q", 0, 4, 3), _rows("k", 0, 2, 3, dest_row=4)]})

        _read_into(checkpoint_dir, planner, {"qk": shard})

        expected = torch.cat([q, k])[3:]
        self.assertTrue(torch.equal(shard, expected), f"shard mismatch: expected={expected}, got={shard}")
        reads = sorted((item.storage_index.fqn, item.storage_offsets, item.lengths)
                       for item in planner.build_local_plan().items)
        expected_reads = [("k", (0, 0), (2, 3)), ("q", (3, 0), (1, 3))]
        self.assertEqual(reads, expected_reads, f"reads mismatch: expected={expected_reads}, got={reads}")

    def test_deferred_read_hands_whole_tensors_to_its_completion(self):
        """
        Feature: RemapLoadPlanner deferred reads.
        Description: Beside a table entry, defer a read of "a" and "b" whose completion records what
            it is given.
        Expectation: The completion runs once, with both checkpoint tensors whole, and the table entry
            is still filled.
        """
        a, b = torch.arange(4, dtype=torch.float32), torch.ones(2, 2)
        checkpoint_dir = _checkpoint(self, {"a": a, "b": b})
        received = []
        target = torch.zeros(2, 2)
        planner = RemapLoadPlanner(
            {"b": [_rows("b", 0, 2, 2)]},
            deferred=[DeferredRead(("a", "b"), received.append)],
        )

        _read_into(checkpoint_dir, planner, {"b": target})

        self.assertEqual(len(received), 1, f"expected one completion, got {len(received)}")
        self.assertEqual(sorted(received[0]), ["a", "b"], f"sources mismatch: got {sorted(received[0])}")
        self.assertTrue(torch.equal(received[0]["a"], a), f"a mismatch: expected={a}, got={received[0]['a']}")
        self.assertTrue(torch.equal(received[0]["b"], b), f"b mismatch: expected={b}, got={received[0]['b']}")
        self.assertTrue(torch.equal(target, b), f"table entry mismatch: expected={b}, got={target}")

    def test_block_reading_past_its_checkpoint_tensor_is_rejected(self):
        """
        Feature: RemapLoadPlanner.build_local_plan bounds check.
        Description: Plan a block reading three rows of a checkpoint tensor that has two, and a block
            naming a tensor the checkpoint does not have.
        Expectation: ValueError naming the checkpoint tensor both times.
        """
        checkpoint_dir = _checkpoint(self, {"w": torch.zeros(2, 3)})
        cases = {"w": _rows("w", 0, 3, 3), "missing": _rows("missing", 0, 2, 3)}
        for source, block in cases.items():
            with self.subTest(source):
                planner = RemapLoadPlanner({"t": [block]})
                with self.assertRaises(ValueError) as ctx:
                    _read_into(checkpoint_dir, planner, {"t": torch.zeros(3, 3)})
                self.assertIn(repr(source), str(ctx.exception), f"{source!r} not named in: {ctx.exception}")

    def test_malformed_blocks_and_reads_are_rejected(self):
        """
        Feature: RemapBlock and DeferredRead validation.
        Description: Build a block with fewer lengths than offsets, one stepping backwards, and a
            deferred read naming a source twice.
        Expectation: ValueError for each.
        """
        with self.assertRaises(ValueError):
            RemapBlock(offsets=(0, 0), lengths=(2,), source="w", base=(0,), coeff=((1, 0),))
        with self.assertRaises(ValueError):
            RemapBlock(offsets=(0,), lengths=(2,), source="w", base=(4,), coeff=((-1,),))
        with self.assertRaises(ValueError):
            DeferredRead(("a", "a"), print)

    def test_replicated_shards_are_weighed_by_the_state_dict_dtype(self):
        """
        Feature: RemapLoadPlanner._shard_element_size.
        Description: Configure the planner with a float16 entry, and ask for it and for a deferred read.
        Expectation: Two bytes for the entry, the default four for the deferred read, whose destination
            is not in the state dict.
        """
        checkpoint_dir = _checkpoint(self, {"w": torch.zeros(2)})
        planner = RemapLoadPlanner({"w": [RemapBlock((0,), (2,), "w", (0,), ((1,),))]})
        reader = HuggingFaceStorageReader(checkpoint_dir)
        planner.configure_planner({"w": torch.zeros(2, dtype=torch.float16)}, reader.load_metadata(), rank=0)

        sizes = (
            planner._shard_element_size(MetadataIndex("w")),  # pylint: disable=protected-access
            planner._shard_element_size(MetadataIndex("<deferred>/0/w")),  # pylint: disable=protected-access
        )
        self.assertEqual(sizes, (2, 4), f"element sizes mismatch: expected=(2, 4), got={sizes}")

    @patch("hyper_parallel.core.distributed_checkpoint.api.dist.get_world_size", return_value=1)
    @patch("hyper_parallel.core.distributed_checkpoint.api.dist.get_rank", return_value=0)
    def test_load_executes_a_remap_plan(self, mock_rank, mock_world_size):
        """
        Feature: dcp load with a RemapLoadPlanner.
        Description: Load a 6x3 concatenation of "q" and "k" through load(), as a world of one.
        Expectation: The tensor equals the concatenation.
        """
        del mock_rank, mock_world_size
        q, k = torch.randn(4, 3), torch.randn(2, 3)
        checkpoint_dir = _checkpoint(self, {"q": q, "k": k})
        fused = torch.zeros(6, 3)
        planner = RemapLoadPlanner({"qk": [_rows("q", 0, 4, 3), _rows("k", 0, 2, 3, dest_row=4)]})

        load({"qk": fused}, storage_reader=HuggingFaceStorageReader(checkpoint_dir), planner=planner)

        expected = torch.cat([q, k])
        self.assertTrue(torch.equal(fused, expected), f"fused mismatch: expected={expected}, got={fused}")


if __name__ == "__main__":
    unittest.main()
