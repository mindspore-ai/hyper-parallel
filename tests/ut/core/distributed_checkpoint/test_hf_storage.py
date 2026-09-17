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
"""UT for :mod:`hyper_parallel.core.distributed_checkpoint.hf_storage`."""
import json
import shutil
import struct
import tempfile
import unittest
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from hyper_parallel.core.distributed_checkpoint.api import load
from hyper_parallel.core.distributed_checkpoint.hf_storage import (
    DCP_SHARDING_INFO_KEY,
    SAFETENSORS_INDEX_SUFFIX,
    SAVED_OFFSETS_KEY,
    HuggingFaceStorageReader,
    TorchShardedSafetensorsReader,
)
from hyper_parallel.core.distributed_checkpoint.metadata import (
    CHUNK_INFO,
    ChunkInfo,
    ChunkStorageMetadata,
    MetadataIndex,
)
from hyper_parallel.core.distributed_checkpoint.standard_planner import StandardLoadPlanner

# The names torch's HuggingFaceStorageWriter gives the shard files of the first two ranks.
_SHARD_1 = "shard-00001-model-00001-of-00001.safetensors"
_SHARD_2 = "shard-00002-model-00001-of-00001.safetensors"


def _temporary_directory(test_case: unittest.TestCase) -> Path:
    """A fresh directory, removed once ``test_case`` is done."""
    path = Path(tempfile.mkdtemp(prefix="test_hf_storage_"))
    test_case.addCleanup(shutil.rmtree, path, ignore_errors=True)
    return path


def _save(path: Path, tensors: dict[str, torch.Tensor], offsets: Optional[dict[str, list[int]]] = None) -> None:
    """
    Write ``tensors`` to one safetensors file.

    Args:
        path (Path): The file to write.
        tensors (dict[str, torch.Tensor]): The tensors, by name.
        offsets (Optional[dict[str, list[int]]]): Where each tensor starts in a larger one, recorded the
            way torch's Hugging Face writer records it. None records nothing.
    """
    metadata = None
    if offsets is not None:
        sharding_info = {key: {SAVED_OFFSETS_KEY: value} for key, value in offsets.items()}
        metadata = {DCP_SHARDING_INFO_KEY: json.dumps(sharding_info)}
    save_file(tensors, str(path), metadata=metadata)


def _raw_safetensors(header: Any, data: bytes = b"") -> bytes:
    """The bytes of a safetensors file with this header, for headers safetensors itself would not write."""
    encoded = json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded + data


def _save_index(checkpoint_dir: Path, weight_map: dict[str, str]) -> None:
    """Write the index of a sharded Hugging Face checkpoint."""
    index = {"metadata": {"total_size": 0}, "weight_map": weight_map}
    index_path = checkpoint_dir / f"model{SAFETENSORS_INDEX_SUFFIX}"
    index_path.write_text(json.dumps(index), encoding="utf-8")


def _stored_files(metadata: Any) -> dict[str, str]:
    """The file each tensor of a checkpoint of whole tensors is stored in, by tensor name."""
    return {index.fqn: info.relative_path for index, info in metadata.storage_data.items()}


def _read_into(reader: Any, state_dict: dict[str, Any]) -> None:
    """Plan and read ``state_dict`` through ``reader`` as a single rank, with no process group."""
    metadata = reader.load_metadata()
    planner = StandardLoadPlanner()
    planner.configure_planner(state_dict, metadata, rank=0)
    plan = planner.build_local_plan()
    reader.configure_reader(metadata, is_coordinator=True, rank=0)
    reader.execute_read(plan, planner)


class TestHuggingFaceStorageReader(unittest.TestCase):
    """Tests for reading Hugging Face checkpoints of whole tensors through DCP."""

    def test_single_file_is_described_by_its_header(self):
        """
        Feature: HuggingFaceStorageReader.load_metadata on a one-file checkpoint.
        Description: Save a bfloat16 matrix, a float32 vector and an int64 scalar to model.safetensors.
        Expectation: Each tensor is one chunk at the origin, with its dtype spelled as DCP spells it,
            stored under its own name in model.safetensors.
        """
        checkpoint_dir = _temporary_directory(self)
        _save(checkpoint_dir / "model.safetensors", {
            "embed.weight": torch.zeros(4, 3, dtype=torch.bfloat16),
            "norm.weight": torch.zeros(3),
            "step": torch.tensor(5),
        })

        metadata = HuggingFaceStorageReader(checkpoint_dir).load_metadata()

        described = {
            key: (md.size, md.properties.dtype, md.chunks) for key, md in metadata.state_dict_metadata.items()
        }
        expected = {
            "embed.weight": ((4, 3), "torch.bfloat16", [ChunkStorageMetadata(offsets=(0, 0), sizes=(4, 3))]),
            "norm.weight": ((3,), "torch.float32", [ChunkStorageMetadata(offsets=(0,), sizes=(3,))]),
            "step": ((), "torch.int64", [ChunkStorageMetadata(offsets=(), sizes=())]),
        }
        self.assertEqual(described, expected, f"metadata mismatch: expected={expected}, got={described}")
        stored = {index.fqn: (info.relative_path, info.tensor_key) for index, info in metadata.storage_data.items()}
        expected_stored = {key: ("model.safetensors", key) for key in expected}
        self.assertEqual(stored, expected_stored, f"storage mismatch: expected={expected_stored}, got={stored}")
        index = MetadataIndex(fqn="embed.weight", offset=(0, 0))
        self.assertIn(index, metadata.storage_data, f"{index} not among {list(metadata.storage_data)}")

    @patch("hyper_parallel.core.distributed_checkpoint.api.dist.get_world_size", return_value=1)
    @patch("hyper_parallel.core.distributed_checkpoint.api.dist.get_rank", return_value=0)
    def test_load_fills_a_state_dict_from_an_indexed_checkpoint(self, mock_rank, mock_world_size):
        """
        Feature: dcp load with HuggingFaceStorageReader as its storage reader.
        Description: Write two shard files and an index the way save_pretrained does, then load a state
            dict holding a float32 matrix stored as bfloat16, a vector and an int64 scalar.
        Expectation: Every entry holds the stored values, in the dtype of the state dict.
        """
        del mock_rank, mock_world_size
        checkpoint_dir = _temporary_directory(self)
        embed = torch.arange(12, dtype=torch.bfloat16).reshape(4, 3)
        norm = torch.tensor([0.5, 1.5, 2.5])
        _save(checkpoint_dir / "model-00001-of-00002.safetensors", {"embed.weight": embed})
        _save(checkpoint_dir / "model-00002-of-00002.safetensors", {"norm.weight": norm, "step": torch.tensor(7)})
        _save_index(checkpoint_dir, {
            "embed.weight": "model-00001-of-00002.safetensors",
            "norm.weight": "model-00002-of-00002.safetensors",
            "step": "model-00002-of-00002.safetensors",
        })
        state_dict = {"embed.weight": torch.zeros(4, 3), "norm.weight": torch.zeros(3), "step": torch.tensor(0)}

        load(state_dict, storage_reader=HuggingFaceStorageReader(checkpoint_dir))

        expected = {"embed.weight": embed.float(), "norm.weight": norm, "step": torch.tensor(7)}
        for key, value in expected.items():
            self.assertTrue(torch.equal(state_dict[key], value),
                            f"{key} mismatch: expected={value}, got={state_dict[key]}")

    def test_index_limits_the_read_to_the_files_it_names(self):
        """
        Feature: HuggingFaceStorageReader.load_metadata with a model.safetensors.index.json.
        Description: Beside two indexed shard files, save a consolidated.safetensors holding one of
            their tensors again and one tensor of its own, which the index does not name.
        Expectation: Only the indexed files are described: the stray file is neither registered nor
            taken for a second copy of a tensor.
        """
        checkpoint_dir = _temporary_directory(self)
        _save(checkpoint_dir / "model-00001-of-00002.safetensors", {"embed.weight": torch.zeros(4, 3)})
        _save(checkpoint_dir / "model-00002-of-00002.safetensors", {"norm.weight": torch.zeros(3)})
        _save(checkpoint_dir / "consolidated.safetensors", {"norm.weight": torch.ones(3), "extra": torch.ones(1)})
        weight_map = {
            "embed.weight": "model-00001-of-00002.safetensors",
            "norm.weight": "model-00002-of-00002.safetensors",
        }
        _save_index(checkpoint_dir, weight_map)

        files = _stored_files(HuggingFaceStorageReader(checkpoint_dir).load_metadata())

        self.assertEqual(files, weight_map, f"files mismatch: expected={weight_map}, got={files}")

    def test_without_index_only_the_weights_file_is_read(self):
        """
        Feature: HuggingFaceStorageReader.load_metadata without an index.
        Description: Beside model.safetensors, save consolidated.safetensors holding the same tensor,
            and a file named like a torch shard whose content is not safetensors at all.
        Expectation: Only model.safetensors is described, and the other two files are never opened.
        """
        checkpoint_dir = _temporary_directory(self)
        _save(checkpoint_dir / "model.safetensors", {"norm.weight": torch.zeros(3)})
        _save(checkpoint_dir / "consolidated.safetensors", {"norm.weight": torch.ones(3)})
        (checkpoint_dir / _SHARD_1).write_bytes(b"not a safetensors file")

        files = _stored_files(HuggingFaceStorageReader(checkpoint_dir).load_metadata())

        expected = {"norm.weight": "model.safetensors"}
        self.assertEqual(files, expected, f"files mismatch: expected={expected}, got={files}")

    def test_tensor_in_two_indexed_files_is_rejected(self):
        """
        Feature: HuggingFaceStorageReader.load_metadata duplicate check.
        Description: Save the same tensor into both of the files an index names.
        Expectation: ValueError naming both files, since either could be the one meant.
        """
        checkpoint_dir = _temporary_directory(self)
        files = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
        _save(checkpoint_dir / files[0], {"norm.weight": torch.zeros(3)})
        _save(checkpoint_dir / files[1], {"norm.weight": torch.ones(3), "embed.weight": torch.ones(2)})
        _save_index(checkpoint_dir, {"norm.weight": files[0], "embed.weight": files[1]})

        with self.assertRaises(ValueError) as ctx:
            HuggingFaceStorageReader(checkpoint_dir).load_metadata()
        for file_name in files:
            self.assertIn(file_name, str(ctx.exception), f"{file_name} not named in: {ctx.exception}")

    def test_recorded_offsets_have_to_describe_whole_tensors(self):
        """
        Feature: HuggingFaceStorageReader on files that record DCP_SHARDING_INFO.
        Description: Save a model.safetensors recording its tensor at zero offsets, and another one
            recording its tensor at offsets (2, 0).
        Expectation: The first is read as a whole tensor. The second is rejected as a shard of a larger
            tensor, naming TorchShardedSafetensorsReader as the reader for it.
        """
        whole_dir = _temporary_directory(self)
        _save(whole_dir / "model.safetensors", {"w": torch.zeros(2, 4)}, {"w": [0, 0]})
        described = HuggingFaceStorageReader(whole_dir).load_metadata().state_dict_metadata["w"]
        expected_chunks = [ChunkStorageMetadata(offsets=(0, 0), sizes=(2, 4))]
        self.assertEqual((described.size, described.chunks), ((2, 4), expected_chunks),
                         f"expected (2, 4) with {expected_chunks}, got {described.size} with {described.chunks}")

        shard_dir = _temporary_directory(self)
        _save(shard_dir / "model.safetensors", {"w": torch.zeros(2, 4)}, {"w": [2, 0]})
        with self.assertRaises(ValueError) as ctx:
            HuggingFaceStorageReader(shard_dir).load_metadata()
        self.assertIn("TorchShardedSafetensorsReader", str(ctx.exception), f"reader not named in: {ctx.exception}")

    def test_malformed_files_are_rejected(self):
        """
        Feature: HuggingFaceStorageReader.load_metadata header and index validation.
        Description: Point the reader at a tensor in a dtype torch has no counterpart for, a header
            entry without a shape, a header longer than its file, a header that is not JSON, and an
            index without a weight_map.
        Expectation: ValueError for each, saying what is wrong.
        """
        cases = [
            ("F4", "model.safetensors",
             _raw_safetensors({"w": {"dtype": "F4", "shape": [2], "data_offsets": [0, 1]}}, b"\0")),
            ("shape", "model.safetensors", _raw_safetensors({"w": {"dtype": "F32", "data_offsets": [0, 0]}})),
            ("too short", "model.safetensors", struct.pack("<Q", 1024) + b"{}"),
            ("JSON", "model.safetensors", struct.pack("<Q", 3) + b"{x}"),
            ("weight_map", f"model{SAFETENSORS_INDEX_SUFFIX}", b"{}"),
        ]
        for fragment, file_name, content in cases:
            with self.subTest(fragment):
                checkpoint_dir = _temporary_directory(self)
                (checkpoint_dir / file_name).write_bytes(content)
                with self.assertRaises(ValueError) as ctx:
                    HuggingFaceStorageReader(checkpoint_dir).load_metadata()
                self.assertIn(fragment, str(ctx.exception), f"{fragment!r} not in: {ctx.exception}")

    def test_directories_without_the_checkpoint_raise_file_not_found(self):
        """
        Feature: HuggingFaceStorageReader.load_metadata on directories without a Hugging Face checkpoint.
        Description: Point the reader at an empty directory, asking with and without rank as load does;
            at an index naming a file that is not there; and at a directory of torch shards.
        Expectation: FileNotFoundError every time, which load passes on to its caller. For the torch
            shards the error names TorchShardedSafetensorsReader.
        """
        empty_dir = _temporary_directory(self)
        for kwargs in ({}, {"rank": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(FileNotFoundError):
                HuggingFaceStorageReader(empty_dir).load_metadata(**kwargs)

        indexed_dir = _temporary_directory(self)
        _save_index(indexed_dir, {"w": "model-00001-of-00002.safetensors"})
        with self.assertRaises(FileNotFoundError):
            HuggingFaceStorageReader(indexed_dir).load_metadata()

        shards_dir = _temporary_directory(self)
        _save(shards_dir / _SHARD_1, {"w": torch.zeros(2)}, {"w": [0]})
        with self.assertRaises(FileNotFoundError) as ctx:
            HuggingFaceStorageReader(shards_dir).load_metadata()
        self.assertIn("TorchShardedSafetensorsReader", str(ctx.exception), f"reader not named in: {ctx.exception}")


class TestTorchShardedSafetensorsReader(unittest.TestCase):
    """Tests for reading the shards torch's HuggingFaceStorageWriter writes, through DCP."""

    def test_shards_are_joined_into_one_tensor(self):
        """
        Feature: TorchShardedSafetensorsReader on shard files recording DCP_SHARDING_INFO.
        Description: Save rows 0-1 and rows 2-4 of a 5x4 tensor under one name in two shard files with
            their saved_offsets, and a whole vector in the first file. Then read the tensor into a
            whole tensor and into a partial one spanning both files.
        Expectation: A 5x4 tensor made of two chunks and the vector are described, and both reads
            return the stored values.
        """
        checkpoint_dir = _temporary_directory(self)
        full = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        norm = torch.tensor([0.5, 1.5, 2.5])
        _save(checkpoint_dir / _SHARD_1, {"w": full[:2].clone(), "norm": norm}, {"w": [0, 0], "norm": [0]})
        _save(checkpoint_dir / _SHARD_2, {"w": full[2:].clone()}, {"w": [2, 0]})

        metadata = TorchShardedSafetensorsReader(checkpoint_dir).load_metadata()

        described = {key: (md.size, md.chunks) for key, md in metadata.state_dict_metadata.items()}
        expected = {
            "w": ((5, 4), [ChunkStorageMetadata(offsets=(0, 0), sizes=(2, 4)),
                           ChunkStorageMetadata(offsets=(2, 0), sizes=(3, 4))]),
            "norm": ((3,), [ChunkStorageMetadata(offsets=(0,), sizes=(3,))]),
        }
        self.assertEqual(described, expected, f"metadata mismatch: expected={expected}, got={described}")

        whole, whole_norm = torch.zeros(5, 4), torch.zeros(3)
        _read_into(TorchShardedSafetensorsReader(checkpoint_dir), {"w": whole, "norm": whole_norm})
        self.assertTrue(torch.equal(whole, full), f"whole read mismatch: expected={full}, got={whole}")
        self.assertTrue(torch.equal(whole_norm, norm), f"norm read mismatch: expected={norm}, got={whole_norm}")

        part = torch.zeros(2, 4)
        chunk = ChunkStorageMetadata(offsets=(1, 0), sizes=(2, 4))
        setattr(part, CHUNK_INFO, ChunkInfo(chunk=chunk, global_shape=(5, 4)))
        _read_into(TorchShardedSafetensorsReader(checkpoint_dir), {"w": part})
        self.assertTrue(torch.equal(part, full[1:3]), f"partial read mismatch: expected={full[1:3]}, got={part}")

    def test_shards_that_do_not_make_up_the_tensor_are_rejected(self):
        """
        Feature: TorchShardedSafetensorsReader.load_metadata coverage check.
        Description: Save two 2x4 shards of one tensor with a row missing between them, and two that
            overlap by a row.
        Expectation: ValueError naming the tensor for both, rather than a tensor with a region that is
            read by nobody or read twice.
        """
        for name, second_offsets in {"gap": [3, 0], "overlap": [1, 0]}.items():
            with self.subTest(name):
                checkpoint_dir = _temporary_directory(self)
                _save(checkpoint_dir / _SHARD_1, {"w": torch.zeros(2, 4)}, {"w": [0, 0]})
                _save(checkpoint_dir / _SHARD_2, {"w": torch.zeros(2, 4)}, {"w": second_offsets})
                with self.assertRaises(ValueError) as ctx:
                    TorchShardedSafetensorsReader(checkpoint_dir).load_metadata()
                self.assertIn("'w'", str(ctx.exception), f"tensor 'w' not named in: {ctx.exception}")

    def test_inconsistent_shards_are_rejected(self):
        """
        Feature: TorchShardedSafetensorsReader.load_metadata consistency checks.
        Description: Save two shards of one tensor in different dtypes; a shard file that records the
            offsets of one of its tensors but not of the other; and the same shard in two files.
        Expectation: ValueError naming the tensor at fault each time, and both files for the shard that
            is stored twice.
        """
        dtypes_dir = _temporary_directory(self)
        _save(dtypes_dir / _SHARD_1, {"w": torch.zeros(2)}, {"w": [0]})
        _save(dtypes_dir / _SHARD_2, {"w": torch.zeros(2, dtype=torch.float16)}, {"w": [2]})
        offsets_dir = _temporary_directory(self)
        _save(offsets_dir / _SHARD_1, {"w": torch.zeros(2), "b": torch.zeros(2)}, {"w": [0]})
        repeated_dir = _temporary_directory(self)
        _save(repeated_dir / _SHARD_1, {"w": torch.zeros(2)}, {"w": [0]})
        _save(repeated_dir / _SHARD_2, {"w": torch.ones(2)}, {"w": [0]})

        cases = (
            ("dtypes", dtypes_dir, ("'w'",)),
            ("offsets", offsets_dir, ("'b'",)),
            ("repeated", repeated_dir, ("'w'", _SHARD_1, _SHARD_2)),
        )
        for name, checkpoint_dir, fragments in cases:
            with self.subTest(name):
                with self.assertRaises(ValueError) as ctx:
                    TorchShardedSafetensorsReader(checkpoint_dir).load_metadata()
                for fragment in fragments:
                    self.assertIn(fragment, str(ctx.exception), f"{fragment} not named in: {ctx.exception}")

    def test_checkpoints_of_whole_tensors_are_rejected(self):
        """
        Feature: TorchShardedSafetensorsReader.load_metadata on Hugging Face checkpoints.
        Description: Point the reader at a directory with a model.safetensors.index.json, and at one
            whose model.safetensors records no DCP_SHARDING_INFO.
        Expectation: ValueError both times, naming HuggingFaceStorageReader as the reader for them.
        """
        indexed_dir = _temporary_directory(self)
        _save(indexed_dir / "model-00001-of-00001.safetensors", {"w": torch.zeros(2)})
        _save_index(indexed_dir, {"w": "model-00001-of-00001.safetensors"})
        plain_dir = _temporary_directory(self)
        _save(plain_dir / "model.safetensors", {"w": torch.zeros(2)})

        for name, checkpoint_dir in (("index", indexed_dir), ("no offsets", plain_dir)):
            with self.subTest(name):
                with self.assertRaises(ValueError) as ctx:
                    TorchShardedSafetensorsReader(checkpoint_dir).load_metadata()
                self.assertIn("HuggingFaceStorageReader", str(ctx.exception), f"reader not named in: {ctx.exception}")

    def test_empty_directory_raises_file_not_found(self):
        """
        Feature: TorchShardedSafetensorsReader.load_metadata on a directory with nothing to read.
        Description: Point the reader at an empty directory, asking with and without rank as load does.
        Expectation: FileNotFoundError both times, which load passes on to its caller.
        """
        empty_dir = _temporary_directory(self)
        for kwargs in ({}, {"rank": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(FileNotFoundError):
                TorchShardedSafetensorsReader(empty_dir).load_metadata(**kwargs)


if __name__ == "__main__":
    unittest.main()
