# Copyright 2026
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
"""Focused tests for the Xing40 pinned-index and batch contracts."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import default_collate

from hyper_parallel.data.batching import FixedBatchDataLoader
from hyper_parallel.data.parallel.batch_sampler import build_dataset_batch_sampler
from hyper_parallel.trainer.config import entries_to_plan_overrides
from hyper_parallel.trainer.config.parser import parse_training_args


# File-based loading avoids the separate RL examples package during collection.
_EXAMPLE_PATH = Path(__file__).resolve().parents[4] / "examples/training_demo/xing40/xing40_alignment_data.py"
_EXAMPLE_SPEC = importlib.util.spec_from_file_location("xing40_alignment_data", _EXAMPLE_PATH)
_EXAMPLE = importlib.util.module_from_spec(_EXAMPLE_SPEC)
_EXAMPLE_SPEC.loader.exec_module(_EXAMPLE)
AlignmentBatch = _EXAMPLE.AlignmentBatch
AlignmentDataset = _EXAMPLE.AlignmentDataset


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mesh(dp: int, cp: int, ep: int, cp_rank: int = 0) -> SimpleNamespace:
    return SimpleNamespace(dp_size=dp, tp_size=1, cp_size=cp, pp_size=1, ep_size=ep, cp_rank=cp_rank)


def _options(length: int) -> dict[str, object]:
    return {
        "seq_length": length,
        "eod": -1,
        "pad": -1,
        "add_extra_token_to_sequence": True,
        "reset_position_ids": False,
        "reset_attention_mask": False,
        "eod_mask_loss": False,
        "create_attention_mask": False,
        "create_compressed_eod_mask": False,
    }


def _write_fixture(directory: Path) -> tuple[str, dict[str, str], dict[str, str], dict[str, str]]:
    prefix = directory / "tokens"
    sequences = [[10, 11, 12], [20, 21, 22, 23], [30, 31]]
    values = np.asarray([value for sequence in sequences for value in sequence], dtype=np.int32)
    values.tofile(str(prefix) + ".bin")
    lengths = np.asarray([len(sequence) for sequence in sequences], dtype=np.int32)
    pointers = np.asarray([0, 3 * 4, 7 * 4], dtype=np.int64)
    documents = np.asarray([0, 1, 2, 3], dtype=np.int64)
    with Path(str(prefix) + ".idx").open("wb") as stream:
        stream.write(b"MMIDIDX\x00\x00")
        stream.write(struct.pack("<QBQQ", 1, 4, len(sequences), len(documents)))
        stream.write(lengths.tobytes())
        stream.write(pointers.tobytes())
        stream.write(documents.tobytes())
    arrays = {
        "document_index": np.asarray([0, 1, 2], dtype=np.int32),
        "sample_index": np.asarray([[0, 1], [1, 1], [2, 0], [2, 1]], dtype=np.int64),
        "shuffle_index": np.asarray([0, 1, 2], dtype=np.int64),
    }
    paths = {}
    for name, values in arrays.items():
        path = directory / f"{name}.npy"
        np.save(path, values)
        paths[name] = str(path)
    index_sha256 = {name: _digest(Path(path)) for name, path in paths.items()}
    data_sha256 = {suffix: _digest(Path(f"{prefix}.{suffix}")) for suffix in ("bin", "idx")}
    return str(prefix), paths, index_sha256, data_sha256


class TestAlignmentData(unittest.TestCase):
    """Cover index mapping, padding, CP sharding, and the native sampler cursor."""

    def test_example_targets_resolve_with_rl_examples_namespace(self) -> None:
        """The launcher-owned path resolves targets even when RL owns examples."""
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(_EXAMPLE_PATH.parent) + os.pathsep + environment.get("PYTHONPATH", "")
        code = (
            "import importlib, pathlib, sys, types; "
            "rl_examples = types.ModuleType('examples'); rl_examples.__path__ = []; "
            "sys.modules['examples'] = rl_examples; "
            "adapter = importlib.import_module('xing40_alignment_data'); "
            "assert pathlib.Path(adapter.__file__).resolve() == pathlib.Path(sys.argv[1]).resolve(); "
            "assert all(callable(getattr(adapter, name)) for name in "
            "('build_alignment_dataset', 'build_alignment_collate_fn', 'AlignmentBatch')); "
            "from hyper_parallel.trainer.config.parser import parse_training_args; "
            "configuration = parse_training_args([sys.argv[2]]); "
            "assert configuration.training.global_batch_size == 2"
        )
        recipe = _EXAMPLE_PATH.parent / "alignment_configs/single.yaml"
        result = subprocess.run([sys.executable, "-B", "-c", code, str(_EXAMPLE_PATH), str(recipe)],
                                env=environment, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @patch.dict(sys.modules, {"xing40_alignment_data": _EXAMPLE})
    @unittest.skipUnless(
        sys.platform.startswith("linux") and shutil.which("bash"), "The Ascend launcher requires Linux and Bash"
    )
    def test_cli_paths_preserve_yaml_special_characters(self) -> None:
        """The real launcher preserves special paths through its dry-run command and the YAML parser."""
        directory = Path(self.temp_dir.name) / "paths # part: one"
        directory.mkdir()
        prefix, indices = directory / "tokens", directory / "indices"
        output, restore = directory / "run", directory / "checkpoint"
        restore.mkdir()
        paths = {"data_prefix": str(prefix), "index_paths": {}}
        for suffix in ("bin", "idx"):
            Path(f"{prefix}.{suffix}").write_bytes(b"\0")
        for name in ("document_index", "sample_index", "shuffle_index"):
            path = Path(f"{indices}-{name}.npy")
            path.write_bytes(b"\0")
            paths["index_paths"][name] = str(path)
        environment = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES="0", PYTHON=sys.executable)
        result = subprocess.run(
            [shutil.which("bash"), str(_EXAMPLE_PATH.parent / "run_alignment.sh"), "single",
             str(prefix), str(indices), str(output), "--resume", str(restore), "--dry-run"],
            env=environment, capture_output=True, text=True, check=False, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        command = shlex.split(result.stdout.removeprefix("Command: ").strip())
        entry = str(_EXAMPLE_PATH.parent.parent / "train_text.py")
        config = parse_training_args(command[command.index(entry) + 1:])
        dataset = config.dataset.to_dict()
        self.assertEqual(dataset["data_prefix"], paths["data_prefix"])
        self.assertEqual(dataset["index_paths"], paths["index_paths"])
        self.assertEqual(config.checkpoint.checkpoint_dir, str(output / "checkpoints"))
        self.assertEqual(config.checkpoint.restore_from, str(restore))
        self.assertFalse(output.exists(), "A dry run must not create training output")

    @patch.dict(sys.modules, {"xing40_alignment_data": _EXAMPLE})
    def test_public_recipes_parse_with_shared_model_and_optimizer(self) -> None:
        """All five runnable recipes preserve the shared numerical contract."""
        recipes = Path(__file__).resolve().parents[4] / "examples/training_demo/xing40/alignment_configs"
        baseline = parse_training_args([str(recipes / "fsdp4.yaml")])
        for name in ("single", "fsdp4", "fsdp4_cp2", "fsdp4_selective", "fsdp4_ep2"):
            with self.subTest(recipe=name):
                config = parse_training_args([str(recipes / f"{name}.yaml")])
                self.assertEqual(config.model.to_dict(), baseline.model.to_dict())
                self.assertEqual(config.optimizer.to_dict(), baseline.optimizer.to_dict())
                self.assertEqual(config.dataset.to_dict(), baseline.dataset.to_dict())
                self.assertEqual(config.training.train_iters, 2500)
                self.assertEqual(config.training.global_batch_size, 2 if name == "single" else 4)
                self.assertEqual(config.training.micro_batch_size, 1)
                plan = entries_to_plan_overrides(config.plan_overrides, cp_size=config.accelerator.cp_size,
                                                 ep_size=config.accelerator.ep_size)
                if name == "fsdp4_cp2":
                    self.assertIsNotNone(plan["*.self_attn"].inner_wrapper)
                    self.assertIsNotNone(plan["model.layers.2.mlp"].inner_wrapper)
                if name == "fsdp4_ep2":
                    self.assertIsNotNone(plan["model.layers.2.mlp"].local_compute_fn)

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        prefix, paths, index_hashes, data_hashes = _write_fixture(Path(self.temp_dir.name))
        self.dataset = AlignmentDataset(prefix, paths, index_hashes, data_hashes, 4)
        self.prefix = prefix
        self.paths = paths
        self.index_hashes = index_hashes
        self.data_hashes = data_hashes

    def test_cross_document_padding_and_hash_validation(self) -> None:
        actual = self.dataset[0]
        expected = {
            "input_ids": [11, 12, 20, 21],
            "labels": [12, 20, 21, 0],
            "loss_mask": [1, 1, 1, 0],
            "position_ids": [0, 1, 2, 3],
        }
        self.assertEqual(len(self.dataset), 3)
        for field, values in expected.items():
            self.assertEqual(actual[field].dtype, np.dtype("int32"), msg=f"field={field}")
            self.assertEqual(actual[field].tolist(), values, msg=f"field={field}")
        wrong_hashes = dict(self.index_hashes, sample_index="0" * 64)
        with self.assertRaisesRegex(ValueError, "hash or identity"):
            AlignmentDataset(self.prefix, self.paths, wrong_hashes, self.data_hashes, 4)

    def test_mask_and_contiguous_cp_halves(self) -> None:
        raw = default_collate([self.dataset[0]])
        single = AlignmentBatch(_mesh(1, 1, 1), "cpu", None, _options(4), False, sequence_length=4)
        whole_model, whole_loss = single.prepare_raw(raw)
        self.assertEqual(whole_model["labels"].tolist(), [[12, 20, 21, 0]])
        self.assertEqual(whole_model["shift_labels"].tolist(), [[12, 20, 21, -100]])
        self.assertEqual(whole_loss["loss_mask"].dtype, torch.int32)
        local = []
        for rank in range(2):
            batch = AlignmentBatch(_mesh(2, 2, 1, rank), "cpu", None, _options(4), False, sequence_length=4)
            local.append(batch.prepare_raw(raw))
        for key in ("input_ids", "labels", "shift_labels", "position_ids"):
            joined = torch.cat([entry[0][key] for entry in local], dim=1)
            self.assertTrue(torch.equal(joined, whole_model[key]), msg=f"field={key}")
        joined_mask = torch.cat([entry[1]["loss_mask"] for entry in local], dim=1)
        self.assertTrue(torch.equal(joined_mask, whole_loss["loss_mask"]))
        self.assertEqual(local[0][0]["position_ids"].tolist(), [[0, 1]])
        self.assertEqual(local[1][0]["position_ids"].tolist(), [[2, 3]])

    def test_stateful_loader_resume_from_fsdp4_to_cp2(self) -> None:
        """A real loader resumes the same four global sample IDs after DP4 to DP2/CP2."""
        samples = range(32)

        def loader(dp_rank: int, dp_size: int) -> FixedBatchDataLoader:
            sampler = build_dataset_batch_sampler(
                total_samples=len(samples), micro_batch_size=1, global_batch_size=4,
                dp_rank=dp_rank, dp_world_size=dp_size, drop_last=True,
            )
            return FixedBatchDataLoader(samples, batch_sampler=sampler, num_workers=0)

        sources = [loader(rank, 4) for rank in range(4)]
        source_iterators = [iter(source) for source in sources]
        for _ in range(3):
            for iterator in source_iterators:
                next(iterator)
        states = [source.state_dict() for source in sources]
        expected_next = [int(next(iterator).item()) for iterator in source_iterators]
        self.assertEqual(expected_next, [12, 13, 14, 15], f"next FSDP4 samples={expected_next}")

        # Rank layout is row-major: two CP coordinates for each DP coordinate.
        resumed = [loader(rank // 2, 2) for rank in range(4)]
        for target, state in zip(resumed, states):
            target.load_state_dict(state)
            target.set_epoch(0)
        iterators = [iter(target) for target in resumed]
        actual = [[int(next(iterator).item()) for iterator in iterators] for _ in range(2)]
        expected = [[12, 12, 13, 13], [14, 14, 15, 15]]
        self.assertEqual(actual, expected, f"expected CP2={expected}, got={actual}")

    def test_native_sampler_resume_does_not_apply_an_offset_twice(self) -> None:
        options = dict(total_samples=len(self.dataset), micro_batch_size=1, global_batch_size=2,
                       dp_rank=0, dp_world_size=1, drop_last=True)
        first_sampler = build_dataset_batch_sampler(**options)
        iterator = iter(first_sampler)
        self.assertEqual(next(iterator), [0])
        self.assertEqual(next(iterator), [1])
        state = first_sampler.state_dict()
        expected = next(iterator)
        restored_sampler = build_dataset_batch_sampler(**options)
        restored_sampler.load_state_dict(state)
        actual = next(iter(restored_sampler))
        self.assertEqual(actual, expected)
        self.assertEqual(actual, [2])
        self.assertEqual(self.dataset[actual[0]]["input_ids"].tolist(), [30, 31, 0, 0])


if __name__ == "__main__":
    unittest.main()
