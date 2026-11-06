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
"""Teacher artifact identity and checkpoint reuse without real model services."""

# Tests inspect package-owned state and stage boundaries without public runtime hooks.
# pylint: disable=protected-access

from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from rl.trainer.checkpoint import RLCheckpointManager
from rl.distillation.config import distillation_fingerprint, parse_distillation_config, teacher_weight_files
from rl.distillation.teacher import TeacherManager


class TestDistillationIdentity(unittest.TestCase):
    """Exercise content changes, broken manifests and frozen startup identities."""

    def setUp(self) -> None:
        """Create temporary teacher artifacts and register their cleanup."""
        resources = ExitStack()
        self.addCleanup(resources.close)
        self.root = Path(resources.enter_context(tempfile.TemporaryDirectory()))
        (self.root / "config.json").write_text('{}', encoding="utf-8")
        (self.root / "tokenizer.json").write_text('{}', encoding="utf-8")
        self.shards = [self.root / f"model-0000{number}-of-00002.safetensors" for number in (1, 2)]
        for number, shard in enumerate(self.shards):
            shard.write_bytes(bytes([number]) * 32)
        self.index = self.root / "model.safetensors.index.json"
        self.index.write_text(json.dumps({"weight_map": {str(i): p.name for i, p in enumerate(self.shards)}}))
        self.config = {"enabled": True, "teachers": {"teacher": {
            "model_path": str(self.root), "endpoint": "http://localhost:9000", "max_model_len": 32,
        }}}

    def validated_teacher(self) -> TeacherManager:
        """Validate local artifacts while substituting only the remote service."""
        manager = TeacherManager(parse_distillation_config(self.config), lambda *args: Mock())
        manager.validate(str(self.root))
        return manager

    def test_each_shard_content_changes_identity(self) -> None:
        """Equal size and preserved mtime must not hide replacement of either shard."""
        original = distillation_fingerprint(self.config)
        for shard in self.shards:
            with self.subTest(shard=shard.name):
                content, stat = shard.read_bytes(), shard.stat()
                shard.write_bytes(b'x' * len(content))
                os.utime(shard, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                self.assertNotEqual(original, distillation_fingerprint(self.config))
                shard.write_bytes(content)
        self.assertEqual(original, distillation_fingerprint(self.config))

    def test_index_mapping_changes_identity(self) -> None:
        """Changing tensor-to-shard assignments matters even with identical shard bytes."""
        original = distillation_fingerprint(self.config)
        self.index.write_text(json.dumps({"weight_map": {str(i): p.name for i, p in enumerate(reversed(self.shards))}}))
        self.assertNotEqual(original, distillation_fingerprint(self.config))

    def test_missing_or_escaping_shard_is_rejected(self) -> None:
        """An incomplete or unsafe shard index cannot become a validated identity."""
        for name in ("missing.safetensors", "../outside.safetensors"):
            with self.subTest(name=name):
                self.index.write_text(json.dumps({"weight_map": {"weight": name}}))
                with self.assertRaises(ValueError):
                    teacher_weight_files(str(self.root))

    def test_startup_requires_weights(self) -> None:
        """Metadata-only model directories cannot start teacher-backed training."""
        self.index.unlink()
        for shard in self.shards:
            shard.unlink()
        with self.assertRaisesRegex(ValueError, "requires local checkpoint weights"):
            self.validated_teacher()

    def test_checkpoint_reuses_startup_identity_and_rejects_changed_resume(self) -> None:
        """Saving does not rescan weights; a new process detects replacements on resume."""
        teacher = self.validated_teacher()
        resolved = {"distillation": self.config}
        output = self.root / "checkpoints"
        directory = output / "step_2"
        manager = RLCheckpointManager(
            SimpleNamespace(teacher_fingerprint=teacher.fingerprint, critic=None),
            {"output_dir": str(output), "load_path": str(directory)}, resolved,
            lambda operation, callback: callback(),
        )
        state = SimpleNamespace(global_step=2, epoch=1, consumed_samples=4, consumed_tokens=32)
        with patch("rl.trainer.checkpoint.dist.get_rank", return_value=0), \
                patch("rl.trainer.checkpoint.dist.get_world_size", return_value=1):
            with patch("rl.distillation.config._file_fingerprint", side_effect=AssertionError("Unexpected rescan")):
                manager._write_metadata(directory, state)
                (directory / "rank_0").mkdir()
                manager.validate_resume()
            saved = json.loads((directory / "checkpoint_complete.json").read_text())
            self.assertEqual(saved["distillation_fingerprint"], teacher.fingerprint)
            self.shards[1].write_bytes(b'new weights')
            manager.resources.teacher_fingerprint = self.validated_teacher().fingerprint
            with self.assertRaisesRegex(RuntimeError, "teacher identity differs"):
                manager.validate_resume()

    def test_checkpoint_rejects_unvalidated_identity(self) -> None:
        """Checkpoint IO cannot bypass teacher startup validation."""
        manager = RLCheckpointManager(
            SimpleNamespace(teacher_fingerprint=None), {"output_dir": str(self.root / "checkpoints")},
            {"distillation": self.config}, lambda operation, callback: callback(),
        )
        with self.assertRaisesRegex(RuntimeError, "validated teacher weight identity"):
            manager._teacher_fingerprint()
