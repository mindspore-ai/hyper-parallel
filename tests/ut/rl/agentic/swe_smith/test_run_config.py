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
"""Host configuration preserves prepared task identity for either sandbox endpoint."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import yaml

from examples.swe_smith import run_config
from examples.swe_smith.run_config import build_config


class TestRunConfig(unittest.TestCase):
    """Check configuration changes without importing or starting the training runtime."""

    def setUp(self) -> None:
        """Create bounded prepared metadata, model directory and public recipe."""
        self.root = Path(tempfile.mkdtemp(prefix="swe-config-"))
        self.addCleanup(shutil.rmtree, self.root)
        self.data = self.root / "prepared data"
        (self.data / "instances").mkdir(parents=True)
        self.model = self.root / "model with spaces"
        self.model.mkdir()
        self.image_path = self.data / "images.json"
        self.images = {"schema_version": 1, "complete": True, "images": {
            "fixture/image": {"image": "sha256:" + "1" * 64, "python": "/usr/local/bin/python"}}}
        self.image_path.write_text(json.dumps(self.images), encoding="utf-8")
        selection = {"schema_version": 1, "requested_ids": None, "max_instances": None, "repeats": 1, "splits": {}}
        records = []
        for split in ("train", "validation"):
            row = {"instance_id": split + "-task", "problem_statement": "Repair source", "repo": "fixture/repo",
                   "image_name": "fixture/image", "FAIL_TO_PASS": ["test_api.py::test_bug"],
                   "PASS_TO_PASS": ["test_api.py::test_ok"]}
            private = self.data / "instances" / (hashlib.sha256(row["instance_id"].encode()).hexdigest() + ".json")
            private.write_text(json.dumps(row) + "\n", encoding="utf-8")
            records.append({**row, "split": split})
            selection["splits"][split] = {"source": str(self.root / f"original-{split}.jsonl"),
                                           "sha256": "2" * 64, "instance_ids": [row["instance_id"]]}
            (self.data / f"{split}.parquet").write_bytes(b"PAR1" + split.encode() + b"PAR1")
        (self.data / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
        (self.data / "instances.jsonl").write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
        source = Path(run_config.__file__).parent / "configs/qwen3_5_9b.yaml"
        self.template = yaml.safe_load(source.read_text(encoding="utf-8"))
        self.arguments = {"data_dir": self.data, "images_path": self.image_path, "model_dir": self.model,
                          "output_dir": self.root / "run output", "checkpoint_dir": self.root / "new checkpoints",
                          "docker_host": None}

    def _build(self, **overrides) -> dict:
        return build_config(self.template, **{**self.arguments, **overrides})

    def _checkpoint(self, config: dict, step: int = 1) -> Path:
        checkpoint = self.root / "old checkpoints" / f"step_{step}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "checkpoint_complete.json").write_text(json.dumps({"step": step}), encoding="utf-8")
        (checkpoint / "resolved_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        return checkpoint

    def test_local_remote_only_change_endpoint_and_preserve_budgets(self) -> None:
        """Both execution modes use identical task identities and learning settings."""
        local = self._build()
        self.assertEqual(local, self._build(docker_host="unix:///var/run/docker.sock"))
        remote = self._build(docker_host="ssh://developer@example.test:2233")
        for section in ("task", "workspace"):
            self.assertEqual(local["agentic"]["codex"][section]["docker_host"], "unix:///var/run/docker.sock")
            self.assertEqual(remote["agentic"]["codex"][section]["docker_host"],
                             "ssh://developer@example.test:2233")
            remote["agentic"]["codex"][section]["docker_host"] = local["agentic"]["codex"][section]["docker_host"]
        self.assertEqual(local, remote)
        self.assertEqual(local["agentic"]["max_turns"], 2)
        self.assertEqual(local["rollout"]["max_new_tokens"], 512)
        self.assertEqual(local["algorithm"], self.template["algorithm"])

    def test_template_is_independent_and_paths_with_spaces_are_absolute(self) -> None:
        """Configuration preparation leaves the public recipe and input data untouched."""
        original = deepcopy(self.template)
        config = self._build(steps=3)
        self.assertEqual(self.template, original)
        self.assertEqual(config["model"]["weights_path"], str(self.model.resolve()))
        self.assertEqual(config["data"]["train_path"], str(self.data.resolve() / "train.parquet"))
        self.assertEqual(config["agentic"]["codex"]["session_root"], str(self.root / "run output" / "sessions"))
        self.assertEqual(config["logging"]["wandb"]["directory"], str(self.root / "run output" / "wandb"))
        self.assertEqual(config["train"]["max_steps"], 3)
        self.assertIsNone(config["train"]["checkpoint"]["load_path"])
        config["rollout"]["max_new_tokens"] = 7
        self.assertEqual(self.template, original)

    def test_image_manifest_change_and_incomplete_manifest_are_rejected(self) -> None:
        """Pinned manifest bytes cannot change while retaining configuration identity."""
        self.template = self._build()
        self.images["images"]["fixture/image"]["image"] = "sha256:" + "3" * 64
        self.image_path.write_text(json.dumps(self.images), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self._build()
        self.images["complete"] = False
        self.image_path.write_text(json.dumps(self.images), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "complete"):
            self._build()

    def test_selection_private_records_and_missing_split_are_rejected(self) -> None:
        """Prepared data must retain the exact selected public and private task records."""
        selection_path = self.data / "selection.json"
        original = selection_path.read_text(encoding="utf-8")
        selection = json.loads(original)
        selection["splits"]["train"]["instance_ids"] = ["unexpected"]
        selection_path.write_text(json.dumps(selection), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "selection"):
            self._build()
        selection_path.write_text(original, encoding="utf-8")
        private = next((self.data / "instances").iterdir())
        private.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "records"):
            self._build()
        private.unlink()
        with self.assertRaisesRegex(ValueError, "records"):
            self._build()

    def test_resume_pins_data_images_and_original_data_paths(self) -> None:
        """A completed checkpoint continues only with the same prepared task identity."""
        original = self._build()
        checkpoint = self._checkpoint(original)
        resumed = self._build(steps=2, resume=checkpoint)
        self.assertEqual(resumed["train"]["checkpoint"]["load_path"], str(checkpoint))
        with self.assertRaisesRegex(ValueError, "greater"):
            self._build(steps=1, resume=checkpoint)
        with self.assertRaisesRegex(ValueError, "inside input"):
            self._build(steps=2, resume=checkpoint, checkpoint_dir=checkpoint)
        saved = checkpoint / "resolved_config.yaml"
        changed = deepcopy(original)
        changed["agentic"]["codex"]["task"]["images_sha256"] = "4" * 64
        saved.write_text(yaml.safe_dump(changed), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "identity differs"):
            self._build(steps=2, resume=checkpoint)
        changed = deepcopy(original)
        changed["data"]["train_path"] = str(self.root / "different.parquet")
        saved.write_text(yaml.safe_dump(changed), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "data source"):
            self._build(steps=2, resume=checkpoint)
        saved.write_text(yaml.safe_dump(original), encoding="utf-8")
        (self.data / "train.parquet").write_bytes(b"different public data")
        with self.assertRaisesRegex(ValueError, "identity differs"):
            self._build(steps=2, resume=checkpoint)

    def test_resume_requires_complete_metadata_and_saved_data_fingerprint(self) -> None:
        """Checkpoints must preserve data identity before this entry can resume them."""
        original = self._build()
        original["agentic"]["codex"]["task"].pop("data_sha256")
        checkpoint = self._checkpoint(original)
        with self.assertRaisesRegex(ValueError, "identities"):
            self._build(steps=2, resume=checkpoint)
        (checkpoint / "checkpoint_complete.json").unlink()
        with self.assertRaisesRegex(ValueError, "checkpoint_complete"):
            self._build(steps=2, resume=checkpoint)

    def test_invalid_remote_endpoints_steps_and_input_output_overlap_are_rejected(self) -> None:
        """Endpoint ambiguity and unsafe output destinations fail before model loading."""
        for endpoint in ("", "unix:///tmp/docker.sock", "tcp://host:2375", "ssh://", "ssh://user:p@host"):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                self._build(docker_host=endpoint)
        for steps in (0, -1, True, 1.5):
            with self.subTest(steps=steps), self.assertRaisesRegex(ValueError, "positive integer"):
                self._build(steps=steps)
        with self.assertRaisesRegex(ValueError, "inside input"):
            self._build(output_dir=self.data / "output")
        for field in ("output_dir", "checkpoint_dir"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "contain"):
                self._build(**{field: self.data.parent})

    def test_inference_recipe_and_fresh_import_do_not_require_torch(self) -> None:
        """The host preflight supports the inference recipe without heavy framework imports."""
        source = Path(run_config.__file__).parent / "configs/qwen3_5_9b_infer.yaml"
        self.template = yaml.safe_load(source.read_text(encoding="utf-8"))
        config = self._build()
        self.assertEqual(config["agentic"]["max_turns"], 100)
        program = ("import examples.swe_smith.run_config,sys; "
                   "print('torch' in sys.modules or 'hyper_parallel' in sys.modules)")
        result = subprocess.run([sys.executable, "-c", program], text=True, capture_output=True, check=True,
                                cwd=Path(run_config.__file__).parents[2])
        self.assertEqual(result.stdout.strip(), "False")
