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
"""Synthetic VLM data: Omni message schema, media paths, and determinism."""

import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from tests.common.mark_utils import arg_mark

_SCRIPT = (Path(__file__).resolve().parents[3]
           / "examples" / "training_demo" / "prepare_vlm_synthetic.py")
_SPEC = importlib.util.spec_from_file_location("prepare_vlm_synthetic", _SCRIPT)
gen = importlib.util.module_from_spec(_SPEC)
sys.modules["prepare_vlm_synthetic"] = gen
_SPEC.loader.exec_module(gen)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestPrepareVlmSynthetic(unittest.TestCase):
    """Records follow the Omni transform schema and are seed-deterministic."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_schema_and_files(self):
        pytest.importorskip("PIL")
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            jsonl = gen.generate(out, num_samples=4, image_size=64, seed=7)
            lines = jsonl.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 4)
            for line in lines:
                record = json.loads(line)
                messages = record["messages"]
                self.assertEqual([m["role"] for m in messages],
                                 ["user", "assistant"])
                image_block, text_block = messages[0]["content"]
                self.assertEqual(image_block["type"], "image")
                self.assertTrue((out / image_block["image"]).is_file(),
                                image_block["image"])
                self.assertEqual(text_block["type"], "text")
                reply = messages[1]["content"][0]
                self.assertIn("contains", reply["text"])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_seeded_runs_are_byte_identical(self):
        pytest.importorskip("PIL")
        with TemporaryDirectory() as tmp_a, TemporaryDirectory() as tmp_b:
            a = gen.generate(Path(tmp_a), num_samples=3, image_size=64, seed=11)
            b = gen.generate(Path(tmp_b), num_samples=3, image_size=64, seed=11)
            self.assertEqual(a.read_text(encoding="utf-8"),
                             b.read_text(encoding="utf-8"))
            self.assertEqual(_digest(Path(tmp_a) / "images" / "00000.png"),
                             _digest(Path(tmp_b) / "images" / "00000.png"))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_different_seed_changes_content(self):
        pytest.importorskip("PIL")
        with TemporaryDirectory() as tmp_a, TemporaryDirectory() as tmp_b:
            a = gen.generate(Path(tmp_a), num_samples=3, image_size=64, seed=1)
            b = gen.generate(Path(tmp_b), num_samples=3, image_size=64, seed=2)
            self.assertNotEqual(a.read_text(encoding="utf-8"),
                                b.read_text(encoding="utf-8"))
