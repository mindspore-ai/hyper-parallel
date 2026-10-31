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
"""The text corpus generator is deterministic and shaped as the recipe expects.

Determinism is the point: the equivalence matrix compares runs that must see
identical data, so a generator that drifts between invocations would make
every bitwise comparison meaningless without saying so.
"""

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.common.mark_utils import arg_mark

_SCRIPT = (Path(__file__).resolve().parents[3]
           / "examples" / "training_demo" / "hf_text_bringup"
           / "prepare_text_data.py")
_SPEC = importlib.util.spec_from_file_location("prepare_text_data", _SCRIPT)
gen = importlib.util.module_from_spec(_SPEC)
sys.modules["prepare_text_data"] = gen
_SPEC.loader.exec_module(gen)


def _run(out_path: Path, samples: int, length: int) -> list[dict]:
    gen.main(["--output", str(out_path), "--num-samples", str(samples),
              "--sequence-length", str(length)])
    return [json.loads(line) for line in
            out_path.read_text(encoding="utf-8").splitlines()]


class TestPrepareTextData(unittest.TestCase):
    """Record count, record shape, and byte-level repeatability."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_writes_requested_count_with_text_field(self):
        with TemporaryDirectory() as tmp:
            records = _run(Path(tmp) / "corpus.jsonl", samples=8, length=64)
            self.assertEqual(len(records), 8)
            for record in records:
                self.assertTrue(isinstance(record, dict) and record,
                                f"record must be a non-empty object: {record}")
                values = [v for v in record.values() if isinstance(v, str)]
                self.assertTrue(values, f"record carries no text: {record}")
                self.assertTrue(any(v.strip() for v in values))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_repeated_generation_is_byte_identical(self):
        with TemporaryDirectory() as tmp:
            first, second = Path(tmp) / "a.jsonl", Path(tmp) / "b.jsonl"
            _run(first, samples=6, length=48)
            _run(second, samples=6, length=48)
            self.assertEqual(first.read_bytes(), second.read_bytes(),
                             "equivalence runs depend on identical data")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_uniform_text_hits_the_target_including_indivisible_ones(self):
        """The unit length rarely divides the target; filling must close it."""
        # Stub tokenizer: 3 tokens per unit "u", 1 token per filler " a".
        def counts(text):
            return text.count("u") * 3 + text.count(" a")

        for target in (12, 13, 14, 300, 8192):
            text = gen.build_uniform_text("u", target, counts)
            self.assertEqual(counts(text), target, f"target {target}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_uniform_text_refuses_when_it_cannot_be_exact(self):
        """No single-token filler means no uniform corpus — say so, do not round."""
        def coarse(text):
            return text.count("u") * 3          # every filler adds zero tokens

        with self.assertRaises(ValueError):
            gen.build_uniform_text("u", 13, coarse)
        with self.assertRaises(ValueError):
            gen.build_uniform_text("u", 0, coarse)
        with self.assertRaises(ValueError):     # unit that never grows
            gen.build_uniform_text("x", 10, lambda text: 1)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_parent_directory_is_created(self):
        with TemporaryDirectory() as tmp:
            nested = Path(tmp) / "deep" / "deeper" / "corpus.jsonl"
            _run(nested, samples=2, length=32)
            self.assertTrue(nested.is_file())
