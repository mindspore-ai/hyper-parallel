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
"""Unit tests of the perf-playbook extra-executed-work estimator.

Covers the Newton-Schulz FLOPs formula (including transpose invariance), the
four term kinds and their per-term sharding, relative `shapes_file`
resolution, the refusal of a term that carries no derivation, and the CLI's
artifact plus derivation block.
"""

import contextlib
import importlib.util
import io
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.common.mark_utils import arg_mark

_SCRIPTS = (Path(__file__).resolve().parents[3]
            / ".agent" / "skills" / "perf-playbook" / "scripts")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


extra_flops = _load("extra_flops")

# 4·m²·n + 2·m³ per iteration, m = min dim: (2,3) -> 48 + 16, (4,4) -> 256 + 128.
_NS_2X3 = 64
_NS_4X4 = 384


def _spec(terms: list) -> dict:
    return {"schema": extra_flops.SPEC_SCHEMA, "terms": terms}


def _flops(detail: dict) -> int:
    return round(detail["tflops_per_step"] * extra_flops.TFLOP)


class TestNewtonSchulzFormula(unittest.TestCase):
    """The per-matrix formula and its guards."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
              level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_formula_is_transpose_invariant_and_linear_in_steps(self):
        self.assertEqual(extra_flops.newton_schulz_flops(2, 3, 1), float(_NS_2X3))
        self.assertEqual(extra_flops.newton_schulz_flops(3, 2, 1), float(_NS_2X3))
        self.assertEqual(extra_flops.newton_schulz_flops(2, 3, 5), float(5 * _NS_2X3))
        self.assertEqual(extra_flops.newton_schulz_flops(4, 4, 1), float(_NS_4X4))

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
              level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_rejects_nonpositive_dims_and_steps(self):
        for args in ((0, 3, 1), (2, -1, 1), (2, 3, 0)):
            with self.assertRaisesRegex(extra_flops.ExtraFlopsError, "positive"):
                extra_flops.newton_schulz_flops(*args)


class TestTermKinds(unittest.TestCase):
    """Each term kind's arithmetic, plus per-term sharding."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
              level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_all_kinds_aggregate_with_sharding(self):
        spec = _spec([
            {"kind": "newton_schulz", "name": "muon-ns", "steps": 1,
             "shapes": [[2, 3], [4, 4]], "source": "shapes dumped under the target layout"},
            {"kind": "matmul", "name": "router-fixup", "dims": [2, 3, 4], "count": 2,
             "source": "two fix-up matmuls per layer, dims from the router module"},
            {"kind": "forward_fraction", "name": "selective-recompute", "fraction": 0.5,
             "model_tflops_per_step_per_device": 10.0, "source": "meter output x recomputed share"},
            {"kind": "measured", "name": "profiled", "tflops_per_step": 2.0,
             "shard_devices": 4, "source": "profiling breakdown, optimizer segment"},
        ])
        result = extra_flops.estimate(spec, Path("."))

        by_name = {detail["name"]: detail for detail in result["terms"]}
        self.assertEqual(_flops(by_name["muon-ns"]), _NS_2X3 + _NS_4X4)
        self.assertEqual(_flops(by_name["router-fixup"]), 2 * 2 * 3 * 4 * 2)
        self.assertAlmostEqual(by_name["selective-recompute"]["tflops_per_step"], 5.0)
        self.assertAlmostEqual(by_name["profiled"]["tflops_per_step"], 0.5)
        self.assertAlmostEqual(result["total_tflops_per_step"],
                               5.5 + (_NS_2X3 + _NS_4X4 + 96) / extra_flops.TFLOP)
        self.assertIn("hfu_extra_tflops_per_step:", result["config_line"])

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
              level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_shapes_file_resolves_against_the_spec_directory(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            (root / "shapes.json").write_text(json.dumps({"shapes": [[2, 3], [4, 4]]}),
                                              encoding="utf-8")
            spec = _spec([{"kind": "newton_schulz", "steps": 1, "shapes_file": "shapes.json",
                           "source": "named_parameters dump"}])
            result = extra_flops.estimate(spec, root)
            self.assertEqual(_flops(result["terms"][0]), _NS_2X3 + _NS_4X4)

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
              level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_a_term_without_a_derivation_is_refused(self):
        spec = _spec([{"kind": "measured", "name": "guess", "tflops_per_step": 3.0}])
        with self.assertRaisesRegex(extra_flops.ExtraFlopsError, "source"):
            extra_flops.estimate(spec, Path("."))
        blank = _spec([{"kind": "measured", "tflops_per_step": 3.0, "source": "   "}])
        with self.assertRaisesRegex(extra_flops.ExtraFlopsError, "source"):
            extra_flops.estimate(blank, Path("."))


class TestSpecValidation(unittest.TestCase):
    """Malformed specs are refused rather than silently coerced."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
              level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_schema_and_terms_are_checked(self):
        with self.assertRaisesRegex(extra_flops.ExtraFlopsError, "schema"):
            extra_flops.estimate({"schema": "other/v9", "terms": []}, Path("."))
        with self.assertRaisesRegex(extra_flops.ExtraFlopsError, "terms"):
            extra_flops.estimate(_spec([]), Path("."))
        with self.assertRaisesRegex(extra_flops.ExtraFlopsError, "object"):
            extra_flops.estimate(_spec(["not-an-object"]), Path("."))

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
              level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_unknown_kind_and_impossible_values_are_refused(self):
        cases = (
            ([{"kind": "vibes", "source": "s"}], "unknown term kind"),
            ([{"kind": "newton_schulz", "steps": 1, "shapes": [[2, 3, 4]], "source": "s"}],
             "not a 2-D shape"),
            ([{"kind": "forward_fraction", "fraction": 1.5,
               "model_tflops_per_step_per_device": 10.0, "source": "s"}], "<= 1"),
            ([{"kind": "measured", "tflops_per_step": 0, "source": "s"}], "positive"),
            ([{"kind": "measured", "tflops_per_step": 1.0, "shard_devices": 0, "source": "s"}],
             "positive integer"),
            ([{"kind": "matmul", "dims": [2, 3], "source": "s"}], "dims"),
        )
        for terms, message in cases:
            with self.assertRaisesRegex(extra_flops.ExtraFlopsError, message):
                extra_flops.estimate(_spec(terms), Path("."))


class TestCli(unittest.TestCase):
    """The command line writes the artifact and the derivation block."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
              level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_writes_artifact_and_prints_derivation(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            spec_path = root / "spec.json"
            spec_path.write_text(json.dumps(_spec([
                {"kind": "measured", "name": "profiled", "tflops_per_step": 2.5,
                 "source": "profiling breakdown"}])), encoding="utf-8")
            out_path = root / "out" / "extra_flops.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = extra_flops.main(["--spec", str(spec_path), "--out", str(out_path),
                                         "--markdown"])

            self.assertEqual(code, 0)
            written = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertAlmostEqual(written["total_tflops_per_step"], 2.5)
            printed = stdout.getvalue()
            self.assertIn("hfu_extra_tflops_per_step: 2.5", printed)
            self.assertIn("profiling breakdown", printed)

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
              level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_reports_a_bad_spec_without_traceback(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            spec_path = Path(tmp) / "spec.json"
            spec_path.write_text("{not json", encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = extra_flops.main(["--spec", str(spec_path)])
            self.assertEqual(code, 2)
            self.assertIn("extra_flops:", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
