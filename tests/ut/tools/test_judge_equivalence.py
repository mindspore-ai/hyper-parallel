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
"""The S3 equivalence judge passes faithful pairs and names each violation."""

import contextlib
import importlib.util
import io
import json
import math
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.common.mark_utils import arg_mark

_SCRIPT = (Path(__file__).resolve().parents[3]
           / "examples" / "training_demo" / "hf_text_bringup"
           / "judge_equivalence.py")
_SPEC = importlib.util.spec_from_file_location("judge_equivalence", _SCRIPT)
judge_mod = importlib.util.module_from_spec(_SPEC)
sys.modules["judge_equivalence"] = judge_mod
_SPEC.loader.exec_module(judge_mod)

_LINE = ("Training: x | step={s} epoch=0 data/step_tokens=16384 "
         "training/foundation_loss={l} training/grad_norm={g} "
         "training/lr=3e-05 training/total_loss={l}\n")


def _write_log(path: Path, rows):
    path.write_text("".join(_LINE.format(s=s, l=l, g=g) for s, l, g in rows),
                    encoding="utf-8")


class TestJudgeEquivalence(unittest.TestCase):
    """Bitwise and band checks on synthetic trainer logs."""

    def _logs(self, tmp):
        base = [(1, "11.1012", "26.7552"), (2, "11.0431", "26.9101"),
                (3, "10.9079", "27.8310")]
        off = [(s, f"{float(l) * 1.005:.4f}", g) for s, l, g in base]
        paths = {}
        for name, rows in (("e1", base), ("e2", base), ("e3", off)):
            paths[name] = Path(tmp) / f"{name}.log"
            _write_log(paths[name], rows)
        return paths

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_identical_runs_pass_bitwise_and_band_holds(self):
        with TemporaryDirectory() as tmp:
            paths = self._logs(tmp)
            code = judge_mod.main([
                "--series", f"e1={paths['e1']}",
                "--series", f"e2={paths['e2']}",
                "--series", f"e3={paths['e3']}",
                "--bitwise", "e1,e2", "--band", "e1,e3,0.02"])
            self.assertEqual(code, 0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_divergence_fails_bitwise_and_tight_band(self):
        with TemporaryDirectory() as tmp:
            paths = self._logs(tmp)
            self.assertEqual(judge_mod.main([
                "--series", f"e1={paths['e1']}",
                "--series", f"e3={paths['e3']}",
                "--bitwise", "e1,e3"]), 2)
            self.assertEqual(judge_mod.main([
                "--series", f"e1={paths['e1']}",
                "--series", f"e3={paths['e3']}",
                "--band", "e1,e3,0.001"]), 2)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_early_bitwise_passes_and_still_reports_the_drift(self):
        """Early steps gate; later drift is measured, not graded."""
        base = [(1, "8.0", "400.0"), (2, "8.0", "400.0"),
                (3, "7.9", "390.0"), (4, "7.5", "380.0")]
        drifts = base[:3] + [(4, "7.8", "380.0")]
        passed, detail = judge_mod.early_bitwise(base, drifts, 3)
        self.assertTrue(passed)
        self.assertIn("divergence from step 4", detail)
        self.assertIn("first 3/3 steps identical", detail)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_early_bitwise_fails_when_the_first_step_differs(self):
        """A first-step difference is the real defect signal."""
        base = [(1, "8.0", "400.0"), (2, "7.9", "390.0")]
        other = [(1, "8.1", "410.0"), (2, "7.9", "390.0")]
        passed, detail = judge_mod.early_bitwise(base, other, 2)
        self.assertFalse(passed)
        self.assertIn("divergence from step 1", detail)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_early_bitwise_refuses_to_judge_what_it_cannot_see(self):
        """Too few common steps is inconclusive, never a pass."""
        base = [(1, "8.0", "400.0"), (2, "7.9", "390.0")]
        self.assertEqual(judge_mod.early_bitwise(base, base[:1], 2)[0], False)
        self.assertEqual(judge_mod.early_bitwise([], base, 1)[0], False)
        passed, detail = judge_mod.early_bitwise(base, base, 2)
        self.assertTrue(passed)
        self.assertIn("no divergence", detail)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_empty_overlap_is_a_failure_not_a_pass(self):
        with TemporaryDirectory() as tmp:
            empty = Path(tmp) / "empty.log"
            empty.write_text("no steps here\n", encoding="utf-8")
            paths = self._logs(tmp)
            self.assertEqual(judge_mod.main([
                "--series", f"e1={paths['e1']}",
                "--series", f"e0={empty}",
                "--bitwise", "e1,e0"]), 2)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_printed_verdicts_say_what_precision_they_judged(self):
        """A text-log verdict never reads as a bitwise one."""
        with TemporaryDirectory() as tmp:
            paths = self._logs(tmp)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                judge_mod.main(["--series", f"e1={paths['e1']}",
                                "--series", f"e2={paths['e2']}",
                                "--identical", "e1,e2"])
        self.assertIn("printed precision", out.getvalue())
        self.assertIn("inputs unchecked", out.getvalue())


def _runtime_rows(losses, norms, inputs):
    """Rows in the model-integration runtime's metrics.jsonl schema."""
    return [{"step": index + 1, "loss": loss, "grad_norm_pre_clip": norm,
             "grad_norm_post_clip": norm, "lr": 3e-05,
             "global_input_sha256": digest}
            for index, (loss, norm, digest) in enumerate(zip(losses, norms, inputs))]


def _write_runtime(root: Path, rows) -> Path:
    """Lay rows out where the runtime writes them under its output directory."""
    target = root / "cases" / "runtime"
    target.mkdir(parents=True)
    (target / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return root


class TestStructuredEvidence(unittest.TestCase):
    """Full-precision judging on the runtime's structured evidence."""

    _LOSSES = [11.101234567, 11.043123456, 10.907912345]
    _NORMS = [26.755212345, 26.910123456, 27.831012345]
    _INPUTS = ["a1", "b2", "c3"]

    def _judge(self, argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = judge_mod.main(argv)
        return code, out.getvalue()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_identical_runs_pass_at_full_precision(self):
        rows = _runtime_rows(self._LOSSES, self._NORMS, self._INPUTS)
        with TemporaryDirectory() as tmp:
            run_a = _write_runtime(Path(tmp) / "a", rows)
            run_b = _write_runtime(Path(tmp) / "b", rows)
            code, out = self._judge(["--series", f"a={run_a}",
                                     "--series", f"b={run_b}",
                                     "--identical", "a,b"])
        self.assertEqual(code, 0, out)
        self.assertIn("full precision", out)
        self.assertIn("3/3 steps exact", out)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_a_sub_print_difference_passes_the_log_and_fails_the_evidence(self):
        """The reason the structured mode exists, as a positive control.

        Step one differs by one unit in the last place. Printed to six
        significant digits, as the trainer logs it, the runs look identical
        for all three steps; the structured evidence names step one.
        """
        nudged = [math.nextafter(self._LOSSES[0], math.inf)] + self._LOSSES[1:]
        with TemporaryDirectory() as tmp:
            logs = {}
            for name, losses in (("a", self._LOSSES), ("b", nudged)):
                logs[name] = Path(tmp) / f"{name}.log"
                logs[name].write_text("".join(
                    _LINE.format(s=step + 1, l=f"{loss:.6g}", g=f"{norm:.6g}")
                    for step, (loss, norm) in enumerate(zip(losses, self._NORMS))),
                    encoding="utf-8")
            printed, _ = self._judge(["--series", f"a={logs['a']}",
                                      "--series", f"b={logs['b']}",
                                      "--early-identical", "a,b,3"])
            run_a = _write_runtime(Path(tmp) / "ra", _runtime_rows(
                self._LOSSES, self._NORMS, self._INPUTS))
            run_b = _write_runtime(Path(tmp) / "rb", _runtime_rows(
                nudged, self._NORMS, self._INPUTS))
            exact, out = self._judge(["--series", f"a={run_a}",
                                      "--series", f"b={run_b}",
                                      "--early-identical", "a,b,3"])
        self.assertEqual(printed, 0)
        self.assertEqual(exact, 2, out)
        self.assertIn("first difference at step 1", out)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_different_inputs_block_the_verdict_instead_of_grading_it(self):
        """Equal numbers over unequal inputs prove nothing either way."""
        rows_a = _runtime_rows(self._LOSSES, self._NORMS, self._INPUTS)
        rows_b = _runtime_rows(self._LOSSES, self._NORMS, ["a1", "zz", "c3"])
        status, detail = judge_mod.early_exact(rows_a, rows_b, 3)
        self.assertEqual(status, "BLOCKED")
        self.assertIn("inputs differ at steps [2]", detail)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_late_divergence_reports_its_onset_and_growth(self):
        losses_b = self._LOSSES[:2] + [self._LOSSES[2] * (1 + 1e-4)]
        status, detail = judge_mod.early_exact(
            _runtime_rows(self._LOSSES, self._NORMS, self._INPUTS),
            _runtime_rows(losses_b, self._NORMS, self._INPUTS), 2)
        self.assertEqual(status, "PASS")
        self.assertIn("first 2/2 steps exact", detail)
        self.assertIn("first difference at step 3", detail)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_mixed_evidence_is_refused(self):
        """A printed value against a full-precision one grades the formatting."""
        with TemporaryDirectory() as tmp:
            log = Path(tmp) / "a.log"
            log.write_text(_LINE.format(s=1, l="11.1012", g="26.7552"),
                           encoding="utf-8")
            run = _write_runtime(Path(tmp) / "r", _runtime_rows(
                self._LOSSES[:1], self._NORMS[:1], self._INPUTS[:1]))
            code, out = self._judge(["--series", f"a={log}",
                                     "--series", f"b={run}",
                                     "--identical", "a,b"])
        self.assertEqual(code, 2)
        self.assertIn("mixed evidence", out)
