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
"""Unit tests for the precision dump collector and comparison reader.

The fixtures reproduce the exact shapes the installed msprobe produces on
torch_npu 2.9.0.post2: the config schema with per-task nested options, the
``statistics`` comparison table whose ``MaxRelativeErr`` is a **percent
string**, and the ``tensor`` table where it is a bare ratio and ``Cosine``
exists. Those two formats are why a single naive ``float()`` reader reports
"no divergence" on a statistics table that genuinely diverges.
"""
from __future__ import annotations

import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


def _find_scripts() -> Path | None:
    """Locate the precision scripts directory, or None when absent.

    The scripts live under ``.agent/skills`` (agent tooling, not a product
    module), so they are reached by path rather than by import. The directory
    is searched for among this file's ancestors instead of at a fixed depth: a
    checkout has it beside ``tests/``, while a test tree copied on its own - as
    the gate does - has no ``.agent`` sibling at all.
    """
    for ancestor in Path(__file__).resolve().parents:
        candidate = ancestor / ".agent" / "skills" / "precision" / "scripts"
        if (candidate / "diff_dump.py").is_file():
            return candidate
    return None


_SCRIPTS = _find_scripts()
if _SCRIPTS is None:
    raise unittest.SkipTest(
        "precision scripts (.agent/skills/precision/scripts) are not reachable "
        "from this test tree; they are agent tooling and are not copied into a "
        "standalone test checkout"
    )
sys.path.insert(0, str(_SCRIPTS))

import collect_dump  # noqa: E402  # pylint: disable=wrong-import-position
import diff_dump  # noqa: E402  # pylint: disable=wrong-import-position
import precision_hook  # noqa: E402  # pylint: disable=wrong-import-position

# Column order as msprobe writes it for a `statistics` dump; MaxRelativeErr
# carries a trailing per-cent sign in this mode and nothing else does.
_STAT_HEADER = [
    "NPU Name", "Bench Name", "NPU Dtype", "Bench Dtype", "Max diff",
    "Min diff", "Mean diff", "L2norm diff", "MaxRelativeErr",
    "MinRelativeErr", "MeanRelativeErr", "NormRelativeErr", "Result",
    "Err_message",
]
# ... and for a `tensor` dump, where Cosine exists and the relative error is
# a bare ratio that is not clamped to 1.
_TENSOR_HEADER = [
    "NPU Name", "Bench Name", "NPU Dtype", "Bench Dtype", "Cosine",
    "EucDist", "MaxAbsErr", "MaxRelativeErr", "One Thousandth Err Ratio",
    "Result", "Err_message",
]


def _csv(path: Path, header: list[str], rows: list[list[str]]) -> Path:
    """Write a comparison table fixture and return its path."""
    lines = [",".join(header)]
    lines += [",".join(f'"{c}"' for c in row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _stat_rows() -> list[list[str]]:
    """A real-shaped statistics table: clean, then the perturbed weight.

    Taken from a measured A-vs-C comparison of a 3-layer bf16 MLP where
    layer 1's weight was scaled by 1.01: the weight row diverges first and
    the divergence then propagates through that layer's output into the next
    layer's input, while layer 0 stays clean.
    """
    clean = ["0", "0", "0", "0", "0", "0", "0", "0", "pass", "[]"]
    return [
        ["Functional.linear.0.forward.input.0", "Functional.linear.0.forward.input.0",
         "torch.bfloat16", "torch.bfloat16"] + clean,
        ["Functional.linear.0.forward.output.0", "Functional.linear.0.forward.output.0",
         "torch.bfloat16", "torch.bfloat16"] + clean,
        # a None bias: msprobe writes "unsupported", which is not a divergence
        ["Functional.linear.1.forward.input.2", "Functional.linear.1.forward.input.2",
         "", "", "unsupported", "unsupported", "unsupported", "unsupported",
         "unsupported", "unsupported", "unsupported", "unsupported", "pass", "[]"],
        ["Functional.linear.1.forward.input.1", "Functional.linear.1.forward.input.1",
         "torch.bfloat16", "torch.bfloat16", "0.00048828125", "-0.00048828125",
         "-4.76837158203125e-07", "0.125", "1.1049723756906076%", "0%",
         "0.1%", "0.2%", "pass", "[]"],
        ["Functional.linear.1.forward.output.0", "Functional.linear.1.forward.output.0",
         "torch.bfloat16", "torch.bfloat16", "0.0078125", "-0.0078125",
         "6.103515625e-05", "0.125", "0.7462686567164178%", "0%",
         "0.1%", "0.2%", "pass", "[]"],
    ]


def _tensor_rows() -> list[list[str]]:
    """A real-shaped tensor table for the same comparison."""
    return [
        ["Functional.linear.0.forward.input.0", "Functional.linear.0.forward.input.0",
         "torch.bfloat16", "torch.bfloat16", "1.0", "0.0", "0.0", "0.0", "1.0",
         "pass", "[]"],
        ["Functional.linear.1.forward.input.2", "Functional.linear.1.forward.input.2",
         "", "", "unsupported", "unsupported", "unsupported", "unsupported",
         "unsupported", "pass", "[]"],
        # the cause: inside a 0.99 / 0.02 band, which is why a band drops it
        ["Functional.linear.1.forward.input.1", "Functional.linear.1.forward.input.1",
         "torch.bfloat16", "torch.bfloat16", "0.999998", "0.125",
         "0.00048828125", "0.013333333656", "0.98", "pass", "[]"],
        # the symptom: outside that band, and msprobe still calls it "pass"
        ["Functional.linear.1.forward.output.0", "Functional.linear.1.forward.output.0",
         "torch.bfloat16", "torch.bfloat16", "0.999994", "0.125", "0.015625",
         "8.833333015442", "0.97", "pass", "[]"],
    ]


class TestParseSteps(unittest.TestCase):
    """`step` is a list of steps to dump, so a range must be expanded."""

    def test_range_expands_to_every_step(self):
        """A dumped range must include the interior steps."""
        self.assertEqual(collect_dump.parse_steps("0-2"), [0, 1, 2])

    def test_enumeration_keeps_only_listed_steps(self):
        """A comma list is an enumeration, not a range."""
        self.assertEqual(collect_dump.parse_steps("0,2"), [0, 2])

    def test_single_step(self):
        """A bare number is that one step."""
        self.assertEqual(collect_dump.parse_steps("3"), [3])

    def test_empty_means_every_step(self):
        """msprobe treats an empty list as every step."""
        self.assertEqual(collect_dump.parse_steps(""), [])

    def test_backwards_range_is_rejected(self):
        """msprobe rejects a reversed range; fail before launching the run."""
        with self.assertRaises(ValueError):
            collect_dump.parse_steps("5-2")

    def test_hook_agrees_with_collector(self):
        """Launch side and run side must expand a range identically."""
        for spec in ("0-2", "0,2", "3", "", "1-1"):
            self.assertEqual(collect_dump.parse_steps(spec),
                             precision_hook.parse_steps(spec), spec)


class TestBuildConfig(unittest.TestCase):
    """The config must match msprobe's real schema, not a plausible one."""

    def test_task_options_are_nested_under_the_task_name(self):
        """Per-task options live in a block named after the task."""
        config = precision_hook.build_config(
            Path("/tmp/dump"), [0, 1], "statistics", "L1", [])
        self.assertIn("statistics", config)
        self.assertEqual(config["statistics"]["data_mode"], ["all"])
        self.assertEqual(config["statistics"]["summary_mode"], "statistics")

    def test_no_top_level_scope(self):
        """A top-level `scope` is not in the schema and would be ignored."""
        config = precision_hook.build_config(
            Path("/tmp/dump"), [0], "tensor", "L1", [])
        self.assertNotIn("scope", config)
        self.assertIn("scope", config["tensor"])

    def test_rank_field_is_present(self):
        """`rank` is required to scope a multi-card dump."""
        config = precision_hook.build_config(
            Path("/tmp/dump"), [0], "statistics", "L1", [2, 3])
        self.assertEqual(config["rank"], [2, 3])

    def test_tensor_task_has_no_summary_mode(self):
        """`summary_mode` belongs to statistics only."""
        config = precision_hook.build_config(
            Path("/tmp/dump"), [0], "tensor", "L1", [])
        self.assertNotIn("summary_mode", config["tensor"])

    def test_unknown_task_is_rejected(self):
        """A task msprobe does not know would dump nothing."""
        with self.assertRaises(ValueError):
            precision_hook.build_config(Path("/tmp/d"), [0], "accuracy", "L1", [])

    def test_collector_and_hook_build_the_same_config(self):
        """Both ends must write one schema."""
        args = (Path("/tmp/dump"), [0, 1], "statistics", "L1", [])
        self.assertEqual(collect_dump.build_config(*args),
                         precision_hook.build_config(*args))


class TestDumpPathCheck(unittest.TestCase):
    """msprobe validates the dump path and rejects special characters."""

    def test_brackets_are_rejected(self):
        """A path named after a config value can carry brackets."""
        with self.assertRaises(ValueError):
            collect_dump.check_dump_path(Path("/tmp/run_[0,2]/dump"))

    def test_plain_path_is_accepted(self):
        """An alphanumeric path passes."""
        collect_dump.check_dump_path(Path("/tmp/run_step02/dump"))


class TestToFloat(unittest.TestCase):
    """Both metric formats must land on one scale."""

    def test_percent_string_becomes_a_ratio(self):
        """statistics mode writes a per-cent string."""
        self.assertAlmostEqual(diff_dump.to_float("1.1049723756906076%"),
                               0.011049723756906076, places=12)

    def test_bare_ratio_is_unchanged(self):
        """tensor mode writes a bare ratio."""
        self.assertAlmostEqual(diff_dump.to_float("0.013333333656"),
                               0.013333333656, places=12)

    def test_non_comparable_cells_are_none(self):
        """A non-tensor argument is not a divergence."""
        for cell in ("unsupported", "", "N/A", None):
            self.assertIsNone(diff_dump.to_float(cell), cell)

    def test_nan_is_kept_not_skipped(self):
        """A non-finite metric is the failure to find, never a skipped cell."""
        self.assertTrue(math.isnan(diff_dump.to_float("nan")))

    def test_zero_parses_as_zero_not_none(self):
        """Zero must stay comparable: it is the bitwise-equal case."""
        self.assertEqual(diff_dump.to_float("0"), 0.0)


class TestTableMode(unittest.TestCase):
    """The metric columns differ by dump task."""

    def test_cosine_means_tensor(self):
        """Only a tensor dump compares with Cosine."""
        self.assertEqual(diff_dump.table_mode(_TENSOR_HEADER), "tensor")

    def test_no_cosine_means_statistics(self):
        """A statistics table has diff columns and no Cosine."""
        self.assertEqual(diff_dump.table_mode(_STAT_HEADER), "statistics")


class TestScan(unittest.TestCase):
    """The scan must report the first cause, on both table shapes."""

    def setUp(self) -> None:
        """Write one table of each shape into a temporary directory."""
        self.tmp = Path(tempfile.mkdtemp(prefix="precision_ut_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.stat = _csv(self.tmp / "stat.csv", _STAT_HEADER, _stat_rows())
        self.tensor = _csv(self.tmp / "tensor.csv", _TENSOR_HEADER, _tensor_rows())

    def test_statistics_divergence_is_found_despite_percent_cells(self):
        """The percent format must not read out as 'not comparable'."""
        summary = diff_dump.scan(self.stat, None, None)
        self.assertEqual(summary["mode"], "statistics")
        self.assertIsNotNone(summary["first"])
        self.assertEqual(summary["first"]["name"],
                         "Functional.linear.1.forward.input.1")

    def test_tensor_divergence_reports_the_same_first_cause(self):
        """Both dump tasks must point at the same row."""
        summary = diff_dump.scan(self.tensor, None, None)
        self.assertEqual(summary["mode"], "tensor")
        self.assertEqual(summary["first"]["name"],
                         "Functional.linear.1.forward.input.1")

    def test_clean_table_reports_no_divergence(self):
        """An identical pair must not be flagged."""
        clean = _csv(self.tmp / "clean.csv", _STAT_HEADER, _stat_rows()[:2])
        self.assertIsNone(diff_dump.scan(clean, None, None)["first"])

    def test_unsupported_rows_are_not_divergence(self):
        """A None bias row must be skipped, not reported."""
        only_unsupported = _csv(self.tmp / "u.csv", _STAT_HEADER,
                                [_stat_rows()[2]])
        summary = diff_dump.scan(only_unsupported, None, None)
        self.assertIsNone(summary["first"])
        self.assertEqual(summary["comparable"], 0)

    def test_band_reports_the_symptom_instead_of_the_cause(self):
        """Why the default is bitwise: the old band drops the real cause.

        The perturbed weight sits at Cosine 0.999998 / MaxRelativeErr 0.0133,
        inside a 0.99 / 0.02 band, while its propagated output sits outside -
        so a banded scan reports the downstream row.
        """
        banded = diff_dump.scan(self.tensor, 0.99, 0.02)
        self.assertEqual(banded["first"]["name"],
                         "Functional.linear.1.forward.output.0")
        bitwise = diff_dump.scan(self.tensor, None, None)
        self.assertLess(bitwise["first"]["order"], banded["first"]["order"])

    def test_msprobe_result_column_is_carried_but_not_trusted(self):
        """msprobe says 'pass' on a diverging row; report it, do not gate."""
        summary = diff_dump.scan(self.tensor, None, None)
        self.assertEqual(summary["first"]["msprobe_result"], "pass")

    def test_diverging_count_covers_propagation(self):
        """Both the cause and its downstream rows are counted."""
        summary = diff_dump.scan(self.stat, None, None)
        self.assertEqual(summary["diverging"], 2)

    def test_nan_metric_is_reported_under_any_band(self):
        """A non-finite metric is reported even inside a loose band.

        Skipping it would hide a nan, which is the first thing this skill is
        asked to localize.
        """
        row = ["Functional.linear.1.forward.output.0",
               "Functional.linear.1.forward.output.0",
               "torch.bfloat16", "torch.bfloat16", "nan", "nan", "nan", "nan",
               "nan%", "0%", "0.1%", "0.2%", "pass", "[]"]
        table = _csv(self.tmp / "nan.csv", _STAT_HEADER, [row])
        for band in ((None, None), (0.99, 1e9)):
            summary = diff_dump.scan(table, *band)
            self.assertIsNotNone(summary["first"], band)
            self.assertEqual(summary["first"]["name"],
                             "Functional.linear.1.forward.output.0")


class TestStepDir(unittest.TestCase):
    """`msprobe compare` wants a step directory, not the dump root."""

    def setUp(self) -> None:
        """Create an empty dump root to populate per test."""
        self.tmp = Path(tempfile.mkdtemp(prefix="precision_ut_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.dump = self.tmp / "dump"
        self.dump.mkdir()

    def _step(self, name: str) -> Path:
        path = self.dump / name
        path.mkdir()
        (path / "dump.json").write_text("{}", encoding="utf-8")
        return path

    def test_single_step_is_resolved_from_the_root(self):
        """One step: descend without being told which."""
        step = self._step("step0")
        self.assertEqual(diff_dump.step_dir(self.dump, None), step)

    def test_step_directory_is_returned_as_is(self):
        """An already-resolved step directory passes through."""
        step = self._step("step0")
        self.assertEqual(diff_dump.step_dir(step, None), step)

    def test_several_steps_require_an_explicit_choice(self):
        """Comparing the wrong step silently compares the wrong data."""
        self._step("step0")
        self._step("step1")
        with self.assertRaises(ValueError):
            diff_dump.step_dir(self.dump, None)

    def test_named_step_is_selected(self):
        """A bare number and the `stepN` form both resolve."""
        self._step("step0")
        step1 = self._step("step1")
        self.assertEqual(diff_dump.step_dir(self.dump, "1"), step1)
        self.assertEqual(diff_dump.step_dir(self.dump, "step1"), step1)

    def test_missing_named_step_is_reported(self):
        """Name the steps that exist rather than failing bare."""
        self._step("step0")
        with self.assertRaises(FileNotFoundError):
            diff_dump.step_dir(self.dump, "7")

    def test_empty_dump_is_reported_as_no_dump(self):
        """An empty dump is the silent-failure case; say so."""
        with self.assertRaises(FileNotFoundError):
            diff_dump.step_dir(self.dump, None)


class TestVerify(unittest.TestCase):
    """A run that exits 0 having dumped nothing is a failure."""

    def setUp(self) -> None:
        """Name a dump root that tests create on demand."""
        self.tmp = Path(tempfile.mkdtemp(prefix="precision_ut_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.dump = self.tmp / "dump"

    def _step(self, name: str, complete: bool = True) -> None:
        path = self.dump / name
        path.mkdir(parents=True)
        if complete:
            (path / "dump.json").write_text("{}", encoding="utf-8")

    def test_missing_dump_reports_every_step_missing(self):
        """No dump at all: name what was expected."""
        present, missing = collect_dump.verify(self.dump, [0, 1])
        self.assertEqual(present, [])
        self.assertEqual(missing, ["step0", "step1"])

    def test_interior_step_gap_is_caught(self):
        """The `[0, 2]` mistake shows up as a missing step1."""
        self._step("step0")
        self._step("step2")
        present, missing = collect_dump.verify(self.dump, [0, 1, 2])
        self.assertEqual(present, ["step0", "step2"])
        self.assertEqual(missing, ["step1"])

    def test_step_without_dump_json_is_not_present(self):
        """A directory created but never finalised is not a dump."""
        self._step("step0", complete=False)
        present, missing = collect_dump.verify(self.dump, [0])
        self.assertEqual(present, [])
        self.assertEqual(missing, ["step0"])

    def test_all_steps_present(self):
        """The passing case."""
        self._step("step0")
        self._step("step1")
        present, missing = collect_dump.verify(self.dump, [0, 1])
        self.assertEqual(present, ["step0", "step1"])
        self.assertEqual(missing, [])

    def test_no_steps_configured_accepts_any_complete_step(self):
        """An empty step list means every step; one landing is enough."""
        self._step("step0")
        present, missing = collect_dump.verify(self.dump, [])
        self.assertEqual(present, ["step0"])
        self.assertEqual(missing, [])


class TestDumpEnv(unittest.TestCase):
    """The exported variables are the live interface the hook reads."""

    def test_window_is_carried_to_the_run(self):
        """Launch side and run side must agree on one window."""
        env = collect_dump.dump_env(Path("/tmp/c.json"), Path("/tmp/dump"),
                                    [0, 1], "statistics", "L1", [3])
        self.assertEqual(env["PRECISION_DUMP_STEPS"], "0,1")
        self.assertEqual(env["PRECISION_DUMP_TASK"], "statistics")
        self.assertEqual(env["PRECISION_DUMP_LEVEL"], "L1")
        self.assertEqual(env["PRECISION_DUMP_RANKS"], "3")
        self.assertEqual(env["PRECISION_DUMP_PATH"], str(Path("/tmp/dump")))

    def test_hook_reads_the_exported_steps(self):
        """The variable the collector writes is the one the hook parses."""
        env = collect_dump.dump_env(Path("/tmp/c.json"), Path("/tmp/dump"),
                                    [0, 1, 2], "tensor", "L1", [])
        self.assertEqual(precision_hook.parse_steps(env["PRECISION_DUMP_STEPS"]),
                         [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
