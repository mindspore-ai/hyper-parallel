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
"""Unit tests for the profiling breakdown parser (CPU, synthetic fixture).

The fixture reproduces the column names and file layout of a real
``torch_npu.profiler`` output set collected on CANN 9.0.0 / torch_npu 2.9.0,
including the two shapes that are easy to parse wrongly: HCCL's AI-CPU launch
kernel, which must not be reported as an operator fallback, and
``communication.json``'s ``Total Op Info`` aggregate, which must not be counted
as one more collective.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


def _find_scripts() -> Path | None:
    """Locate the profiling skill's scripts directory, or None when absent.

    The scripts live under ``.agent/skills`` (agent tooling, not a product
    module), so they are reached by path rather than by import. The directory
    is searched for among this file's ancestors instead of at a fixed depth: a
    checkout has it beside ``tests/``, while a test tree copied on its own - as
    the gate does - has no ``.agent`` sibling at all.
    """
    for ancestor in Path(__file__).resolve().parents:
        candidate = ancestor / ".agent" / "skills" / "profiling" / "scripts"
        if (candidate / "breakdown.py").is_file():
            return candidate
    return None


_SCRIPTS = _find_scripts()
if _SCRIPTS is None:
    raise unittest.SkipTest(
        "profiling skill scripts (.agent/skills/profiling/scripts) are not "
        "reachable from this test tree; they are agent tooling and are not "
        "copied into a standalone test checkout"
    )
# Ahead of everything else on the path so the script's own sibling imports
# resolve here and not to an unrelated module.
sys.path.insert(0, str(_SCRIPTS))

import breakdown  # noqa: E402  # pylint: disable=wrong-import-position
import collect_prof  # noqa: E402  # pylint: disable=wrong-import-position

# Real header, verbatim. `Stage` is `Computing + Communication(Not Overlapped)
# + Free`; `Communication`, `Overlapped`, `Bubble` and `Preparing` sit outside
# that sum.
_STEP_TRACE = """\
Device_id,Step,Computing,Communication(Not Overlapped),Overlapped,Communication,Free,Stage,\
Bubble,Communication(Not Overlapped and Exclude Receive),Preparing
0,2,100.0,10.0,40.0,50.0,5.0,115.0,0,10.0,3.0
0,3,200.0,20.0,30.0,50.0,10.0,230.0,0,20.0,4.0
"""

# `allreduceAicpuKernel` is HCCL's per-collective launch kernel: AI CPU, but
# communication rather than an operator that fell back.
_OP_STATISTIC = """\
Device_id,OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)
0,MatMulV3,AI_CORE,4,400.0,90.0,100.0,110.0,40.0
0,Sort,AI_CPU,2,300.0,140.0,150.0,160.0,30.0
0,allreduceAicpuKernel,AI_CPU,6,200.0,30.0,33.3,40.0,20.0
0,Mul,AI_VECTOR_CORE,4,100.0,20.0,25.0,30.0,10.0
"""

# The optimizer range carries its device time in `Device Total Duration(us)`;
# `Device Self Duration(us)` is 0 because it is a parent range. The first row
# keeps the real file's shape notation - a CSV-quoted field whose own content
# is quoted - so the reader is exercised against it.
_OPERATOR_DETAILS = "\n".join([
    "Name,Input Shapes,Call Stack,Host Self Duration(us),Host Total Duration(us),"
    "Device Self Duration(us),Device Total Duration(us),"
    "Device Self Duration With AICore(us),Device Total Duration With AICore(us)",
    'aten::mm,"""1,2048,128""",,1.0,2.0,0,10.0,0,10.0',
    "Optimizer.step#AdamW.step,,,5.0,20.0,0,60.0,0,60.0",
    "Optimizer.step#AdamW.step,,,5.0,20.0,0,40.0,0,40.0",
    "",
])


def _comm_payload() -> dict:
    """Two steps, one collective each, plus the per-step `Total Op Info`."""
    def step(elapse_ms: float, wait_ms: float, tag: str) -> dict:
        """One step's collective block: the aggregate plus a single allreduce."""
        return {
            "p2p": {},
            "collective": {
                "Total Op Info": {
                    "Communication Time Info": {
                        "Elapse Time(ms)": elapse_ms, "Wait Time(ms)": wait_ms,
                    },
                },
                f"hcom_allReduce__1_{tag}_1@999": {
                    "Communication Time Info": {
                        "Elapse Time(ms)": elapse_ms, "Wait Time(ms)": wait_ms,
                    },
                    "Communication Bandwidth Info": {
                        "SDMA": {"Transit Time(ms)": 0.01, "Transit Size(MB)": 2.0},
                        "RDMA": {"Transit Time(ms)": 0, "Transit Size(MB)": 0},
                    },
                },
            },
        }
    return {"step2": step(0.05, 0.04, "0"), "step3": step(0.05, 0.04, "1")}


def _info_payload(warmup: int = 2, active: int = 3) -> dict:
    """A `profiler_info_<rank>.json` body with the fields the parser reads."""
    return {
        "config": {
            "common_config": {"schedule": {"wait": 0, "active": active,
                                           "warmup": warmup, "repeat": 1}},
            "experimental_config": {"_profiler_level": "Level1"},
        },
        "torch_npu_version": "2.9.0.post2",
        "cann_version": "9.0.0",
        "rank_id": "0",
    }


class ProfilingBreakdownTest(unittest.TestCase):
    """Parse a synthetic output set shaped like the real one."""

    def setUp(self) -> None:
        """Lay out `<run>_ascend_pt/ASCEND_PROFILER_OUTPUT` under a temp dir."""
        self.root = Path(tempfile.mkdtemp())
        self.run_dir = self.root / "host_123_20260101000000000_ascend_pt"
        self.out_dir = self.run_dir / "ASCEND_PROFILER_OUTPUT"
        self.out_dir.mkdir(parents=True)
        (self.out_dir / "step_trace_time.csv").write_text(_STEP_TRACE, encoding="utf-8")
        (self.out_dir / "op_statistic.csv").write_text(_OP_STATISTIC, encoding="utf-8")
        (self.out_dir / "operator_details.csv").write_text(_OPERATOR_DETAILS, encoding="utf-8")
        (self.out_dir / "communication.json").write_text(
            json.dumps(_comm_payload()), encoding="utf-8")
        self._write_info()
        (self.run_dir / "profiler_metadata.json").write_text(json.dumps(
            {"parallel_group_info": {"group_name_0": {
                "group_name": "default_group", "global_ranks": [0, 1]}}}), encoding="utf-8")

    def tearDown(self) -> None:
        """Remove the temp tree."""
        shutil.rmtree(self.root, ignore_errors=True)

    def _write_info(self, **kwargs) -> None:
        """(Re)write `profiler_info_0.json` with an optional schedule override."""
        (self.run_dir / "profiler_info_0.json").write_text(
            json.dumps(_info_payload(**kwargs)), encoding="utf-8")

    def test_find_output_dirs_accepts_the_three_input_shapes(self):
        """The output dir, the run dir and their parent all resolve the same."""
        for candidate in (self.out_dir, self.run_dir, self.root):
            self.assertEqual(breakdown.find_output_dirs(candidate), [self.out_dir])

    def test_find_output_dirs_is_empty_when_unreachable(self):
        """An unrelated directory yields no output dirs rather than raising."""
        self.assertEqual(breakdown.find_output_dirs(self.root / "nope"), [])

    def test_step_trace_averages_over_steps_and_checks_the_identity(self):
        """Parts are per-step means and Stage equals their sum."""
        trace = breakdown.parse_step_trace(self.out_dir)
        self.assertEqual(trace["steps"], 2)
        self.assertAlmostEqual(trace["parts"]["计算"], 150.0)
        self.assertAlmostEqual(trace["parts"]["通信（未掩盖）"], 15.0)
        self.assertAlmostEqual(trace["parts"]["Free"], 7.5)
        self.assertAlmostEqual(trace["context"]["Stage"], 172.5)
        self.assertAlmostEqual(trace["identity_drift_us"], 0.0)

    def test_step_trace_keeps_overlapped_outside_the_additive_sum(self):
        """`Overlapped` and `Communication` are context, not parts."""
        trace = breakdown.parse_step_trace(self.out_dir)
        self.assertNotIn("Overlapped", trace["parts"])
        self.assertAlmostEqual(trace["context"]["Overlapped"], 35.0)
        self.assertAlmostEqual(trace["context"]["Communication"], 50.0)

    def test_optimizer_reads_device_total_once_per_step(self):
        """The `Optimizer.step#` range gives one row and one total per step."""
        opt = breakdown.parse_optimizer(self.out_dir)
        self.assertEqual(opt["name"], "Optimizer.step#AdamW.step")
        self.assertEqual(opt["rows"], 2)
        self.assertAlmostEqual(opt["per_step_us"], 50.0)

    def test_comm_launch_kernel_is_excluded_from_fallbacks(self):
        """HCCL's AI-CPU kernel counts in the class but is not a fallback."""
        compute = breakdown.parse_compute(self.out_dir)
        self.assertAlmostEqual(compute["classes"]["AI_CPU"], 500.0)
        self.assertEqual([op for op, _, _ in compute["fallbacks"]], ["Sort"])

    def test_compute_classes_map_every_core_type(self):
        """Each observed `Core Type` lands in its own labelled class."""
        classes = breakdown.parse_compute(self.out_dir)["classes"]
        self.assertAlmostEqual(classes["Cube (AI_CORE)"], 400.0)
        self.assertAlmostEqual(classes["Vector (AI_VECTOR_CORE)"], 100.0)

    def test_communication_uses_the_aggregate_but_does_not_count_it(self):
        """`Total Op Info` supplies the totals and is not counted as an op."""
        comm = breakdown.parse_communication(self.out_dir)
        self.assertEqual(comm["steps"], 2)
        self.assertEqual(comm["ops"], 2)
        self.assertAlmostEqual(comm["elapse_us"], 50.0)
        self.assertAlmostEqual(comm["wait_us"], 40.0)
        self.assertEqual(comm["groups"], ["999"])

    def test_communication_drops_idle_links(self):
        """A link that moved no bytes is left out of the bandwidth table."""
        links = breakdown.parse_communication(self.out_dir)["links"]
        self.assertIn("SDMA", links)
        self.assertNotIn("RDMA", links)

    def test_provenance_reads_versions_schedule_and_groups(self):
        """Versions, schedule and parallel groups come from the two JSONs."""
        prov = breakdown.parse_provenance(self.run_dir)
        self.assertEqual(prov["cann"], "9.0.0")
        self.assertEqual(prov["rank"], "0")
        self.assertEqual(prov["schedule"]["warmup"], 2)
        self.assertEqual(prov["groups"], {"default_group": [0, 1]})

    def test_checks_all_pass_on_a_consistent_set(self):
        """A consistent fixture satisfies every invariant."""
        report = breakdown.analyse(self.out_dir)
        self.assertTrue(all(ok for _, ok, _ in report["checks"]))

    def test_checks_flag_a_collection_that_skipped_no_warmup(self):
        """`warmup=0` and `active=1` are reported as failed invariants."""
        self._write_info(warmup=0, active=1)
        failed = [name for name, ok, _ in breakdown.analyse(self.out_dir)["checks"] if not ok]
        self.assertIn("采集跳过了 warmup", failed)
        self.assertIn("稳态步数 >= 2（单步不可靠）", failed)

    def test_cross_file_check_fails_when_the_two_totals_disagree(self):
        """A communication.json that contradicts step_trace is flagged."""
        payload = _comm_payload()
        payload["step2"]["collective"]["Total Op Info"][
            "Communication Time Info"]["Elapse Time(ms)"] = 5.0
        (self.out_dir / "communication.json").write_text(json.dumps(payload), encoding="utf-8")
        failed = [name for name, ok, _ in breakdown.analyse(self.out_dir)["checks"] if not ok]
        self.assertIn("communication.json Elapse = step_trace Communication", failed)

    def test_missing_optional_files_degrade_instead_of_raising(self):
        """A non-distributed run has no communication.json; parsing still works."""
        (self.out_dir / "communication.json").unlink()
        report = breakdown.analyse(self.out_dir)
        self.assertEqual(report["comm"], {})
        self.assertTrue(report["trace"])

    def test_render_leaves_no_unfilled_markers(self):
        """Every template marker is replaced, including the optional sections."""
        template = _SCRIPTS.parent / "templates" / "perf-breakdown.html"
        html = breakdown.render(breakdown.analyse(self.out_dir), template)
        self.assertNotIn("<!--", html)

    def test_verdict_names_the_fallback_as_the_largest_movable_item(self):
        """With a 300us AI-CPU Sort, the verdict points at that operator."""
        template = _SCRIPTS.parent / "templates" / "perf-breakdown.html"
        html = breakdown.render(breakdown.analyse(self.out_dir), template)
        self.assertIn("Sort", html.split("最大可动项")[1][:200])

    def test_main_writes_the_report_and_the_json_record(self):
        """The CLI writes both outputs and exits 0 when the checks pass."""
        out_html, out_json = self.root / "r.html", self.root / "r.json"
        code = breakdown.main(["--prof-dir", str(self.root), "--out", str(out_html),
                               "--json", str(out_json)])
        self.assertEqual(code, 0)
        self.assertTrue(out_html.is_file())
        record = json.loads(out_json.read_text(encoding="utf-8"))
        self.assertEqual(len(record), 1)
        self.assertAlmostEqual(record[0]["optimizer"]["per_step_us"], 50.0)

    def test_main_exits_nonzero_when_an_invariant_fails(self):
        """A failed check must not be reported as a clean parse."""
        self._write_info(warmup=0, active=1)
        code = breakdown.main(["--prof-dir", str(self.root),
                               "--out", str(self.root / "r.html")])
        self.assertEqual(code, 2)

    def test_main_reports_a_directory_with_no_profiler_output(self):
        """An input with no ASCEND_PROFILER_OUTPUT exits 1."""
        empty = self.root / "empty"
        empty.mkdir()
        self.assertEqual(breakdown.main(
            ["--prof-dir", str(empty), "--out", str(self.root / "r.html")]), 1)


class CollectOutputCheckTest(unittest.TestCase):
    """A collection is complete only when the profiler's own markers are there."""

    def setUp(self) -> None:
        """Lay out an output directory with both completion markers."""
        self.root = Path(tempfile.mkdtemp())
        self.out = self.root / "run_ascend_pt" / "ASCEND_PROFILER_OUTPUT"
        self.out.mkdir(parents=True)
        for name in collect_prof.REQUIRED:
            (self.out / name).write_text("", encoding="utf-8")

    def tearDown(self) -> None:
        """Remove the temp tree."""
        shutil.rmtree(self.root, ignore_errors=True)

    def test_a_complete_set_reports_no_problem(self):
        """Both markers present means the set is usable."""
        complete, problems = collect_prof.check_output(self.root)
        self.assertEqual(complete, [self.out])
        self.assertEqual(problems, [])

    def test_a_non_empty_directory_is_not_enough(self):
        """Output that never finished parsing is reported, not accepted."""
        (self.out / "analyse.done").unlink()
        complete, problems = collect_prof.check_output(self.root)
        self.assertEqual(complete, [])
        self.assertIn("analyse.done", problems[0])

    def test_level0_output_without_step_trace_is_rejected(self):
        """Below Level1 there is no step_trace_time.csv, so there is no breakdown."""
        (self.out / "step_trace_time.csv").unlink()
        complete, problems = collect_prof.check_output(self.root)
        self.assertEqual(complete, [])
        self.assertIn("step_trace_time.csv", problems[0])

    def test_missing_output_dir_is_reported(self):
        """A run that produced nothing is reported rather than passing silently."""
        complete, problems = collect_prof.check_output(self.root / "nothing")
        self.assertEqual(complete, [])
        self.assertIn("ASCEND_PROFILER_OUTPUT", problems[0])


if __name__ == "__main__":
    unittest.main()
