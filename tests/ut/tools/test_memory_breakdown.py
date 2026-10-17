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
"""Unit tests for the memory breakdown parser (CPU, synthetic fixture).

The fixture reproduces the frame shapes of a real
``torch_npu.npu.memory._snapshot()`` taken on torch_npu 2.9.0.post2: the
recorded stack is innermost-first and starts with the capture machinery, the
optimizer and parameter blocks carry Python frames, and the gradient blocks
carry none at all because the autograd engine allocates them from C++.
"""
from __future__ import annotations

import json
import pickle
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


def _find_scripts() -> Path | None:
    """Locate the memory-analysis scripts directory, or None when absent.

    The scripts live under ``.agent/skills`` (agent tooling, not a product
    module), so they are reached by path rather than by import. The directory
    is searched for among this file's ancestors instead of at a fixed depth: a
    checkout has it beside ``tests/``, while a test tree copied on its own - as
    the gate does - has no ``.agent`` sibling at all.
    """
    for ancestor in Path(__file__).resolve().parents:
        candidate = ancestor / ".agent" / "skills" / "memory-analysis" / "scripts"
        if (candidate / "breakdown.py").is_file():
            return candidate
    return None


_SCRIPTS = _find_scripts()
if _SCRIPTS is None:
    raise unittest.SkipTest(
        "memory-analysis scripts (.agent/skills/memory-analysis/scripts) are "
        "not reachable from this test tree; they are agent tooling and are not "
        "copied into a standalone test checkout"
    )
# Ahead of everything else on the path so the script's own sibling imports
# resolve here and not to an unrelated module.
sys.path.insert(0, str(_SCRIPTS))

import breakdown  # noqa: E402  # pylint: disable=wrong-import-position
import collect_snapshot  # noqa: E402  # pylint: disable=wrong-import-position

MIB = 1024 * 1024
SITE = "/opt/py/lib/python3.10/site-packages"

# Every recorded stack starts with the snapshot machinery; `frames[0]` is the
# same for every block and says nothing about the allocation.
_NOISE = [
    {"name": "torch_npu::unwind::unwind()", "filename": "libtorch_npu.so", "line": 0},
    {"name": "torch_npu::CapturedTraceback::gather(bool, bool, bool)",
     "filename": "libtorch_npu.so", "line": 0},
    {"name": "??", "filename": "libtorch_npu.so", "line": 0},
]
# Allocator plumbing every device allocation goes through.
_PLUMBING = [
    {"name": "at::_ops::empty_strided::redispatch(c10::DispatchKeySet)",
     "filename": "libtorch_cpu.so", "line": 0},
    {"name": "c10::Allocator::allocate(unsigned long)", "filename": "libc10.so", "line": 0},
]


def _param_frames() -> list[dict]:
    """A parameter block: module `.to()` conversion, then the caller."""
    return _NOISE + _PLUMBING + [
        {"name": "convert", "filename": f"{SITE}/torch_npu/utils/_module.py", "line": 76},
        {"name": "_apply", "filename": f"{SITE}/torch/nn/modules/module.py", "line": 957},
        {"name": "main", "filename": "/home/u/train.py", "line": 31},
    ]


def _optimizer_frames() -> list[dict]:
    """An optimizer-state block: AdamW allocates through torch/optim/adam.py."""
    return _NOISE + _PLUMBING + [
        {"name": "_init_group", "filename": f"{SITE}/torch/optim/adam.py", "line": 177},
        {"name": "step", "filename": f"{SITE}/torch/optim/optimizer.py", "line": 485},
        {"name": "main", "filename": "/home/u/train.py", "line": 41},
    ]


def _gradient_frames() -> list[dict]:
    """A gradient block: no Python frame at all, only the autograd engine."""
    return _NOISE + _PLUMBING + [
        {"name": "torch::autograd::AccumulateGrad::apply(std::vector<at::Tensor>&&)",
         "filename": "libtorch_cpu.so", "line": 0},
        {"name": "torch::autograd::Engine::thread_main(std::shared_ptr<GraphTask> const&)",
         "filename": "libtorch_cpu.so", "line": 0},
    ]


def _activation_frames() -> list[dict]:
    """An activation block: the innermost own-code frame is a forward."""
    return _NOISE + _PLUMBING + [
        {"name": "gelu", "filename": f"{SITE}/torch/nn/functional.py", "line": 2060},
        {"name": "forward", "filename": "/home/u/train.py", "line": 22},
        {"name": "main", "filename": "/home/u/train.py", "line": 44},
    ]


def _segment(blocks: list[tuple[int, str, list]], pad: int = 0) -> dict:
    """Build one segment whose fields agree with its blocks."""
    entries = [{"address": 0, "size": size, "requested_size": size,
                "state": state, "frames": frames} for size, state, frames in blocks]
    if pad:
        entries.append({"address": 0, "size": pad, "requested_size": 0,
                        "state": "inactive", "frames": []})
    live = sum(e["size"] for e in entries if e["state"].startswith("active"))
    return {"device": 0, "segment_type": "large", "blocks": entries,
            "total_size": sum(e["size"] for e in entries),
            "allocated_size": live, "active_size": live, "frames": []}


def _snapshot() -> dict:
    """Two segments covering every category the rules recognise."""
    return {"segments": [
        _segment([(8 * MIB, "active_allocated", _param_frames()),
                  (8 * MIB, "active_allocated", _optimizer_frames())], pad=2 * MIB),
        _segment([(8 * MIB, "active_allocated", _gradient_frames()),
                  (4 * MIB, "active_allocated", _activation_frames())]),
    ], "device_traces": []}


class MemoryBreakdownTest(unittest.TestCase):
    """Parse a synthetic snapshot shaped like the real one."""

    def setUp(self) -> None:
        """Write the snapshot and its counter sidecar to a temp dir."""
        self.root = Path(tempfile.mkdtemp())
        self.snapshot = self.root / "memory_snapshot.step2.pickle"
        with self.snapshot.open("wb") as handle:
            pickle.dump(_snapshot(), handle)
        self._write_sidecar(28 * MIB)

    def tearDown(self) -> None:
        """Remove the temp tree."""
        shutil.rmtree(self.root, ignore_errors=True)

    def _write_sidecar(self, peak: int) -> None:
        """(Re)write the allocator counters beside the snapshot."""
        sidecar = self.snapshot.with_suffix(self.snapshot.suffix + ".stats.json")
        sidecar.write_text(json.dumps({
            "device": 0, "memory_reserved": 30 * MIB, "memory_allocated": 28 * MIB,
            "max_memory_allocated": peak, "max_memory_reserved": 30 * MIB,
        }), encoding="utf-8")

    def test_call_site_skips_the_capture_machinery(self):
        """`frames[0]` is the snapshot machinery and must never be the answer."""
        site = breakdown.call_site(_param_frames())
        self.assertNotIn("unwind", site)
        self.assertEqual(site, "train.py:31 main")

    def test_call_site_prefers_own_code_over_framework(self):
        """The innermost frame outside site-packages wins."""
        self.assertEqual(breakdown.call_site(_optimizer_frames()), "train.py:41 main")

    def test_call_site_skips_allocator_plumbing_for_native_blocks(self):
        """With no Python frame, the first non-`at::`/`c10::` symbol is used."""
        self.assertEqual(breakdown.call_site(_gradient_frames()),
                         "torch::autograd::AccumulateGrad::apply")

    def test_categories_match_each_rule(self):
        """Each frame shape lands in its own category."""
        self.assertEqual(breakdown.categorize(_param_frames()), "参数")
        self.assertEqual(breakdown.categorize(_optimizer_frames()), "优化器状态")
        self.assertEqual(breakdown.categorize(_gradient_frames()), "梯度")
        self.assertEqual(breakdown.categorize(_activation_frames()), "激活")

    def test_gradient_is_recognised_without_any_python_frame(self):
        """Gradients carry no Python frame, so the native symbol is the signal."""
        self.assertEqual(breakdown._python_frames(_gradient_frames()), [])
        self.assertEqual(breakdown.categorize(_gradient_frames()), "梯度")

    def test_totals_come_from_the_segment_fields(self):
        """Reserved / allocated / fragmentation use the recorded segment sizes."""
        stats = breakdown.summarize(_snapshot())
        self.assertAlmostEqual(stats["reserved_mib"], 30.0)
        self.assertAlmostEqual(stats["allocated_mib"], 28.0)
        self.assertAlmostEqual(stats["fragmentation_mib"], 2.0)
        self.assertEqual(stats["block_count"], 4)

    def test_category_totals_add_up_to_the_allocated_bytes(self):
        """Every active block lands in exactly one category."""
        stats = breakdown.summarize(_snapshot())
        self.assertAlmostEqual(sum(stats["categories"].values()), stats["allocated_mib"])
        self.assertAlmostEqual(stats["categories"]["优化器状态"], 8.0)
        self.assertAlmostEqual(stats["categories"]["激活"], 4.0)

    def test_checks_pass_when_the_snapshot_is_at_the_peak(self):
        """A snapshot whose total equals the recorded peak satisfies every check."""
        report = breakdown.analyse(self.snapshot)
        self.assertTrue(all(ok for _, ok, _ in report["checks"]),
                        [c for c in report["checks"] if not c[1]])

    def test_peak_check_fails_when_the_snapshot_is_below_the_peak(self):
        """A snapshot taken off-peak must say so rather than pass quietly."""
        self._write_sidecar(40 * MIB)
        failed = [name for name, ok, _ in breakdown.analyse(self.snapshot)["checks"] if not ok]
        self.assertTrue(any("峰值" in name for name in failed))

    def test_peak_is_unknown_without_the_sidecar(self):
        """With no counters recorded, the peak question is reported unanswered."""
        self.snapshot.with_suffix(self.snapshot.suffix + ".stats.json").unlink()
        checks = breakdown.analyse(self.snapshot)["checks"]
        peak = [c for c in checks if "峰值" in c[0]][0]
        self.assertFalse(peak[1])
        self.assertIn("无 max_memory_allocated", peak[2])

    def test_segment_identity_catches_a_corrupt_segment(self):
        """A segment whose blocks do not tile it is reported, not averaged over."""
        snap = _snapshot()
        snap["segments"][0]["total_size"] += 1234
        name, ok, detail = breakdown._segment_identity(snap)
        self.assertFalse(ok)
        self.assertIn("seg0", detail)
        self.assertIn("total_size", name)

    def test_unreadable_snapshot_explains_the_write_only_mode(self):
        """torch_npu writes snapshots mode 0200; the error must say how to fix it."""
        if sys.platform.startswith("win"):
            self.skipTest("POSIX file modes are not enforced the same way on Windows")
        self.snapshot.chmod(0o200)
        with self.assertRaises(PermissionError) as caught:
            breakdown.load_snapshot(self.snapshot)
        self.assertIn("chmod 600", str(caught.exception))

    def test_main_writes_the_report_and_the_json_record(self):
        """The CLI writes both outputs and exits 0 when the checks pass."""
        out_html, out_json = self.root / "r.html", self.root / "r.json"
        code = breakdown.main(["--snapshot", str(self.snapshot), "--out", str(out_html),
                               "--json", str(out_json)])
        self.assertEqual(code, 0)
        record = json.loads(out_json.read_text(encoding="utf-8"))
        self.assertAlmostEqual(record["stats"]["allocated_mib"], 28.0)

    def test_main_exits_nonzero_when_an_invariant_fails(self):
        """An off-peak snapshot must not be reported as a clean parse."""
        self._write_sidecar(40 * MIB)
        self.assertEqual(breakdown.main(
            ["--snapshot", str(self.snapshot), "--out", str(self.root / "r.html")]), 2)

    def test_main_reports_a_missing_snapshot(self):
        """A path that does not exist exits 1."""
        self.assertEqual(breakdown.main(
            ["--snapshot", str(self.root / "nope.pickle"),
             "--out", str(self.root / "r.html")]), 1)

    def test_render_leaves_no_unfilled_markers(self):
        """Every template marker is replaced, with and without a comparison."""
        template = _SCRIPTS.parent / "templates" / "memory-breakdown.html"
        report = breakdown.analyse(self.snapshot)
        self.assertNotIn("<!--", breakdown.render(report, template))
        report["compare"] = breakdown.compare(report["stats"], report["stats"])
        self.assertNotIn("<!--", breakdown.render(report, template))

    def test_compare_reports_no_growth_between_identical_snapshots(self):
        """A steady step pair shows zero growth in every category."""
        stats = breakdown.summarize(_snapshot())
        diff = breakdown.compare(stats, stats)
        self.assertAlmostEqual(diff["allocated_delta_mib"], 0.0)
        self.assertTrue(all(abs(delta) < 1e-9 for delta in diff["categories"].values()))

    def test_compare_attributes_growth_to_its_category(self):
        """A category that grew between steps is named with its delta."""
        baseline = breakdown.summarize(_snapshot())
        grown = _snapshot()
        grown["segments"].append(
            _segment([(16 * MIB, "active_allocated", _activation_frames())]))
        diff = breakdown.compare(baseline, breakdown.summarize(grown))
        self.assertAlmostEqual(diff["allocated_delta_mib"], 16.0)
        self.assertAlmostEqual(diff["categories"]["激活"], 16.0)
        self.assertAlmostEqual(diff["categories"]["参数"], 0.0)

    def test_compare_covers_a_category_absent_from_one_side(self):
        """A category present in only one snapshot still gets a delta."""
        bare = breakdown.summarize(
            {"segments": [_segment([(8 * MIB, "active_allocated", _param_frames())])]})
        diff = breakdown.compare(bare, breakdown.summarize(_snapshot()))
        self.assertAlmostEqual(diff["categories"]["梯度"], 8.0)


class CollectSnapshotCheckTest(unittest.TestCase):
    """A collection is complete only when each dump has its counter sidecar."""

    def setUp(self) -> None:
        """Lay out two tagged dumps with sidecars."""
        self.root = Path(tempfile.mkdtemp())
        self.base = self.root / "memory_snapshot.pickle"
        for tag in ("step2", "step3"):
            dump = self.base.with_name(f"memory_snapshot.{tag}.pickle")
            dump.write_bytes(b"x")
            dump.with_suffix(dump.suffix + ".stats.json").write_text("{}", encoding="utf-8")

    def tearDown(self) -> None:
        """Remove the temp tree."""
        shutil.rmtree(self.root, ignore_errors=True)

    def test_tagged_dumps_are_found(self):
        """Both tagged dumps count as complete."""
        complete, problems = collect_snapshot.check_output(self.base)
        self.assertEqual(len(complete), 2)
        self.assertEqual(problems, [])

    def test_a_dump_without_its_sidecar_is_reported(self):
        """Without the counters the peak cannot be judged, so it is not complete."""
        dump = self.base.with_name("memory_snapshot.step2.pickle")
        dump.with_suffix(dump.suffix + ".stats.json").unlink()
        complete, problems = collect_snapshot.check_output(self.base)
        self.assertEqual(len(complete), 1)
        self.assertIn("stats.json", problems[0])

    def test_no_dump_at_all_is_reported(self):
        """A run whose entry never called the hook is reported, not accepted."""
        complete, problems = collect_snapshot.check_output(self.root / "other.pickle")
        self.assertEqual(complete, [])
        self.assertIn("mem_hook", problems[0])


if __name__ == "__main__":
    unittest.main()
