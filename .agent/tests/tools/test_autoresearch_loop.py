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
"""Unit tests for the autoresearch loop engine (CPU, temp git repo).

The benchmark command is a tiny script that prints whatever metric the test
writes into ``metric.txt``, so keep/discard/crash paths are driven
deterministically without any accelerator.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

def _find_engine() -> Path | None:
    """Locate the autoresearch engine directory, or None when it is absent.

    The engine lives under ``.agent/skills`` (it is agent tooling, not a
    product module), so it is reached by path rather than by import. The
    directory is searched for among this file's ancestors instead of at a
    fixed depth: a checkout has it beside ``tests/``, while a test tree
    copied on its own - as the gate does - has no ``.agent`` sibling at all.
    """
    for ancestor in Path(__file__).resolve().parents:
        candidate = ancestor / ".agent" / "skills" / "autoresearch" / "scripts"
        if (candidate / "loop.py").is_file():
            return candidate
    return None


_ENGINE = _find_engine()
if _ENGINE is None:
    raise unittest.SkipTest(
        "autoresearch engine (.agent/skills/autoresearch/scripts) is not "
        "reachable from this test tree; the engine is agent tooling and is "
        "not copied into a standalone test checkout"
    )
# The standalone engine uses sibling imports and has no importable package.
sys.path.insert(0, str(_ENGINE))  # pylint: disable=sys-path-mutation

from config import load_run_config  # noqa: E402  # pylint: disable=wrong-import-position
from loop import (  # noqa: E402  # pylint: disable=wrong-import-position
    best_recorded_metric,
    iterate,
    pending_retries,
    record_baseline,
    run_status,
)

_BENCH_SCRIPT = """\
import pathlib
print(pathlib.Path("metric.txt").read_text().strip())
"""

# Pops one line per call so consecutive benchmark runs see different metrics.
_SEQ_SCRIPT = """\
import pathlib
import sys
queue = pathlib.Path(sys.argv[1])
lines = [line for line in queue.read_text().splitlines() if line]
print(lines[0])
if len(lines) > 1:
    queue.write_text("\\n".join(lines[1:]) + "\\n")
"""


class AutoresearchLoopTest(unittest.TestCase):
    """Exercise baseline / keep / discard / crash against a temp repo."""

    def setUp(self) -> None:
        """Create a git repo with one target file and a scripted benchmark."""
        self.repo = Path(tempfile.mkdtemp())
        self._git("init", "-q", "-b", "main")
        self._git("config", "user.email", "test@example.com")
        self._git("config", "user.name", "test")
        (self.repo / "target.py").write_text("STATE = 0\n", encoding="utf-8")
        (self.repo / "bench.py").write_text(_BENCH_SCRIPT, encoding="utf-8")
        (self.repo / "metric.txt").write_text("total_ms=100 engaged=1\n", encoding="utf-8")
        run_dir = self.repo / "run"
        run_dir.mkdir()
        (run_dir / "run.json").write_text(json.dumps({
            "target_files": ["target.py"],
            "benchmark_cmd": [sys.executable, "bench.py"],
            "bench_must_match": ["engaged=1"],
        }), encoding="utf-8")
        self._git("add", "-A")
        self._git("commit", "-q", "-m", "init")
        self.config = load_run_config(run_dir, self.repo)

    def tearDown(self) -> None:
        """Drop the temp repo."""
        shutil.rmtree(self.repo, ignore_errors=True)

    def _git(self, *args: str) -> str:
        """Run git inside the temp repo."""
        done = subprocess.run(["git", "-C", str(self.repo), *args],
                              capture_output=True, text=True, check=True)
        return done.stdout.strip()

    def _set_metric(self, line: str, commit: bool = True) -> None:
        """Rewrite the scripted benchmark output."""
        (self.repo / "metric.txt").write_text(line + "\n", encoding="utf-8")
        if commit:
            self._git("add", "-A")
            self._git("commit", "-q", "-m", "metric update")

    def _edit_target(self, value: int) -> None:
        """Leave a pending edit in the target file."""
        (self.repo / "target.py").write_text(f"STATE = {value}\n", encoding="utf-8")

    def test_baseline_records_first_keep(self):
        """The baseline becomes the best recorded metric."""
        outcome = record_baseline(self.config)
        self.assertEqual(outcome.status, "keep")
        self.assertEqual(best_recorded_metric(self.config), 100.0)

    def test_baseline_checks_configured_gate(self):
        """A failing baseline gate blocks the benchmark and leaves no best."""
        self.config.gate_cmd = [sys.executable, "-c", "raise SystemExit(1)"]
        self.config.benchmark_cmd = ["unused-baseline-benchmark"]
        outcome = record_baseline(self.config)
        self.assertEqual(outcome.status, "crash")
        self.assertEqual(outcome.detail, "gate failed")
        self.assertIsNone(best_recorded_metric(self.config))

    def test_improvement_is_kept_and_committed(self):
        """A beyond-noise improvement keeps the experiment commit."""
        record_baseline(self.config)
        self._set_metric("total_ms=80 engaged=1")
        self._edit_target(1)
        outcome = iterate(self.config, "faster")
        self.assertEqual(outcome.status, "keep")
        self.assertEqual(best_recorded_metric(self.config), 80.0)
        self.assertIn("STATE = 1", (self.repo / "target.py").read_text(encoding="utf-8"))
        self.assertIn("[autoresearch] keep: faster", self._git("log", "--oneline", "-2"))

    def test_within_noise_is_discarded_and_restored(self):
        """A within-noise result restores the target via a forward commit."""
        record_baseline(self.config)
        self._set_metric("total_ms=99.5 engaged=1")
        self._edit_target(2)
        outcome = iterate(self.config, "noise")
        self.assertEqual(outcome.status, "discard")
        self.assertIn("STATE = 0", (self.repo / "target.py").read_text(encoding="utf-8"))
        log = self._git("log", "--oneline", "-3")
        self.assertIn("[autoresearch] discard: noise", log)
        self.assertIn("[autoresearch] exp: noise", log)
        self.assertEqual(best_recorded_metric(self.config), 100.0)

    def test_missing_positive_marker_is_a_crash(self):
        """Without the engagement marker the measurement is void."""
        record_baseline(self.config)
        self._set_metric("total_ms=10")
        self._edit_target(3)
        outcome = iterate(self.config, "no marker")
        self.assertEqual(outcome.status, "crash")
        self.assertIn("STATE = 0", (self.repo / "target.py").read_text(encoding="utf-8"))
        self.assertEqual(best_recorded_metric(self.config), 100.0)

    def test_deferred_candidate_requires_a_matching_keep(self):
        """Only a later matching keep makes a deferred edit eligible again."""
        record_baseline(self.config)
        self.assertEqual(pending_retries(self.config), [])
        self._set_metric("total_ms=99.5 engaged=1")
        parent = self._git("rev-parse", "HEAD")
        self._edit_target(1)
        iterate(self.config, "prefetch", changes=("prefetch",), revisit_on=("optimizer",))
        ledger = self.config.run_dir / "candidates.jsonl"
        receipt = json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])
        deferred = receipt["commit"]
        self.assertEqual(receipt["parent"], parent)
        self.assertEqual(receipt["status"], "discard")
        self.assertEqual(receipt["stage"], "decision")
        self._edit_target(2)
        before_retry = self._git("rev-parse", "HEAD")
        with self.assertRaisesRegex(RuntimeError, "not due"):
            iterate(self.config, "premature", retry_of=deferred)
        self.assertEqual(self._git("rev-parse", "HEAD"), before_retry)
        self.assertEqual(len(ledger.read_text(encoding="utf-8").splitlines()), 1)
        self._git("restore", "--", "target.py")

        self._set_metric("total_ms=90")
        self._edit_target(3)
        iterate(self.config, "failed optimizer", changes=("optimizer",))
        self._set_metric("total_ms=90 engaged=1")
        self._edit_target(4)
        iterate(self.config, "unrelated keep", changes=("attention",))
        self.assertEqual(pending_retries(self.config), [])

        self._set_metric("total_ms=80 engaged=1")
        self._edit_target(5)
        iterate(self.config, "kept optimizer", changes=("optimizer",))
        self.assertEqual([row["commit"] for row in pending_retries(self.config)], [deferred])
        self.assertIn(f"retry_ready={deferred}", run_status(self.config))

        self._set_metric("total_ms=75 engaged=1")
        self._edit_target(6)
        outcome = iterate(self.config, "retry prefetch", retry_of=deferred)
        self.assertTrue(outcome.kept)
        self.assertEqual(pending_retries(self.config), [])
        self.assertEqual(json.loads(ledger.read_text(encoding="utf-8").splitlines()[-1])["retry_of"], deferred)

    def test_gate_failure_records_stage_and_evidence(self):
        """A failed gate preserves the candidate receipt and invalidates its metric."""
        record_baseline(self.config)
        self.config.gate_cmd = [sys.executable, "-c", "print('failure=first_mismatch'); raise SystemExit(1)"]
        self.config.aux_patterns = {"failure": r"failure=(?P<value>\w+)"}
        self._edit_target(1)
        iterate(self.config, "gate failure")
        ledger = self.config.run_dir / "candidates.jsonl"
        receipt = json.loads(ledger.read_text(encoding="utf-8"))
        self.assertEqual(receipt["stage"], "gate")
        self.assertEqual(receipt["gate"], "fail")
        self.assertEqual(receipt["aux"], {"failure": "first_mismatch"})
        self.assertIsNone(receipt["metric"])
        self.assertEqual(best_recorded_metric(self.config), 100.0)

    @staticmethod
    def _training_command(phase, steps, *, status="passed", exit_code=0, metric=80, **evidence):
        """Simulate the final output and exit status of a training wrapper."""
        receipt = {
            "workload_id": "fixed-4k", "phase": phase, "completed_steps": steps,
            "status": status, "engaged": True, **evidence,
        }
        output = f"total_ms={metric} engaged=1\nRSI_TRAINING_RECEIPT " + json.dumps(receipt)
        return [sys.executable, "-c", f"print({output!r}); raise SystemExit({exit_code})"]

    def _training_run(self):
        """Create a fresh training series with a full gate and benchmark."""
        gate = self._training_command("gate", 30)
        bench = self._training_command("benchmark", 30)
        run_dir = self._make_run("training_run", {
            "gate_cmd": gate,
            "training_protocol": {"workload_id": "fixed-4k", "gate_steps": 30, "benchmark_steps": 30},
        }, bench)
        return load_run_config(run_dir, self.repo)

    @staticmethod
    def _latest_receipt(config):
        """Read the last committed experiment ledger row."""
        return json.loads((config.run_dir / "candidates.jsonl").read_text(encoding="utf-8").splitlines()[-1])

    def test_training_baseline_self_check_and_workload_freeze(self):
        """Reject a late baseline mismatch and prevent mixing later workloads."""
        config = self._training_run()
        baseline_gate = config.gate_cmd
        config.gate_cmd = self._training_command(
            "gate", 17, status="numerical_mismatch", first_error_step=18, exit_code=1)
        config.benchmark_cmd = ["unused-benchmark"]
        outcome = record_baseline(config)
        receipt = self._latest_receipt(config)
        self.assertEqual(outcome.status, "crash")
        self.assertIsNone(best_recorded_metric(config))
        self.assertEqual(len(receipt["jobs"]), 1)
        self.assertEqual(receipt["jobs"][0]["failure"], "numerical_mismatch")
        self.assertEqual(receipt["jobs"][0]["receipt"]["first_error_step"], 18)

        config.gate_cmd = baseline_gate
        config.benchmark_cmd = self._training_command("benchmark", 30)
        self.assertTrue(record_baseline(config).kept)
        self.assertEqual(pending_retries(config), [])
        self.assertEqual(self._latest_receipt(config)["kind"], "baseline")
        config.training_protocol = None
        self._edit_target(1)
        head = self._git("rev-parse", "HEAD")
        with self.assertRaisesRegex(RuntimeError, "differs from the accepted baseline"):
            iterate(config, "attempt to change comparison contract")
        self.assertEqual(self._git("rev-parse", "HEAD"), head)

    def test_training_receipt_checks_windows_before_keeping(self):
        """A fast partial run or an unsuccessful exit must never become best."""
        config = self._training_run()
        record_baseline(config)
        cases = [
            (29, "passed", 0, "incomplete_window"),
            (30, "futile", 143, "futility_stop_after_window"),
            (30, "cleanup_failed", 1, "cleanup_failed"),
            (30, "passed", 1, "exit_failed_after_window"),
        ]
        for completed, status, exit_code, failure in cases:
            with self.subTest(failure=failure):
                config.benchmark_cmd = self._training_command(
                    "benchmark", completed, status=status, exit_code=exit_code, allocation_retries=12)
                self._edit_target(1)
                outcome = iterate(config, failure)
                job = self._latest_receipt(config)["jobs"][-1]
                self.assertEqual(outcome.status, "crash")
                self.assertIsNone(outcome.metric)
                self.assertEqual(job["failure"], failure)
                self.assertEqual(job["receipt"]["completed_steps"], completed)
                self.assertEqual(job["receipt"]["allocation_retries"], 12)
                self.assertEqual(best_recorded_metric(config), 80.0)
                self.assertEqual((self.repo / "target.py").read_text(encoding="utf-8"), "STATE = 0\n")

        config.benchmark_cmd = self._training_command("benchmark", 30, metric=70, allocation_retries=12)
        self._edit_target(2)
        self.assertTrue(iterate(config, "complete measurement with allocator evidence").kept)
        self.assertEqual(best_recorded_metric(config), 70.0)
        self.assertEqual(self._latest_receipt(config)["jobs"][-1]["receipt"]["allocation_retries"], 12)

    def test_training_noise_confirmation_keeps_failed_attempt_evidence(self):
        """An invalid rerun voids the median and retains both job receipts."""
        config = self._training_run()
        record_baseline(config)
        initial = self._training_command("benchmark", 30, allocation_retries=12)[-1]
        failed = self._training_command("benchmark", 30, workload_id="different-4k")[-1]
        stamp = config.run_dir / "rerun.stamp"
        code = (
            "from pathlib import Path\n"
            f"stamp = Path({str(stamp)!r})\n"
            f"command = {failed!r} if stamp.exists() else {initial!r}\n"
            "stamp.touch()\nexec(command)\n"
        )
        config.benchmark_cmd = [sys.executable, "-c", code]
        config.noise_confirm_reruns = 1
        self._edit_target(1)
        self.assertEqual(iterate(config, "confirm a noisy sample").status, "crash")
        receipt = self._latest_receipt(config)
        self.assertEqual(receipt["stage"], "noise_confirmation")
        self.assertEqual([job["phase"] for job in receipt["jobs"]], ["gate", "benchmark", "benchmark"])
        self.assertEqual(receipt["jobs"][1]["metric"], 80.0)
        self.assertEqual(receipt["jobs"][1]["receipt"]["allocation_retries"], 12)
        self.assertEqual(receipt["jobs"][2]["failure"], "invalid_training_receipt")
        self.assertEqual(best_recorded_metric(config), 80.0)

    def test_dirty_non_target_refuses_to_run(self):
        """Edits outside the declared targets abort before any benchmark."""
        record_baseline(self.config)
        (self.repo / "other.py").write_text("x = 1\n", encoding="utf-8")
        self._git("add", "other.py")
        self._edit_target(4)
        with self.assertRaises(RuntimeError):
            iterate(self.config, "dirty tree")

    def _make_run(self, name: str, extra: dict, bench: list[str]) -> Path:
        """Create and commit an additional run directory."""
        run_dir = self.repo / name
        run_dir.mkdir()
        (run_dir / "run.json").write_text(json.dumps({
            "target_files": ["target.py"],
            "benchmark_cmd": bench,
            "bench_must_match": ["engaged=1"],
            **extra,
        }), encoding="utf-8")
        self._git("add", "-A")
        self._git("commit", "-q", "-m", f"add {name}")
        return run_dir

    def test_within_noise_rerun_decides_on_the_median(self):
        """A within-noise first sample is confirmed by reruns, not trusted."""
        (self.repo / "bench_seq.py").write_text(_SEQ_SCRIPT, encoding="utf-8")
        run_dir = self._make_run("run_seq", {
            "noise_fraction": 0.05, "noise_confirm_reruns": 2,
        }, [sys.executable, "bench_seq.py", "run_seq/queue.txt"])
        # The queue lives inside the run dir, where benchmark-made dirt is legal.
        (run_dir / "queue.txt").write_text(
            "total_ms=100 engaged=1\ntotal_ms=99 engaged=1\n"
            "total_ms=90 engaged=1\ntotal_ms=91 engaged=1\n", encoding="utf-8")
        self._git("add", "-A")
        self._git("commit", "-q", "-m", "seed queue")
        config = load_run_config(run_dir, self.repo)
        record_baseline(config)
        self.assertEqual(best_recorded_metric(config), 100.0)
        self._edit_target(7)
        outcome = iterate(config, "confirmed win")
        self.assertEqual(outcome.status, "keep")
        self.assertEqual(outcome.metric, 91.0)
        self.assertIn("median of 3 runs", outcome.detail)
        self.assertEqual(best_recorded_metric(config), 91.0)

    def test_aux_patterns_land_in_the_description(self):
        """Captured auxiliary metrics ride along in the TSV description."""
        run_dir = self._make_run("run_aux", {
            "aux_patterns": {"wall_s": r"wall_s=(?P<value>[0-9.]+)"},
        }, [sys.executable, "bench.py"])
        config = load_run_config(run_dir, self.repo)
        self._set_metric("total_ms=100 wall_s=3.5 engaged=1")
        record_baseline(config)
        last_row = config.results_path.read_text(encoding="utf-8").splitlines()[-1]
        self.assertIn("baseline [wall_s=3.5]", last_row)


if __name__ == "__main__":
    unittest.main()
