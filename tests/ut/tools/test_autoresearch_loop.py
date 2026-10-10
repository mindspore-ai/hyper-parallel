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
# Ahead of everything else on the path so the engine's own sibling imports
# (loop.py imports config) resolve here and not to an unrelated module.
sys.path.insert(0, str(_ENGINE))

from config import load_run_config  # noqa: E402  # pylint: disable=wrong-import-position
from loop import (  # noqa: E402  # pylint: disable=wrong-import-position
    best_recorded_metric,
    iterate,
    record_baseline,
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
