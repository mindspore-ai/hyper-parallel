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
"""Exercise the offline accumulation planner from completed trial receipts."""

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


SCRIPT = Path(__file__).resolve().parents[3] / ".agent/skills/mfu-riser/scripts/plan_accumulation.py"


class TestAccumulationPlanning(unittest.TestCase):
    """Check arithmetic and reject incomparable or incomplete evidence."""

    def setUp(self):
        temporary_dir = TemporaryDirectory()
        self.addCleanup(temporary_dir.cleanup)
        self.directory = Path(temporary_dir.name)

    def _trial(self, count, seconds, *, seq_len=4096, token_factor=1, complete=True):
        trial = self.directory / f"trial-{count}"
        trial.mkdir()
        scope = {"window": [3, 4], "sequence_length": seq_len, "micro_batch_size": 1,
                 "global_batch_size": 8 * count, "gradient_accumulation_steps": count}
        source = "matched-source"
        context = {"config": {"workload": scope}, "source_digest": source}
        tokens = 32768 * count * token_factor
        samples = [{"step": step, "time_s": seconds, "model_tokens": tokens, "model_flops": 1e15 * count}
                   for step in ([3, 4] if complete else [3])]
        result = {"cleanup": "PASS", "decision": "baseline", "id": trial.name, "source_digest": source,
                  "validation_level": "screening",
                  "metrics": {"samples": samples, "mean_step_seconds": seconds, "mfu_percent": 75 * count / seconds,
                              "tokens_per_second_per_device": tokens / (8 * seconds)}}
        (trial / "context.json").write_text(json.dumps(context), encoding="utf-8")
        (trial / "result.json").write_text(json.dumps(result), encoding="utf-8")
        return trial

    def _run(self, *trials):
        return subprocess.run(
            [sys.executable, str(SCRIPT), *(str(path) for path in trials),
             "--probe", "6", "--target-mfu-percent", "35", "--max-relative-time-error", "0.01"],
            text=True, capture_output=True, timeout=20, check=False,
        )

    def test_matched_holdout_predicts_larger_probe(self):
        """Use two fit points and an independent replay to size the next probe."""
        output = self._run(self._trial(1, 3), self._trial(2, 5), self._trial(4, 9.06))
        self.assertEqual(output.returncode, 0, output.stderr)
        report = json.loads(output.stdout)
        self.assertEqual((report["fixed_seconds"], report["seconds_per_microbatch"]), (1, 2))
        self.assertTrue(report["independent_checks"][0]["within_tolerance"])
        estimate = report["probe_estimates"][0]
        self.assertEqual(estimate["predicted_mean_step_seconds"], 13)
        self.assertAlmostEqual(estimate["estimated_mfu_percent"], 450 / 13)
        self.assertAlmostEqual(estimate["target_step_budget_seconds"], 450 / 35)
        self.assertAlmostEqual(estimate["required_step_savings_seconds"], 13 - 450 / 35)

    def test_failed_holdout_suppresses_probe_predictions(self):
        """Preserve missed predictions for diagnosis and stop extrapolating."""
        output = self._run(self._trial(1, 3), self._trial(2, 5), self._trial(4, 10))
        self.assertEqual(output.returncode, 0, output.stderr)
        report = json.loads(output.stdout)
        self.assertFalse(report["independent_checks"][0]["within_tolerance"])
        self.assertEqual(report["probe_estimates"], [])

    def test_mismatched_shape_or_token_work_is_rejected(self):
        """Changing length or token work cannot masquerade as amortization."""
        one = self._trial(1, 3)
        two = self._trial(2, 5)
        cases = (({"seq_len": 8192}, "measurement scope"), ({"token_factor": 2}, "Effective tokens per microbatch"))
        for index, (override, reason) in enumerate(cases, start=3):
            with self.subTest(reason=reason):
                output = self._run(one, two, self._trial(index, 2 * index + 1, **override))
                self.assertEqual(output.returncode, 1)
                self.assertIn(reason, output.stderr)

    def test_incomplete_window_is_rejected(self):
        """Do not size a probe using a truncated trial."""
        output = self._run(self._trial(1, 3), self._trial(2, 5, complete=False))
        self.assertEqual(output.returncode, 1)
        self.assertIn("Incomplete timing window", output.stderr)
