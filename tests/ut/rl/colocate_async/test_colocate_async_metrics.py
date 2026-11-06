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
"""Batch diagnostics and reporting retain the previous observable metrics contract."""
import unittest
from unittest.mock import patch

from rl.trainer.colocate_async import ColocateUpdateResult
from rl.utils.monitoring.colocate_async_metrics import report_colocate_update, summarize_colocate_payloads
from tests.ut.rl.colocate_async.support import trainer_fixture


class TestColocateAsyncMetrics(unittest.TestCase):
    """Exercise real metadata summaries and prevent logging past a failed learning gate."""

    def test_payload_statistics_preserve_behavior_age_and_versions(self):
        trainer, _, payloads = trainer_fixture()
        metrics = summarize_colocate_payloads(payloads, trainer.candidates, trainer.settings.max_group_bytes, 3)
        self.assertEqual(metrics, {"colocate/sample_age_max": 2, "colocate/behavior_versions": 1})
        self.assertEqual(trainer.state.global_step, 0)
        trainer.rollouter.commit_batch.assert_not_called()

    def test_reporting_keeps_diagnostics_and_enforces_gate_before_log_on_non_root(self):
        """Verify reporting keeps diagnostics and enforces gate before log on non root."""
        trainer, _, payloads = trainer_fixture()
        trainer.rank = 1
        trainer.config["train"]["learning_gate"] = {"enabled": True, "min_gradient_norm": 0.25}
        trainer.rollout_engine.policy_version = 1
        rollout = trainer._build_local_batch(payloads)
        update = trainer.actor.update.return_value
        diagnostics = {"off_policy/is_ess_fraction": .75, "timing_s/sample_wait": .2}
        result = ColocateUpdateResult(rollout, update, diagnostics)
        events = []
        prefix = "rl.utils.monitoring.colocate_async_metrics."
        with patch(prefix + "summarize_rollout", side_effect=lambda *_, **__: (
                events.append("rollout_metrics") or ({"reward/mean": .5}, []))), patch(
                "rl.utils.monitoring.metrics.dist.get_world_size", return_value=2), patch(
                prefix + "enforce_learning_gate", side_effect=RuntimeError("learning gate failed")) as gate:
            with self.assertRaisesRegex(RuntimeError, "learning gate failed"):
                report_colocate_update(trainer, result.rollout, result.update,
                                       lambda: events.append("sampling_metrics") or result.diagnostics)
        trainer.tracker.log.assert_not_called()
        self.assertEqual(gate.call_args.args[0]["off_policy/is_ess_fraction"], .75)
        self.assertEqual(gate.call_args.kwargs["config"], trainer.config["train"]["learning_gate"])
        self.assertEqual(events, ["rollout_metrics", "sampling_metrics"])
        self.assertEqual(diagnostics, {"off_policy/is_ess_fraction": .75, "timing_s/sample_wait": .2})
