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
"""Verl stage-order contracts on actual colocated orchestration and CPU samples."""
import os
import unittest
from unittest.mock import Mock, patch

from tests.ut.rl.colocate_async.support import trainer_fixture


class TestColocateAsyncTrainer(unittest.TestCase):
    """Reject transitions that can race generation, training or publication."""

    def setUp(self) -> None:
        """Install CPU fixtures and register cleanup for each isolated case."""
        self.coordinator = patch("rl.trainer.colocate_async.coordinator_call",
                                 side_effect=lambda _, callback: callback())
        self.coordinator.start()
        self.addCleanup(self.coordinator.stop)

    def test_real_loop_sleeps_before_device_batch_and_commits_before_publication(self):
        """Verify real loop sleeps before device batch and commits before publication."""
        trainer, events, _ = trainer_fixture()
        trainer.train()
        expected = ["sample_return", "partial_pause_done", "sleep_done", "device_batch_built",
                    "update_start", "update_end", "buffer_commit", "publication_end", "partial_resume_done"]
        positions = [events.index(event) for event in expected]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(trainer.state.global_step, 1)
        self.assertEqual(trainer.state.consumed_samples, 4)
        self.assertEqual(trainer.state.consumed_tokens, 4)
        batch = trainer.actor.update.call_args.args[0]
        self.assertTrue(batch.mixed_policy_versions)
        self.assertEqual(len(batch.trajectories), 2)
        trainer.publication.publish.assert_called_once_with(2)
        trainer.rollouter.close.assert_called_once_with(graceful=True)

    def test_failures_never_cross_the_next_unsafe_stage(self):
        """Verify failures never cross the next unsafe stage."""
        for stage in ("pause", "sleep", "actor", "publish", "resume"):
            trainer, _, _ = trainer_fixture()
            trainer.state.max_steps = 2
            target = {"pause": trainer.rollouter.pause,
                      "sleep": trainer.rollout_engine.prepare_for_training,
                      "actor": trainer.actor.update, "publish": trainer.publication.publish,
                      "resume": trainer.rollouter.resume}[stage]
            target.side_effect = RuntimeError(stage)
            with self.subTest(stage=stage), self.assertRaisesRegex(RuntimeError, stage):
                trainer.train()
            self.assertEqual(trainer.rollouter.sample_batch.call_count, 1)
            trainer._report_update.assert_not_called()
            trainer.checkpoints.save.assert_not_called()
            trainer.rollouter.drain_and_snapshot.assert_not_called()
            if stage == "resume":
                trainer.rollouter.resume.assert_called_once_with(2)
            else:
                trainer.rollouter.resume.assert_not_called()
            if stage in ("pause", "sleep", "actor"):
                trainer.rollouter.commit_batch.assert_not_called()
                trainer.publication.publish.assert_not_called()
                self.assertEqual(trainer.state.global_step, 0)
                self.assertEqual(trainer.optimizer_steps, 0)
                self.assertEqual(trainer.state.consumed_samples, 0)
                self.assertEqual(trainer.state.consumed_tokens, 0)
            else:
                trainer.rollouter.commit_batch.assert_called_once_with()
                self.assertEqual(trainer.state.global_step, 1)
            if stage in ("pause", "sleep"):
                trainer.actor.update.assert_not_called()
            if stage == "pause":
                trainer.rollout_engine.prepare_for_training.assert_not_called()
            trainer.rollouter.close.assert_called_once_with(graceful=False)
            trainer.close.assert_called_once()

    def test_non_root_never_dereferences_root_only_rollouter(self):
        trainer, _, _ = trainer_fixture()
        trainer.rank, trainer.rollouter = 1, None
        with patch("rl.trainer.colocate_async.coordinator_call", return_value=None):
            trainer._pause_rollout_for_training()
            trainer._publish_policy_and_resume()
        trainer.rollout_engine.prepare_for_training.assert_called_once_with()
        trainer.publication.publish.assert_called_once_with(1)

    def test_initial_publication_precedes_producer_and_scopes_server_environment(self):
        """Verify initial publication precedes producer and scopes server environment."""
        trainer, events, _ = trainer_fixture()
        del trainer._restore_and_start
        trainer.checkpoints.begin.return_value = None
        previous = os.environ.get("HYPER_RL_ASYNC_TRAINING")
        trainer.rollout_engine.start_service.side_effect = lambda: self.assertEqual(
            os.environ.get("HYPER_RL_ASYNC_TRAINING"), "1")
        with patch("rl.trainer.colocate_async.AsyncCheckpoint.resolve_path", return_value=None), patch(
                "rl.trainer.colocate_async.AsyncGenerationClient.from_config") as client, patch(
                "rl.trainer.colocate_async.ColocateAsyncRollouter") as producer:
            producer.side_effect = lambda *_: (events.append("producer_started") or Mock())
            trainer._restore_and_start()
        client.return_value.pause.assert_called_once_with()
        client.return_value.close.assert_called_once_with()
        self.assertLess(events.index("publication_end"), events.index("producer_started"))
        self.assertEqual(os.environ.get("HYPER_RL_ASYNC_TRAINING"), previous)

    def test_model_snapshot_step_mismatch_prevents_service_start(self):
        """Verify model snapshot step mismatch prevents service start."""
        trainer, _, _ = trainer_fixture()
        del trainer._restore_and_start
        restored = {"execution_mode": "colocate_async", "learner_update": 3,
                    "source_update": 3, "optimizer_steps": 3}
        with patch("rl.trainer.colocate_async.AsyncCheckpoint.resolve_path",
                   return_value="step_3"), patch(
                "rl.trainer.colocate_async.AsyncCheckpoint.load", return_value=restored):
            with self.assertRaisesRegex(ValueError, "identities"):
                trainer._restore_and_start()
        trainer.rollout_engine.start_service.assert_not_called()

    def test_cleanup_preserves_the_primary_error(self):
        trainer, _, _ = trainer_fixture()
        trainer.rollouter.pause.side_effect = ValueError("primary")
        trainer.rollouter.close.side_effect = RuntimeError("cleanup")
        trainer.close.side_effect = OSError("role cleanup")
        with self.assertRaisesRegex(ValueError, "primary"):
            trainer.train()
        trainer.close.assert_called_once()

    def test_cleanup_attempts_both_resources_and_raises_first_failure(self):
        trainer, _, _ = trainer_fixture()
        failure = ValueError("sampler cleanup")
        trainer.rollouter.close.side_effect = failure
        trainer.close.side_effect = OSError("role cleanup")
        with self.assertRaises(ValueError) as caught:
            trainer._close(completed=True, failed=False)
        self.assertIs(caught.exception, failure)
        trainer.rollouter.close.assert_called_once_with(graceful=True)
        trainer.close.assert_called_once_with()

    def test_non_root_cleanup_reports_role_failure_without_sampler(self):
        trainer, _, _ = trainer_fixture()
        trainer.rank, trainer.rollouter = 1, None
        failure = OSError("role cleanup")
        trainer.close.side_effect = failure
        with self.assertRaises(OSError) as caught:
            trainer._close(completed=True, failed=False)
        self.assertIs(caught.exception, failure)
        trainer.close.assert_called_once_with()

    def test_update_result_carries_its_own_diagnostics(self):
        trainer, _, payloads = trainer_fixture()
        result = trainer._update_actor(payloads)
        self.assertEqual(result.rejected_groups, ())
        self.assertEqual(result.diagnostics, {})
        self.assertIsNotNone(result.update)
        self.assertFalse(hasattr(trainer, "_correction_metrics"))

    def test_gateway_ownership_depends_on_evaluation_and_runner(self):
        """Only active Codex evaluation releases the producer gateway."""
        for runner, enabled, expected in (("codex", False, False), ("internal", True, False), ("codex", True, True)):
            with self.subTest(runner=runner, enabled=enabled):
                trainer, _, _ = trainer_fixture()
                trainer.config["agentic"] = {"runner": runner}
                trainer.evaluator = Mock() if enabled else None
                trainer._evaluate_policy = Mock()
                with patch("rl.trainer.colocate_async.coordinator_call", side_effect=lambda _, fn: fn()):
                    self.assertEqual(trainer._evaluate_checkpoint(), expected)
                self.assertEqual(trainer.rollouter.close.call_count, int(expected))
                self.assertEqual(trainer._evaluate_policy.call_count, int(enabled))

    def test_rejected_result_releases_device_batch_before_rollout_resumes(self):
        trainer, _, payloads = trainer_fixture()
        trainer._correct_experience = lambda batch: (batch, {"off_policy/rejected_groups": 1}, ("0",))
        result = trainer._update_actor(payloads)
        self.assertEqual(result.rejected_groups, ("0",))
        self.assertIsNone(result.rollout)
        self.assertIsNone(result.update)
        self.assertEqual(result.diagnostics, {"off_policy/rejected_groups": 1})
        trainer.actor.update.assert_not_called()
        trainer.rollouter.commit_batch.assert_not_called()
