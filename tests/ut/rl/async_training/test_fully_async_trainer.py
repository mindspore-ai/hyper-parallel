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
"""Publication failure propagation and synchronous runtime isolation."""
from concurrent.futures import Future
import os
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from rl.async_training.fully_async_rollouter import ExperienceBuffer, FullyAsyncRollouter
from rl.async_training.fully_async_trainer import FullyAsyncTrainer
from rl.async_training.protocol import AsyncCheckpoint, AsyncTrainingConfig, SamplingTask
from tests.ut.rl.async_training.support import group


class TestAsyncDecisions(unittest.TestCase):
    """Avoid a stale-generation loop when the trainer outruns its published policy."""

    def test_selected_checkpoint_reaches_both_restore_owners(self):
        """Test selected checkpoint reaches both restore owners."""
        trainer = FullyAsyncTrainer.__new__(FullyAsyncTrainer)
        trainer.config = {"train": {"checkpoint": {}}}
        trainer.roles = Mock()
        trainer.roles.checkpoints.config = {}
        trainer.roles.checkpoints.validate_resume.side_effect = ValueError("stop before model loading")
        trainer.rank, trainer.sampler, trainer.trace = 0, None, Mock()
        with patch("rl.async_training.fully_async_trainer.coordinator_call", side_effect=lambda _, call: call()):
            with patch.object(AsyncCheckpoint, "resolve_path", return_value="checkpoints/step_10"):
                with patch.object(AsyncCheckpoint, "load", return_value=None) as load:
                    with self.assertRaisesRegex(ValueError, "stop before model loading"):
                        trainer.train()
        self.assertEqual(trainer.roles.checkpoints.config["load_path"], "checkpoints/step_10")
        load.assert_called_once_with("checkpoints/step_10", trainer.config)
        trainer.roles.checkpoints.begin.assert_not_called()

    def test_cleanup_preserves_primary_failure_and_attempts_every_resource(self):
        """Test cleanup preserves primary failure and attempts every resource."""
        trainer = FullyAsyncTrainer.__new__(FullyAsyncTrainer)
        trainer.config, trainer.rank = {}, 0
        trainer.roles, trainer.sampler, trainer.trace = Mock(), Mock(), Mock()
        trainer.roles.state.global_step = 0
        trainer.sampler.close.side_effect = RuntimeError("sampler cleanup failed")
        trainer.roles.close.side_effect = RuntimeError("role cleanup failed")
        trainer.trace.emit.side_effect = OSError("audit write failed")
        with patch("rl.async_training.fully_async_trainer.coordinator_call", side_effect=lambda _, call: call()):
            with patch.object(AsyncCheckpoint, "resolve_path", side_effect=ValueError("original failure")):
                with self.assertRaisesRegex(ValueError, "original failure"):
                    trainer.train()
        trainer.sampler.close.assert_called_once_with(graceful=False)
        trainer.roles.close.assert_called_once()
        trainer.trace.emit.assert_called_once_with("run_end", completed=False, update=0)

    def test_exhausted_freshness_budget_requests_publish_before_new_sampling(self):
        """Test exhausted freshness budget requests publish before new sampling."""
        trainer = FullyAsyncTrainer.__new__(FullyAsyncTrainer)
        trainer.settings = AsyncTrainingConfig()
        trainer.buffer = ExperienceBuffer(trainer.settings, 2)
        trainer.roles = SimpleNamespace(state=SimpleNamespace(global_step=5),
                                        parallel_dims=SimpleNamespace(dp_size=1))
        trainer.sampler = Mock()
        trainer.source_update = 0
        trainer.version = 1
        self.assertEqual(trainer._root_command(), ("publish", None))
        trainer.sampler.fill.assert_not_called()

    def test_fresh_synchronous_import_does_not_load_async_execution_or_reporting(self):
        """Shared config/data helpers must not activate the asynchronous runtime for SyncTrainer."""
        script = (
            "import rl.trainer, sys; "
            "assert not any(name in sys.modules for name in ("
            "'rl.async_training', 'rl.roles.rollout.collection', 'rl.utils.monitoring.async_metrics'))"
        )
        result = subprocess.run([sys.executable, "-c", script],
                                env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
                                capture_output=True, text=True, timeout=60, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


class TestPartialPublication(unittest.TestCase):
    """Assert safe transaction ordering and complete-group checkpoint boundaries."""

    @staticmethod
    def _trainer():
        trainer = FullyAsyncTrainer.__new__(FullyAsyncTrainer)
        trainer.settings = AsyncTrainingConfig()
        trainer.roles = SimpleNamespace(state=SimpleNamespace(global_step=2),
                                        publish_policy=Mock(), tracker=Mock())
        trainer.sampler, trainer.trace = Mock(), Mock()
        return trainer

    def test_periodic_publication_pauses_and_checkpoint_publication_drains(self):
        """Test periodic publication pauses and checkpoint publication drains."""
        for resume in (True, False):
            trainer = self._trainer()
            order = Mock()
            order.attach_mock(trainer.sampler.pause, "pause")
            order.attach_mock(trainer.sampler.drain, "drain")
            order.attach_mock(trainer.roles.publish_policy, "publish")
            order.attach_mock(trainer.sampler.resume, "resume")
            with self.subTest(resume=resume), patch(
                "rl.async_training.fully_async_trainer.coordinator_call", side_effect=lambda name, call: call(),
            ):
                trainer._publish(resume=resume)
            self.assertEqual([call[0] for call in order.mock_calls],
                             ["pause", "publish", "resume"] if resume else ["drain", "publish"])
            trainer.roles.publish_policy.assert_called_once_with(3)

    def test_failed_publication_never_releases_continuations(self):
        trainer = self._trainer()
        trainer.roles.publish_policy.side_effect = RuntimeError("transfer failed")
        with patch("rl.async_training.fully_async_trainer.coordinator_call", side_effect=lambda name, call: call()):
            with self.assertRaisesRegex(RuntimeError, "transfer failed"):
                trainer._publish()
        trainer.sampler.resume.assert_not_called()

    def test_non_coordinator_never_dereferences_the_coordinator_only_sampler(self):
        trainer = self._trainer()
        trainer.sampler = None
        with patch("rl.async_training.fully_async_trainer.coordinator_call", return_value=None):
            trainer._publish()
        trainer.roles.publish_policy.assert_called_once_with(3)


class TestCandidateFailures(unittest.TestCase):
    """Infrastructure rejection never becomes a zero-reward training candidate."""

    def test_group_discard_waits_for_all_candidates_and_releases_once(self):
        """Test group discard waits for all candidates and releases once."""
        task = SamplingTask(0, 1, 0)
        failed, completed = Future(), Future()
        failed.set_exception(RuntimeError("execution transport failed"))
        producer = FullyAsyncRollouter.__new__(FullyAsyncRollouter)
        producer.candidates = 2
        producer.groups = {0: (task, 0., [failed, completed])}
        producer.trace, producer.connection, producer.results = Mock(), Mock(), Mock()
        producer._accept_group(0)
        producer.connection.send.assert_not_called()
        completed.set_result((group(task).trajectories[1],))
        producer._accept_group(0)
        producer.connection.send.assert_called_once_with(("discarded", task))
        producer.results.put.assert_not_called()
        self.assertFalse(producer.groups)

    def test_group_contract_failure_rejects_without_a_discard_or_training_payload(self):
        future = Future()
        future.set_exception(ValueError("invalid behavior versions"))
        producer = FullyAsyncRollouter.__new__(FullyAsyncRollouter)
        producer.groups = {0: (SamplingTask(0, 1, 0), 0., [future])}
        producer.connection, producer.results = Mock(), Mock()
        with self.assertRaisesRegex(ValueError, "behavior versions"):
            producer._accept_group(0)
        producer.connection.send.assert_not_called()
        producer.results.put.assert_not_called()

    def test_runtime_failure_retries_without_changing_candidate_identity(self):
        """Test runtime failure retries without changing candidate identity."""
        task = SamplingTask(0, 1, 0)
        rows = group(task).trajectories[:1]
        producer = FullyAsyncRollouter.__new__(FullyAsyncRollouter)
        producer.settings = AsyncTrainingConfig(max_candidate_retries=1)
        producer.client = SimpleNamespace(closed=False)
        producer.task_local = SimpleNamespace(tokenizer=object())
        producer.trace, producer.collector = Mock(), Mock()
        producer.collector.generate.side_effect = [RuntimeError("temporary service failure"), rows]
        prompt = object()
        self.assertEqual(producer._generate_candidate(task, prompt, 0), rows)
        self.assertEqual(producer.collector.generate.call_count, 2)
        self.assertEqual(producer.collector.generate.call_args_list[0], producer.collector.generate.call_args_list[1])

    def test_contract_failure_is_not_retried_as_infrastructure(self):
        producer = FullyAsyncRollouter.__new__(FullyAsyncRollouter)
        producer.settings = AsyncTrainingConfig(max_candidate_retries=1)
        producer.client = SimpleNamespace(closed=False)
        producer.task_local = SimpleNamespace(tokenizer=object())
        producer.trace, producer.collector = Mock(), Mock()
        producer.collector.generate.side_effect = ValueError("invalid trajectory contract")
        with self.assertRaisesRegex(ValueError, "contract"):
            producer._generate_candidate(SamplingTask(0, 1, 0), object(), 0)
        self.assertEqual(producer.collector.generate.call_count, 1)
