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
"""CPU contracts for policy binding and cleanup across external Agent rollout managers."""

import unittest
from unittest.mock import MagicMock, patch, sentinel

from rl.roles.rollout import worker


class TestProgramRolloutManager(unittest.TestCase):
    """Exercise real manager orchestration without starting a CLI, SDK, or model service."""

    def _manager(self, manager_type: type) -> tuple:
        """Build a program manager with a mocked runner and a versioned engine."""
        runtime = MagicMock()
        runtime.engine.generation_policy_version.return_value = 3
        with patch.object(worker, "ProgramAgentRunner") as runner:
            manager = manager_type(
                runtime=runtime, num_return_sequences=4, max_new_tokens=256,
                temperature=0.7, top_p=0.9, top_k=5, pad_token_id=0, eos_token_id=2,
                do_sample=False, seed=17,
            )
        manager.agent_runner.rollout.return_value = sentinel.batch
        return manager, runtime, runner

    def test_generation_preserves_version_binding_and_returns_batch(self) -> None:
        """Both external harnesses bind the served version only for the duration of rollout."""
        for manager_type in (worker.CodexRolloutManager, worker.DeepSeekRolloutManager):
            with self.subTest(manager=manager_type):
                manager, runtime, runner = self._manager(manager_type)
                events = MagicMock()
                events.attach_mock(runtime.ensure_started, "start")
                events.attach_mock(runtime.bind_episode_version, "bind")
                events.attach_mock(manager.agent_runner.rollout, "rollout")
                events.attach_mock(runtime.clear_episode_version, "clear")
                result = manager.generate([sentinel.prompt], policy_version=3)
                self.assertIs(result, sentinel.batch)
                self.assertEqual([call[0] for call in events.mock_calls], ["start", "bind", "rollout", "clear"])
                runtime.bind_episode_version.assert_called_once_with(3)
                manager.agent_runner.rollout.assert_called_once_with([sentinel.prompt], 3)
                settings = runner.call_args.kwargs["settings"]
                self.assertEqual(settings.seed, 17)
                self.assertEqual(settings.eos_token_ids, (2,))
                self.assertTrue(settings.collect_log_probs)
                self.assertFalse(settings.do_sample)
                manager.close()
                runtime.close.assert_called_once_with()

    def test_stale_request_fails_before_binding_or_sampling(self) -> None:
        """A request for old policy weights cannot be silently served by a newer engine."""
        for manager_type in (worker.CodexRolloutManager, worker.DeepSeekRolloutManager):
            with self.subTest(manager=manager_type):
                manager, runtime, _ = self._manager(manager_type)
                with self.assertRaisesRegex(RuntimeError, "requested=2, served=3"):
                    manager.generate([sentinel.prompt], policy_version=2)
                runtime.bind_episode_version.assert_not_called()
                manager.agent_runner.rollout.assert_not_called()
                runtime.clear_episode_version.assert_not_called()

    def test_rollout_failure_always_clears_episode_binding(self) -> None:
        """A failed tool execution must release the episode version before propagating its error."""
        for manager_type in (worker.CodexRolloutManager, worker.DeepSeekRolloutManager):
            with self.subTest(manager=manager_type):
                manager, runtime, _ = self._manager(manager_type)
                manager.agent_runner.rollout.side_effect = RuntimeError("tool failed")
                with self.assertRaisesRegex(RuntimeError, "tool failed"):
                    manager.generate([sentinel.prompt], policy_version=3)
                runtime.clear_episode_version.assert_called_once_with()
                runtime.engine.generation_policy_version.assert_called_once_with()

    def test_concurrent_policy_change_rejects_the_completed_batch(self) -> None:
        """Version drift during an Agent episode invalidates the returned training batch."""
        for manager_type in (worker.CodexRolloutManager, worker.DeepSeekRolloutManager):
            with self.subTest(manager=manager_type):
                manager, runtime, _ = self._manager(manager_type)
                runtime.engine.generation_policy_version.side_effect = [3, 4]
                with self.assertRaisesRegex(RuntimeError, "before=3, after=4"):
                    manager.generate([sentinel.prompt], policy_version=3)
                runtime.clear_episode_version.assert_called_once_with()
