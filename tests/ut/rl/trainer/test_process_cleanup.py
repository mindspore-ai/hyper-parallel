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
"""CPU regression tests for RL service shutdown and process-group cache cleanup."""
# White-box regression tests intentionally exercise internal state and lifecycle hooks.
# pylint: disable=protected-access

import subprocess
import unittest
from contextlib import ExitStack
from unittest.mock import MagicMock, call, patch, sentinel

from rl import process_cleanup as runtime
from rl.trainer import SyncTrainer

from hyper_parallel.core.dtensor import device_mesh
from hyper_parallel.core.pipeline_parallel import _p2p


class TestDistributedTeardown(unittest.TestCase):
    """Exercise actual cache identities while mocking only backend process groups."""

    def setUp(self) -> None:
        """Isolate populated runtime caches without creating a hardware process group."""
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.caches = (
            device_mesh.EXISTING_COMM_GROUPS,
            runtime._P2P_MULTI_STREAM_GROUPS,
            device_mesh._DEVICE_MESH_MAP,
            runtime._LAYOUT_CACHE,
            runtime._HYBRID_MESH_CACHE,
            runtime._tensor_redistribution._transform_cache,
        )
        self.caches += tuple(cache for cache in (
            getattr(runtime.cc, "_EXISTING_COMM_GROUPS", None),
            getattr(runtime.hsdp_param, "_GROUP_INFO_CACHE", None),
        ) if cache is not None)
        for cache in self.caches:
            self.stack.enter_context(patch.dict(cache, {"stale-group": sentinel.group}, clear=True))
        self.stack.enter_context(patch.object(runtime._tensor_redistribution, "is_init", True))
        self.stack.enter_context(patch.object(runtime._tensor_redistribution, "rank_id", 3))
        self.group = self.stack.enter_context(
            patch.object(runtime.dist, "is_initialized", return_value=True)
        )
        self.destroy = self.stack.enter_context(patch.object(runtime.dist, "destroy_process_group"))

    def _assert_caches_cleared(self) -> None:
        for cache in self.caches:
            self.assertEqual(cache, {}, f"Expected no references to destroyed groups, got={cache}")
        self.assertFalse(runtime._tensor_redistribution.is_init)
        self.assertIsNone(runtime._tensor_redistribution.rank_id)

    def test_teardown_uses_native_p2p_cache_and_clears_all_group_references(self) -> None:
        """A complete teardown must clear the relocated P2P cache as well as DTensor caches."""
        self.assertIs(runtime._P2P_MULTI_STREAM_GROUPS, _p2p._P2P_MULTI_STREAM_GROUPS)
        runtime.destroy_process_group()
        self.destroy.assert_called_once_with()
        self._assert_caches_cleared()

    def test_repeated_teardown_does_not_destroy_an_absent_group(self) -> None:
        """A second cleanup after the default group is gone remains safe."""
        self.group.side_effect = [True, False]
        runtime.destroy_process_group()
        runtime.destroy_process_group()
        self.destroy.assert_called_once_with()
        self._assert_caches_cleared()

    def test_uninitialized_runtime_still_drops_stale_caches(self) -> None:
        """Initialization failures may leave caches even when no default group exists."""
        self.group.return_value = False
        runtime.destroy_process_group()
        self.destroy.assert_not_called()
        self._assert_caches_cleared()

    def test_backend_error_propagates_after_cache_cleanup(self) -> None:
        """Backend destruction errors must stay observable without retaining stale references."""
        self.destroy.side_effect = RuntimeError("backend teardown failed")
        with self.assertRaisesRegex(RuntimeError, "backend teardown failed"):
            runtime.destroy_process_group()
        self._assert_caches_cleared()

    def test_trainer_cleanup_calls_the_rl_teardown_without_shared_patches(self) -> None:
        """The real Trainer cleanup must reach the RL helper and close every owned service."""
        trainer = object.__new__(SyncTrainer)
        tracker = MagicMock()
        trainer._tracker = tracker
        trainer.rollout_manager = MagicMock()
        trainer.rollout_engine = MagicMock()
        trainer._runtime_started = True
        order = MagicMock()
        order.attach_mock(tracker.finish, "tracker")
        order.attach_mock(trainer.rollout_manager.close, "manager")
        order.attach_mock(trainer.rollout_engine.close, "engine")
        order.attach_mock(self.destroy, "distributed")
        trainer._cleanup()
        self.assertEqual(order.mock_calls, [call.tracker(), call.manager(), call.engine(), call.distributed()])
        tracker.finish.assert_called_once_with()
        trainer.rollout_manager.close.assert_called_once_with()
        trainer.rollout_engine.close.assert_called_once_with()
        self.destroy.assert_called_once_with()
        self.assertIsNone(trainer._tracker)
        self.assertFalse(trainer._runtime_started)
        self._assert_caches_cleared()

    def test_service_failures_still_release_groups_and_reset_state(self) -> None:
        """A failure at each cleanup stage must not prevent later stages from running."""
        trainer = object.__new__(SyncTrainer)
        tracker = MagicMock()
        tracker.finish.side_effect = RuntimeError("tracker failed")
        trainer._tracker = tracker
        trainer.rollout_manager = MagicMock()
        trainer.rollout_manager.close.side_effect = OSError("manager failed")
        trainer.rollout_engine = MagicMock()
        trainer.rollout_engine.close.side_effect = subprocess.SubprocessError("engine failed")
        trainer._runtime_started = True
        self.destroy.side_effect = RuntimeError("backend teardown failed")

        with self.assertLogs(runtime.logger, level="WARNING") as logs:
            trainer._cleanup()

        self.assertEqual(len(logs.records), 4)
        tracker.finish.assert_called_once_with()
        trainer.rollout_manager.close.assert_called_once_with()
        trainer.rollout_engine.close.assert_called_once_with()
        self.destroy.assert_called_once_with()
        self.assertIsNone(trainer._tracker)
        self.assertFalse(trainer._runtime_started)
        self._assert_caches_cleared()

    def test_partial_initialization_closes_tracker_without_destroying_groups(self) -> None:
        """Cleanup accepts a Trainer that has not created rollout services or a group."""
        trainer = object.__new__(SyncTrainer)
        tracker = MagicMock()
        trainer._tracker = tracker
        trainer._runtime_started = False

        trainer._cleanup()
        trainer._cleanup()

        tracker.finish.assert_called_once_with()
        self.destroy.assert_not_called()
        self.assertIsNone(trainer._tracker)
        self.assertFalse(trainer._runtime_started)
