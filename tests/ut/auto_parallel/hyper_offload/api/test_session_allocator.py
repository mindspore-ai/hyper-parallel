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
"""CPU tests for session allocator activation on all execution paths."""

from contextlib import ExitStack
import unittest
from unittest.mock import MagicMock, patch

from hyper_parallel.auto_parallel.hyper_offload import OffloadConfig, OffloadSession
from hyper_parallel.auto_parallel.hyper_offload.ir.trace import ActivationTrace
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
class TestSessionAllocatorLifecycle(unittest.TestCase):
    """Exercise session entry without initializing real accelerator allocators."""

    def setUp(self) -> None:
        """Replace hardware-dependent executors and allocation owners."""
        stack = ExitStack()
        self.addCleanup(stack.close)
        executor = stack.enter_context(patch(
            "hyper_parallel.auto_parallel.hyper_offload.api.session.WarmupExecutor",
        )).return_value
        self.executor = executor
        executor.finish.return_value = (ActivationTrace(), [])
        stack.enter_context(patch("hyper_parallel.auto_parallel.hyper_offload.api.session.ResidencyManager"))
        factory = stack.enter_context(patch("hyper_parallel.auto_parallel.hyper_offload.api.session.SessionAllocator"))
        self.allocator = factory.return_value
        self.session = OffloadSession(OffloadConfig())

    def test_allocator_is_lazy_and_reused_for_warmup_and_native_replay(self) -> None:
        """Feature: Lazy allocator activation.

        Description: Construct a session, run warmup and then a native replay.
        Expectation: Resolve the same allocator on each entry without replacing it.
        """
        self.allocator.activate.assert_not_called()
        with self.session:
            pass
        with self.session:
            pass
        self.assertEqual(self.allocator.activate.call_count, 2)
        self.assertTrue(self.session._native_replay)
        self.assertIsNone(OffloadSession.get_active())

    def test_allocator_failure_does_not_enter_runtime_contexts(self) -> None:
        """Feature: Failed allocator initialization.

        Description: Native configuration fails before dispatch or autograd entry.
        Expectation: Propagate failure without activating session runtime state.
        """
        self.allocator.activate.side_effect = RuntimeError("allocator failed")
        with self.assertRaisesRegex(RuntimeError, "allocator failed"), self.session:
            pass
        self.executor.reset.assert_not_called()
        self.assertIsNone(OffloadSession.get_active())

    def test_native_saved_tensor_replay_also_activates_the_allocator(self) -> None:
        """Feature: Saved-tensor replay allocator ownership.

        Description: Enter a session whose native saved-tensor replay is ready.
        Expectation: Activate its allocator before installing saved-tensor hooks.
        """
        replay = MagicMock()
        self.session._saved_replay = replay
        with self.session:
            self.assertTrue(self.session.executor.native_saved_replay)
        self.allocator.activate.assert_called_once_with()
        replay.finish.assert_called_once_with()
        self.assertFalse(self.session.executor.native_saved_replay)
        self.assertIsNone(OffloadSession.get_active())

    def test_close_releases_policy_after_optimizer_work(self) -> None:
        """Keep the lease across context exits and release it only after optimizer work."""
        with self.session:
            with self.assertRaisesRegex(RuntimeError, "cannot close an active"):
                self.session.close()
        self.allocator.close.assert_not_called()
        self.session.close()
        self.allocator.close.assert_called_once_with()
