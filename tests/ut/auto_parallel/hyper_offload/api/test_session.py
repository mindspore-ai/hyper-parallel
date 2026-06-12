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
"""Unit tests for OffloadSession lifecycle."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from hyper_parallel.auto_parallel.hyper_offload import OffloadConfig, OffloadSession
from hyper_parallel.auto_parallel.hyper_offload.execution.replay import ReplayExecutor
from hyper_parallel.auto_parallel.hyper_offload.execution.warmup import WarmupExecutor


class _MockExecutorContext:
    """Context manager that applies patches needed by OffloadSession tests.

    Patches executor-level components (``DeviceTimer``,
    ``profile_transfer_bandwidth``) and accelerator-level helpers
    (``torch.Stream``, ``torch.accelerator.current_stream``) so that
    :class:`OffloadSession` can be constructed and exercised on
    CPU-only hosts.
    """

    def __init__(self) -> None:
        self._patches = []

    def start(self) -> None:
        """Apply patches for CPU-only testing."""
        # Patch DeviceTimer so WarmupExecutor doesn't need accelerator.
        timer_patch = patch(
            "hyper_parallel.auto_parallel.hyper_offload.execution.warmup.executor.DeviceTimer"
        )
        mock_timer_cls = timer_patch.start()
        mock_timer_cls.return_value.start.return_value = None
        mock_timer_cls.return_value.stop.return_value = 0.0
        self._patches.append(timer_patch)

        # Patch profile_transfer_bandwidth so finish() returns dummy values.
        bw_patch = patch(
            "hyper_parallel.auto_parallel.hyper_offload.execution.warmup.executor.profile_transfer_bandwidth",
            return_value=(16.0, 16.0),
        )
        bw_patch.start()
        self._patches.append(bw_patch)

        # Patch torch.Stream and accelerator stream for wait_for_transfers.
        stream_patch = patch("torch.Stream")
        stream_patch.start()
        self._patches.append(stream_patch)

        acc_stream_patch = patch("torch.accelerator.current_stream")
        mock_stream = acc_stream_patch.start()
        mock_stream.return_value.wait_stream.return_value = None
        mock_stream.return_value.synchronize.return_value = None
        self._patches.append(acc_stream_patch)

    def stop(self) -> None:
        for p in reversed(self._patches):
            p.stop()


class TestOffloadSessionLifecycle(unittest.TestCase):
    """OffloadSession enter/exit, warmup→replay transition, error cleanup."""

    def setUp(self) -> None:
        self._mock_ctx = _MockExecutorContext()
        self._mock_ctx.start()

    def tearDown(self) -> None:
        self._mock_ctx.stop()

    def test_enter_exit_safe(self) -> None:
        """Entering and exiting without model operations should be safe."""
        session = OffloadSession(OffloadConfig(max_resident_activation_mb=1))
        with session:
            pass
        self.assertEqual(session.mode, "replay")

    def test_context_manager_entry_point(self) -> None:
        """The context manager should return the session itself."""
        config = OffloadConfig(max_resident_activation_mb=1)
        with OffloadSession(config) as sess:
            self.assertIsInstance(sess, OffloadSession)
            self.assertEqual(sess.mode, "warmup")

    def test_warmup_creates_warmup_executor(self) -> None:
        """During warmup, the executor should be a WarmupExecutor."""
        session = OffloadSession(OffloadConfig())
        with session:
            self.assertIsInstance(session.executor, WarmupExecutor)
        self.assertEqual(session.mode, "replay")

    def test_replay_creates_replay_executor(self) -> None:
        """After warmup exit, the executor should switch to ReplayExecutor."""
        session = OffloadSession(OffloadConfig(max_resident_activation_mb=1))
        with session:
            _ = torch.randn(4, 4) + 1
        self.assertIsInstance(session.executor, ReplayExecutor)

    def test_second_enter_resets_op_idx(self) -> None:
        """Second enter should reset op_idx for a new pass."""
        session = OffloadSession(OffloadConfig(max_resident_activation_mb=1))
        with session:
            _ = torch.randn(4, 4) + 1
        self.assertEqual(session.mode, "replay")
        # After warmup, finish() resets the executor, so op_idx is -1
        self.assertEqual(session.executor.op_idx, -1)

        # Second pass: should still have op_idx = -1 at entry
        with session:
            self.assertEqual(session.executor.op_idx, -1)

    def test_tensor_operation_during_session(self) -> None:
        """Simple tensor operations should work inside a session."""
        session = OffloadSession(OffloadConfig(max_resident_activation_mb=1))
        with session:
            t = torch.randn(4, 4, requires_grad=True)
            result = t + 1
            loss = result.sum()
            loss.backward()
        self.assertEqual(session.mode, "replay")

    def test_tensor_operation_during_replay(self) -> None:
        """Replay should also work and not crash."""
        session = OffloadSession(OffloadConfig(max_resident_activation_mb=1))
        # Warmup
        with session:
            t = torch.randn(4, 4, requires_grad=True)
            result = t + 1
            loss = result.sum()
            loss.backward()

        # Replay
        with session:
            t = torch.randn(4, 4, requires_grad=True)
            result = t + 1
            loss = result.sum()
            loss.backward()

    def test_exception_in_warmup_clears_runtime(self) -> None:
        """If an exception occurs during warmup, the runtime should be cleared."""
        session = OffloadSession(OffloadConfig(max_resident_activation_mb=16))

        class _RaisesOnForward(torch.nn.Module):
            """Module that raises during forward."""

            def forward(self, x):
                raise ValueError("boom")

        model = _RaisesOnForward()
        with self.assertRaisesRegex(ValueError, "boom"):
            with session:
                model(torch.randn(4, 4))

        # State should be cleaned up
        self.assertEqual(len(session.executor._alive_shadows), 0)  # pylint: disable=protected-access

    def test_config_plumbs_host_memory(self) -> None:
        """Config values should be plumbed to the residency manager."""
        config = OffloadConfig(max_offload_activation_mb=512)
        session = OffloadSession(config)
        self.assertEqual(
            session.residency_manager._host_pool.max_host_bytes,  # pylint: disable=protected-access
            512 * 1024**2,
        )

    def test_config_defaults(self) -> None:
        """Default config should set a large host pool."""
        session = OffloadSession(OffloadConfig())
        self.assertEqual(
            session.residency_manager._host_pool.max_host_bytes,  # pylint: disable=protected-access
            65536 * 1024**2,
        )

    def test_get_active_inside_session(self) -> None:
        """get_active() should return the current session inside the context."""
        session = OffloadSession(OffloadConfig())
        self.assertIsNone(OffloadSession.get_active())
        with session:
            self.assertIs(OffloadSession.get_active(), session)
        self.assertIsNone(OffloadSession.get_active())

    def test_get_active_outside_session(self) -> None:
        """get_active() should return None outside a session."""
        self.assertIsNone(OffloadSession.get_active())

    def test_dispatch_mode_delegates_to_executor(self) -> None:
        """The dispatch mode should delegate to the executor's dispatch method."""
        session = OffloadSession(OffloadConfig(max_resident_activation_mb=1))
        t = torch.randn(2, 2)
        expected = t + t
        with session:
            result = t + t
            self.assertIsNotNone(result)
            # Ops should have been recorded by the executor
            self.assertGreater(len(session.executor._ops), 0)  # pylint: disable=protected-access
            # The result should be mathematically correct
            torch.testing.assert_close(result, expected)

    def test_multiple_tensor_ops_in_session(self) -> None:
        """Multiple tensor operations should each be recorded as separate ops."""
        session = OffloadSession(OffloadConfig(max_resident_activation_mb=1))
        with session:
            t = torch.randn(4, 4, requires_grad=True)
            t = t + 1
            t = t * 2
            t = t.sin()
            loss = t.sum()
            loss.backward()
            # Should have multiple ops recorded (access inside session while WarmupExecutor)
            self.assertGreaterEqual(len(session.executor._ops), 4)  # pylint: disable=protected-access

    def test_replay_op_idx_tracking(self) -> None:
        """During replay, op_idx should increment through ops."""
        session = OffloadSession(OffloadConfig(max_resident_activation_mb=1))
        t = torch.randn(4, 4, requires_grad=True)

        # First pass: warmup
        with session:
            _ = t + 1
        self.assertEqual(session.mode, "replay")

        # Second pass: replay mode, op_idx starts at -1 and increments
        with session:
            self.assertEqual(session.executor.op_idx, -1)
            _ = t + 1
            # After executing one op in replay, op_idx should be 0
            self.assertEqual(session.executor.op_idx, 0)
