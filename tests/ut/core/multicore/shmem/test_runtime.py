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
"""Unit tests for the private SHMEM reference-counted lifecycle."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from hyper_parallel.core.multicore.shmem import _lifecycle


class TestRuntimeLifecycle(unittest.TestCase):
    """Validate lifecycle coordination without Torch distributed or NPU hardware."""

    def setUp(self) -> None:
        """Install deterministic framework and Native boundaries."""
        self.world = object()
        self.dist = SimpleNamespace(
            group=SimpleNamespace(WORLD=self.world),
            is_initialized=Mock(return_value=True),
            get_world_size=Mock(return_value=2),
            get_process_group_ranks=Mock(return_value=[0, 1]),
            get_rank=Mock(return_value=0),
            barrier=Mock(),
        )
        self.torch = SimpleNamespace(npu=SimpleNamespace(synchronize=Mock()))
        self.native = SimpleNamespace(
            _initialize=Mock(),
            _validate_shutdown=Mock(),
            _shutdown=Mock(),
        )
        self.framework_patch = patch.object(_lifecycle, "_torch_modules", return_value=(self.torch, self.dist))
        self.native_patch = patch.object(_lifecycle, "_load_native", return_value=self.native)
        self.framework_patch.start()
        self.native_patch.start()
        self.addCleanup(self.framework_patch.stop)
        self.addCleanup(self.native_patch.stop)
        self.addCleanup(self._reset_runtime)
        self._reset_runtime()

    @staticmethod
    def _reset_runtime() -> None:
        """Reset private process state after each isolated test."""
        _lifecycle._users = 0  # pylint: disable=protected-access
        _lifecycle._root_group = None  # pylint: disable=protected-access
        _lifecycle._root_uses_distributed = None  # pylint: disable=protected-access
        _lifecycle._shutdown_failed = False  # pylint: disable=protected-access

    def test_single_process_runtime_is_reference_counted(self) -> None:
        """Keep one Root PE active until its final reference is released."""
        self.dist.is_initialized.return_value = False
        self.assertEqual(_lifecycle._reference_count(), 0)  # pylint: disable=protected-access

        _lifecycle.acquire()
        self.assertEqual(_lifecycle._reference_count(), 1)  # pylint: disable=protected-access
        _lifecycle.acquire()
        self.assertEqual(_lifecycle._reference_count(), 2)  # pylint: disable=protected-access
        _lifecycle.release()
        self.assertEqual(_lifecycle._reference_count(), 1)  # pylint: disable=protected-access

        self.native._validate_shutdown.assert_not_called()
        self.torch.npu.synchronize.assert_not_called()
        self.native._shutdown.assert_not_called()

        _lifecycle.release()
        self.assertEqual(_lifecycle._reference_count(), 0)  # pylint: disable=protected-access

        self.native._initialize.assert_called_once_with(0, 1)
        self.native._validate_shutdown.assert_called_once_with()
        self.torch.npu.synchronize.assert_called_once_with()
        self.dist.barrier.assert_not_called()
        self.native._shutdown.assert_called_once_with()

    def test_equivalent_process_group_shares_runtime(self) -> None:
        """Count a distinct ProcessGroup object describing the same complete WORLD."""
        equivalent_world = object()

        _lifecycle.acquire(self.world)
        _lifecycle.acquire(equivalent_world)

        self.native._initialize.assert_called_once_with(0, 2)
        self.dist.barrier.assert_called_once_with(group=self.world)
        self.assertIs(_lifecycle._root_group, self.world)  # pylint: disable=protected-access
        _lifecycle.release()
        self.native._shutdown.assert_not_called()
        _lifecycle.release()

    def test_different_root_is_rejected(self) -> None:
        """Reject a later ProcessGroup with a different ordered membership."""
        different_group = object()

        def ranks(group: object) -> list[int]:
            """Return complete WORLD only for the lifecycle's first ProcessGroup."""
            return [0, 1] if group is self.world else [1, 0]

        self.dist.get_process_group_ranks.side_effect = ranks
        _lifecycle.acquire(self.world)

        with self.assertRaisesRegex(RuntimeError, "complete Torch WORLD"):
            _lifecycle.acquire(different_group)

        self.native._initialize.assert_called_once_with(0, 2)
        self.assertEqual(_lifecycle._reference_count(), 1)  # pylint: disable=protected-access
        _lifecycle.release()

    def test_distributed_state_change_is_rejected(self) -> None:
        """Keep the framework mode fixed throughout one active lifecycle."""
        self.dist.is_initialized.return_value = False
        _lifecycle.acquire()
        self.dist.is_initialized.return_value = True

        with self.assertRaisesRegex(RuntimeError, "state changed"):
            _lifecycle.acquire()
        with self.assertRaisesRegex(RuntimeError, "state changed"):
            _lifecycle.release()

        self.dist.is_initialized.return_value = False
        _lifecycle.release()

    def test_collective_runtime_operation_order(self) -> None:
        """Order Root convergence before initialization and quiescence before shutdown."""
        operations = Mock()
        operations.attach_mock(self.dist.barrier, "barrier")
        operations.attach_mock(self.native._initialize, "initialize")
        operations.attach_mock(self.native._validate_shutdown, "validate_shutdown")
        operations.attach_mock(self.torch.npu.synchronize, "synchronize")
        operations.attach_mock(self.native._shutdown, "shutdown")

        _lifecycle.acquire()
        _lifecycle.release()

        self.assertEqual(
            operations.mock_calls,
            [
                call.barrier(group=self.world),
                call.initialize(0, 2),
                call.validate_shutdown(),
                call.synchronize(),
                call.barrier(group=self.world),
                call.shutdown(),
            ],
        )

    def test_clean_release_allows_new_runtime(self) -> None:
        """Run Native initialization and shutdown again after a clean lifecycle."""
        for _ in range(2):
            _lifecycle.acquire()
            _lifecycle.release()

        self.assertEqual(self.native._initialize.call_count, 2)
        self.assertEqual(self.native._shutdown.call_count, 2)
        self.assertEqual(self.dist.barrier.call_count, 4)
        self.assertFalse(_lifecycle._shutdown_failed)  # pylint: disable=protected-access

    def test_entry_barrier_failure_is_retryable(self) -> None:
        """Retry when Root convergence fails before entering Native initialization."""
        self.dist.barrier.side_effect = RuntimeError("root convergence failed")

        with self.assertRaisesRegex(RuntimeError, "root convergence failed"):
            _lifecycle.acquire()

        self.native._initialize.assert_not_called()
        self.assertEqual(_lifecycle._reference_count(), 0)  # pylint: disable=protected-access
        self.assertFalse(_lifecycle._shutdown_failed)  # pylint: disable=protected-access
        self.dist.barrier.side_effect = None
        _lifecycle.acquire()
        _lifecycle.release()

    def test_final_release_precondition_failure_is_retryable(self) -> None:
        """Retain lifecycle state when shutdown validation, sync, or barrier fails."""
        for operation in (self.native._validate_shutdown, self.torch.npu.synchronize, self.dist.barrier):
            with self.subTest(operation=operation):
                self._reset_runtime()
                self.native._validate_shutdown.reset_mock(side_effect=True)
                self.torch.npu.synchronize.reset_mock(side_effect=True)
                self.dist.barrier.reset_mock(side_effect=True)
                self.native._shutdown.reset_mock(side_effect=True)
                _lifecycle.acquire()
                self.dist.barrier.reset_mock()
                operation.side_effect = RuntimeError("quiescence failed")

                with self.assertRaisesRegex(RuntimeError, "quiescence failed"):
                    _lifecycle.release()

                self.assertEqual(_lifecycle._reference_count(), 1)  # pylint: disable=protected-access
                self.assertIsNotNone(_lifecycle._root_uses_distributed)  # pylint: disable=protected-access
                self.assertFalse(_lifecycle._shutdown_failed)  # pylint: disable=protected-access
                self.native._shutdown.assert_not_called()
                operation.side_effect = None
                _lifecycle.release()

    def test_native_initialize_failure_is_retryable(self) -> None:
        """Retry Native initialization after a failed attempt; only shutdown failure blocks."""
        self.native._initialize.side_effect = RuntimeError("initialize failed")

        with self.assertRaisesRegex(RuntimeError, "initialize failed"):
            _lifecycle.acquire()

        self.assertEqual(_lifecycle._reference_count(), 0)  # pylint: disable=protected-access
        self.assertFalse(_lifecycle._shutdown_failed)  # pylint: disable=protected-access
        self.assertIsNone(_lifecycle._root_uses_distributed)  # pylint: disable=protected-access

        self.native._initialize.side_effect = None
        _lifecycle.acquire()
        self.assertEqual(self.native._initialize.call_count, 2)
        _lifecycle.release()

    def test_native_shutdown_failure_blocks_future_acquire(self) -> None:
        """Clear Root ownership but reject a new lifecycle after Native shutdown fails."""
        _lifecycle.acquire()
        self.native._shutdown.side_effect = RuntimeError("finalize failed")

        with self.assertRaisesRegex(RuntimeError, "finalize failed"):
            _lifecycle.release()

        self.assertEqual(_lifecycle._reference_count(), 0)  # pylint: disable=protected-access
        self.assertIsNone(_lifecycle._root_uses_distributed)  # pylint: disable=protected-access
        self.assertTrue(_lifecycle._shutdown_failed)  # pylint: disable=protected-access
        with self.assertRaisesRegex(RuntimeError, "Native shutdown failure"):
            _lifecycle.acquire()

    def test_release_without_reference_is_rejected(self) -> None:
        """Expose an unmatched release instead of silently underflowing the user count."""
        with self.assertRaisesRegex(RuntimeError, "no active reference"):
            _lifecycle.release()


if __name__ == "__main__":
    unittest.main()
