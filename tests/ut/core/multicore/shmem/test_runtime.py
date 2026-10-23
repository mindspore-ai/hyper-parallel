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

import torch

from hyper_parallel.core.multicore import _collective
from hyper_parallel.core.multicore.shmem import _lifecycle
from tests.common.mark_utils import arg_mark


# These tests deliberately inspect the private state machine and mocked native entry points.
class TestRuntimeLifecycle(unittest.TestCase):
    """Validate lifecycle coordination without Torch distributed or NPU hardware."""

    # pylint: disable=protected-access

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
            broadcast_object_list=Mock(),
            all_gather_object=Mock(
                side_effect=lambda output, value, **_kw: output.__setitem__(slice(None), [value] * 2)),
        )
        self.torch = SimpleNamespace(npu=SimpleNamespace(
            synchronize=Mock(),
            is_current_stream_capturing=Mock(return_value=False),
            current_device=Mock(return_value=0),
        ), device=lambda device_type, index: f"{device_type}:{index}")
        self.native = SimpleNamespace(
            _initialize=Mock(),
            _get_unique_id=Mock(return_value=b"group-bootstrap"),
            _validate_shutdown=Mock(return_value=None),
            _shutdown=Mock(return_value=None),
            _debug_state=Mock(return_value={
                "state": "Ready",
                "config": {"heap_size_bytes": 128},
                "device_index": 0,
                "active_allocations": [],
            }),
        )
        self.framework_patch = patch.object(_lifecycle, "_torch_modules", return_value=(self.torch, self.dist))
        self.native_patch = patch.object(_lifecycle, "_load_native", return_value=self.native)
        self.collective_patch = patch.object(_collective, "dist", self.dist)
        self.collective_patch.start()
        self.addCleanup(self.collective_patch.stop)
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
        _lifecycle._root_ranks = None  # pylint: disable=protected-access
        _lifecycle._root_size = None  # pylint: disable=protected-access
        _lifecycle._device_index = None  # pylint: disable=protected-access
        _lifecycle._runtime_invalidated = False  # pylint: disable=protected-access

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

        with self.assertRaisesRegex(RuntimeError, "different ordered membership"):
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
        with self.assertRaisesRegex(RuntimeError, "restart the process"):
            _lifecycle.release()
        self.assertTrue(_lifecycle._runtime_invalidated)  # pylint: disable=protected-access

    def test_collective_runtime_operation_order(self) -> None:
        """Order Root convergence before initialization and quiescence before shutdown."""
        operations = Mock()
        operations.attach_mock(self.dist.barrier, "barrier")
        operations.attach_mock(self.native._initialize, "initialize")
        operations.attach_mock(self.native._validate_shutdown, "validate_shutdown")
        operations.attach_mock(self.torch.npu.synchronize, "synchronize")
        operations.attach_mock(self.native._shutdown, "shutdown")
        self.native._shutdown.side_effect = lambda: self.assertTrue(_lifecycle._runtime_invalidated)

        _lifecycle.acquire()
        _lifecycle.release()

        self.assertEqual(
            operations.mock_calls,
            [
                call.barrier(group=self.world),
                call.initialize(0, 2),
                call.validate_shutdown(),
                call.synchronize(),
                call.shutdown(),
            ],
        )

    def test_bootstrap_failure_has_root_rank_summary_and_full_cause(self) -> None:
        """The bootstrap error names its owner without duplicating the Native exception type."""
        native_error = RuntimeError(
            "[HYPER_PARALLEL_SHMEM_ERROR] error_code=CANN_ERROR operation=get_unique_id "
            "phase=CannCall cann_error_code=-4 message=SHMEM unique ID creation failed: "
            "ACLSHMEM_INNER_ERROR(-4), cann_detail=[[SHM_SHMEM bootstrap.cpp:42] failed]"
        )
        self.native._get_unique_id.side_effect = native_error

        with self.assertRaises(RuntimeError) as raised:
            _lifecycle._subgroup_unique_id(  # pylint: disable=protected-access
                self.native,
                self.dist,
                self.world,
                0,
                (0, 1),
            )

        self.assertEqual(
            str(raised.exception),
            "SHMEM bootstrap failed on root_rank=0: "
            "SHMEM unique ID creation failed: ACLSHMEM_INNER_ERROR(-4)",
        )
        self.assertNotIn("RuntimeError:", str(raised.exception))
        self.assertNotIn("cann_detail", str(raised.exception))
        self.assertIs(raised.exception.__cause__, native_error)

    def test_clean_release_allows_new_runtime(self) -> None:
        """Run Native initialization and shutdown again after a clean lifecycle."""
        for _ in range(2):
            _lifecycle.acquire()
            _lifecycle.release()

        self.assertEqual(self.native._initialize.call_count, 2)
        self.assertEqual(self.native._shutdown.call_count, 2)
        self.assertEqual(self.dist.barrier.call_count, 2)
        self.assertEqual(self.dist.all_gather_object.call_count, 6)
        self.assertFalse(_lifecycle._runtime_invalidated)  # pylint: disable=protected-access

    def test_entry_barrier_failure_is_retryable(self) -> None:
        """Retry when Root convergence fails before entering Native initialization."""
        self.dist.barrier.side_effect = RuntimeError("root convergence failed")

        with self.assertRaisesRegex(RuntimeError, "root convergence failed"):
            _lifecycle.acquire()

        self.native._initialize.assert_not_called()
        self.assertEqual(_lifecycle._reference_count(), 0)  # pylint: disable=protected-access
        self.assertFalse(_lifecycle._runtime_invalidated)  # pylint: disable=protected-access
        self.dist.barrier.side_effect = None
        _lifecycle.acquire()
        _lifecycle.release()

    def test_final_release_precondition_failure_is_retryable(self) -> None:
        """Retry validation failures but stop after NPU quiescence fails."""
        cases = (
            (self.native._validate_shutdown, False, "The active SHMEM Runtime remains usable."),
            (self.torch.npu.synchronize, True, "restart the process"),
        )
        for operation, terminal, expected_hint in cases:
            with self.subTest(operation=operation, terminal=terminal):
                self._reset_runtime()
                self.native._validate_shutdown.reset_mock(side_effect=True)
                self.torch.npu.synchronize.reset_mock(side_effect=True)
                self.dist.barrier.reset_mock(side_effect=True)
                self.native._shutdown.reset_mock(side_effect=True)
                _lifecycle.acquire()
                self.dist.barrier.reset_mock()
                operation.side_effect = RuntimeError("quiescence failed")

                with self.assertRaises(RuntimeError) as raised:
                    _lifecycle.release()

                self.assertIn("quiescence failed", str(raised.exception))
                self.assertIn(expected_hint, str(raised.exception))
                self.assertIs(raised.exception.__cause__, operation.side_effect)
                payload = self.dist.all_gather_object.call_args.args[1]
                self.assertEqual(payload, (None, ("RuntimeError", "quiescence failed"), terminal))
                self.assertEqual(_lifecycle._reference_count(), 1)  # pylint: disable=protected-access
                self.assertIsNotNone(_lifecycle._root_uses_distributed)  # pylint: disable=protected-access
                self.assertEqual(_lifecycle._runtime_invalidated, terminal)  # pylint: disable=protected-access
                self.native._shutdown.assert_not_called()
                if not terminal:
                    operation.side_effect = None
                    _lifecycle.release()

    def test_native_initialize_failure_is_retryable(self) -> None:
        """Retry Native initialization after a failed attempt; only shutdown failure blocks."""
        self.native._initialize.side_effect = RuntimeError("initialize failed")

        with self.assertRaisesRegex(RuntimeError, "initialize failed"):
            _lifecycle.acquire()

        self.assertEqual(_lifecycle._reference_count(), 0)  # pylint: disable=protected-access
        self.assertFalse(_lifecycle._runtime_invalidated)  # pylint: disable=protected-access
        self.assertIsNone(_lifecycle._root_uses_distributed)  # pylint: disable=protected-access

        self.native._initialize.side_effect = None
        _lifecycle.acquire()
        self.assertEqual(self.native._initialize.call_count, 2)
        _lifecycle.release()

    def test_peer_shutdown_errors_are_coordinated(self) -> None:
        """A peer precheck failure blocks quiescence; a peer Native failure is terminal."""
        _lifecycle.acquire()
        self.dist.barrier.reset_mock()

        def peer_preflight_error(output: list, value: object, **_kwargs: object) -> None:
            """Inject one retryable peer-side release preflight failure."""
            output[:] = [value, (None, ("RuntimeError", "peer busy"), False)]

        self.dist.all_gather_object.side_effect = peer_preflight_error
        with self.assertRaises(RuntimeError) as raised:
            _lifecycle.release()
        self.assertEqual(
            str(raised.exception),
            "SHMEM release failed during Preflight on root_rank=1: peer busy. "
            "The active SHMEM Runtime remains usable.",
        )
        self.dist.barrier.assert_not_called()
        self.native._shutdown.assert_not_called()
        self.assertEqual(_lifecycle._reference_count(), 1)
        self.dist.all_gather_object.side_effect = (
            lambda output, value, **_kwargs: output.__setitem__(slice(None), [value] * 2)
        )
        shutdown_stage = 0

        def peer_shutdown_error(output: list, value: object, **_kwargs: object) -> None:
            """Inject a peer failure only after release preflight and quiescence succeed."""
            nonlocal shutdown_stage
            shutdown_stage += 1
            if shutdown_stage == 3:
                output[:] = [value, (None, ("RuntimeError", "peer finalize"), False)]
            else:
                output[:] = [value] * 2

        self.dist.all_gather_object.side_effect = peer_shutdown_error
        with self.assertRaises(RuntimeError) as raised:
            _lifecycle.release()
        self.assertEqual(
            str(raised.exception),
            "SHMEM release failed during Shutdown on root_rank=1: peer finalize. "
            "The SHMEM Runtime is no longer usable; restart the process.",
        )
        self.native._shutdown.assert_called_once_with()
        self.assertTrue(_lifecycle._runtime_invalidated)
        with self.assertRaisesRegex(RuntimeError, "restart the process"):
            _lifecycle.acquire()

    def test_release_convergence_failure_invalidates_runtime(self) -> None:
        """Do not promise retry after Root ranks fail to converge release results."""
        for method in ("is_initialized", "get_world_size", "all_gather_object"):
            with self.subTest(method=method):
                self._reset_runtime()
                _lifecycle.acquire()
                self.dist.all_gather_object.reset_mock()
                collective_error = ValueError("invalid group") if method != "all_gather_object" else RuntimeError(
                    "gather unavailable"
                )
                effect = [True, collective_error] if method == "is_initialized" else collective_error
                with patch.object(self.dist, method, side_effect=effect), self.assertRaises(RuntimeError) as raised:
                    _lifecycle.release()

                self.assertIn("SHMEM release failed during Preflight while converging", str(raised.exception))
                self.assertIn("restart the process", str(raised.exception))
                self.assertNotIn("remains usable", str(raised.exception))
                self.assertIs(raised.exception.__cause__, collective_error)
                self.assertEqual(_lifecycle._reference_count(), 1)
                self.assertTrue(_lifecycle._runtime_invalidated)
                if method != "all_gather_object":
                    self.dist.all_gather_object.assert_not_called()
                self.torch.npu.synchronize.assert_not_called()
                self.native._shutdown.assert_not_called()

    def test_native_shutdown_failure_blocks_future_acquire(self) -> None:
        """Clear Root ownership but reject a new lifecycle after Native shutdown fails."""
        _lifecycle.acquire()
        self.native._shutdown.side_effect = RuntimeError("finalize failed")

        with self.assertRaisesRegex(RuntimeError, "finalize failed"):
            _lifecycle.release()

        self.assertEqual(_lifecycle._reference_count(), 0)  # pylint: disable=protected-access
        self.assertIsNone(_lifecycle._root_uses_distributed)  # pylint: disable=protected-access
        self.assertTrue(_lifecycle._runtime_invalidated)  # pylint: disable=protected-access
        with self.assertRaisesRegex(RuntimeError, "restart the process"):
            _lifecycle.acquire()

    def test_release_without_reference_is_rejected(self) -> None:
        """Expose an unmatched release instead of silently underflowing the user count."""
        with self.assertRaisesRegex(RuntimeError, "no active reference"):
            _lifecycle.release()

    def test_runtime_restart_preserves_collective_operation_order(self) -> None:
        """Rebuild explicit Allocations around one Native restart."""
        _lifecycle.acquire()
        operations = []
        clock = [10.0]
        self.torch.npu.synchronize.side_effect = lambda _device: operations.append("quiesce")

        def validate_shutdown() -> None:
            """Record validation and advance the deterministic stage clock."""
            operations.append("validate")
            clock[0] += 0.002

        def shutdown() -> None:
            """Record shutdown and advance the deterministic stage clock."""
            operations.append("finalize")
            clock[0] += 0.003

        self.native._validate_shutdown.side_effect = validate_shutdown
        self.native._shutdown.side_effect = shutdown
        current = Mock(shape=(4,), dtype=torch.float32, _base=None)
        current.data_ptr.return_value = 100
        current.numel.return_value = 4
        current.element_size.return_value = 4
        current.is_contiguous.return_value = True
        replacement = torch.empty(8, dtype=torch.float32)
        active_allocations = [{"allocation_base": 100, "allocation_bytes": 16}]

        def debug_state() -> dict:
            """Expose the mutable mocked Allocation registry."""
            return {
                "state": "Ready",
                "config": {"heap_size_bytes": 256},
                "device_index": 0,
                "active_allocations": list(active_allocations),
            }

        self.native._debug_state.side_effect = debug_state

        def bootstrap() -> bytes:
            """Record bootstrap ordering and return one group identifier."""
            operations.append("bootstrap")
            return b"replacement-bootstrap"

        def initialize(*_args: object) -> None:
            """Record Native initialize ordering."""
            operations.append("initialize")

        self.native._get_unique_id.side_effect = bootstrap
        self.native._initialize.side_effect = initialize
        allocation = _lifecycle._AllocationReplacement(  # pylint: disable=protected-access
            current,
            _lifecycle._AllocationSpec((8,), torch.float32, 16),  # pylint: disable=protected-access
        )

        def free(tensor: object) -> None:
            """Record release ordering and clear the mocked registry."""
            del tensor
            operations.append("free")
            active_allocations.clear()

        with (
            patch("hyper_parallel.core.multicore.shmem._api.free", side_effect=free),
            patch("hyper_parallel.core.multicore.shmem._api.empty", side_effect=lambda *_a, **_k: replacement),
            patch.object(_lifecycle.time, "perf_counter", side_effect=lambda: clock[0]),
            patch.object(
                _lifecycle,
                "_reconfiguration_stage",
                wraps=_lifecycle._reconfiguration_stage,  # pylint: disable=protected-access
            ) as stage_call,
        ):
            result = _lifecycle._restart(  # pylint: disable=protected-access
                heap_size_bytes=256,
                replacements=(allocation,),
                preflight_hook=lambda: ("layout", 1),
            )

        self.assertEqual(
            operations,
            ["quiesce", "free", "validate", "finalize", "bootstrap", "initialize"],
        )
        self.assertEqual(
            set(result.stages_ms),
            {
                "preflight_ms", "quiesce_ms", "release_ms", "finalize_ms",
                "bootstrap_ms", "initialize_ms", "restore_ms",
            },
        )
        self.assertEqual(result.tensors, (replacement,))
        self.assertAlmostEqual(result.stages_ms["finalize_ms"], 5.0)
        self.assertEqual(
            [call.kwargs["stage"] for call in stage_call.call_args_list],
            [
                "Preflight",
                "Quiesce",
                "ReleaseAllocations",
                "ValidateShutdown",
                "Shutdown",
                "Initialize",
                "RestoreAllocations",
            ],
        )
        self.assertTrue(all(call.kwargs["root_group"] is self.world for call in stage_call.call_args_list))
        self.assertEqual(self.dist.all_gather_object.call_count, 7)
        payloads = [call.args[1] for call in self.dist.all_gather_object.call_args_list]
        self.assertEqual(payloads[0], (
            (256, ("layout", 1), (((4,), "torch.float32", 16),), (((8,), "torch.float32", 16, 32),)),
            None,
            False,
        ))
        self.assertEqual(payloads[1:], [(None, None, False)] * 6)
        self.assertFalse(_lifecycle._runtime_invalidated)  # pylint: disable=protected-access

    def test_runtime_restart_preflight_failure_preserves_runtime(self) -> None:
        """Converge ownership, device, Allocation and hook failures before teardown."""
        _lifecycle.acquire()
        _lifecycle.acquire()
        with self.assertRaisesRegex(RuntimeError, "one healthy Runtime user"):
            _lifecycle._restart(  # pylint: disable=protected-access
                heap_size_bytes=256,
                replacements=(),
                preflight_hook=lambda: "state",
            )
        _lifecycle.release()
        self.assertFalse(_lifecycle._runtime_invalidated)  # pylint: disable=protected-access

        self.torch.npu.current_device.return_value = 1
        with self.assertRaisesRegex(RuntimeError, "current NPU device"):
            _lifecycle._restart(  # pylint: disable=protected-access
                heap_size_bytes=256,
                replacements=(),
                preflight_hook=lambda: "state",
            )
        self.torch.npu.current_device.return_value = 0

        self.native._debug_state.return_value = {
            "state": "Ready",
            "config": {"heap_size_bytes": 128},
            "device_index": 0,
            "active_allocations": [{"allocation_base": 1234, "allocation_bytes": 16}],
        }
        with self.assertRaisesRegex(RuntimeError, "cover every active"):
            _lifecycle._restart(  # pylint: disable=protected-access
                heap_size_bytes=256,
                replacements=(),
                preflight_hook=lambda: "state",
            )
        current = Mock(shape=(4,), dtype=torch.float32, _base=None)
        current.data_ptr.return_value = 1234
        current.numel.return_value = 4
        current.element_size.return_value = 4
        current.is_contiguous.return_value = True
        for alignment in (None, 0, 3):
            with self.subTest(alignment=alignment), self.assertRaisesRegex(RuntimeError, "alignment_bytes"):
                invalid_target = _lifecycle._AllocationSpec(  # type: ignore[arg-type]
                    (8,), torch.float32, alignment,
                )
                replacement = _lifecycle._AllocationReplacement(current, invalid_target)
                _lifecycle._restart(  # pylint: disable=protected-access
                    heap_size_bytes=256,
                    replacements=(replacement,),
                    preflight_hook=lambda: "state",
                )
        self.torch.npu.is_current_stream_capturing.return_value = True
        with self.assertRaisesRegex(RuntimeError, "graph capture"):
            _lifecycle._restart(  # pylint: disable=protected-access
                heap_size_bytes=256,
                replacements=(),
                preflight_hook=lambda: "state",
            )
        self.torch.npu.is_current_stream_capturing.return_value = False
        self.native._debug_state.return_value = {
            "state": "Ready", "config": {"heap_size_bytes": 128}, "device_index": 0, "active_allocations": [],
        }

        def peer_preflight_error(output: list, value: object, **_kwargs: object) -> None:
            """Inject one peer-side preflight failure."""
            output[:] = [value, (None, ("RuntimeError", "peer preflight failed"), False)]

        self.dist.all_gather_object.side_effect = peer_preflight_error
        with self.assertRaises(RuntimeError) as raised:
            _lifecycle._restart(  # pylint: disable=protected-access
                heap_size_bytes=256,
                replacements=(),
                preflight_hook=lambda: "state",
            )
        self.assertEqual(
            str(raised.exception),
            "SHMEM restart failed during Preflight on root_rank=1: peer preflight failed. "
            "The active SHMEM Runtime remains usable.",
        )
        self.assertNotIn("None", str(raised.exception))
        self.assertNotIn("state", str(raised.exception))
        self.assertIsNone(raised.exception.__cause__)
        self.torch.npu.synchronize.assert_not_called()
        self.native._shutdown.assert_not_called()
        self.assertFalse(_lifecycle._runtime_invalidated)  # pylint: disable=protected-access

        def peer_terminal_error(output: list, value: object, **_kwargs: object) -> None:
            """Inject one peer-side terminal lifecycle failure."""
            output[:] = [value, (None, ("RuntimeError", "peer Runtime is terminal"), True)]

        self.dist.all_gather_object.side_effect = peer_terminal_error
        with self.assertRaises(RuntimeError) as raised:
            _lifecycle._restart(  # pylint: disable=protected-access
                heap_size_bytes=256,
                replacements=(),
                preflight_hook=lambda: "state",
            )
        self.assertEqual(
            str(raised.exception),
            "SHMEM restart failed during Preflight on root_rank=1: peer Runtime is terminal. "
            "The SHMEM Runtime is no longer usable; restart the process.",
        )
        self.assertIsNone(raised.exception.__cause__)
        self.assertTrue(_lifecycle._runtime_invalidated)  # pylint: disable=protected-access

    def test_restart_local_value_error_keeps_traceback_and_root_owner(self) -> None:
        """A one-rank validation failure includes Root ownership and its Python cause."""
        self.dist.is_initialized.return_value = False
        _lifecycle.acquire()

        with self.assertRaises(RuntimeError) as raised:
            _lifecycle._restart(  # pylint: disable=protected-access
                heap_size_bytes=0,
                replacements=(),
                preflight_hook=lambda: "state",
            )

        self.assertEqual(
            str(raised.exception),
            "SHMEM restart failed during Preflight on root_rank=0: "
            "ValueError: heap_size_bytes must be a positive integer. "
            "The active SHMEM Runtime remains usable.",
        )
        self.assertIsInstance(raised.exception.__cause__, ValueError)
        self.assertIn("heap_size_bytes", str(raised.exception.__cause__))
        self.assertTrue(_lifecycle._runtime_is_healthy())  # pylint: disable=protected-access

    def test_restart_collective_failure_reports_current_stage_and_invalidates_runtime(self) -> None:
        """A convergence failure reports its stage and stops further Runtime use."""
        for method in ("is_initialized", "get_world_size", "all_gather_object"):
            with self.subTest(method=method):
                self._reset_runtime()
                _lifecycle.acquire()
                self.dist.all_gather_object.reset_mock()
                collective_error = ValueError("invalid group") if method != "all_gather_object" else RuntimeError(
                    "gather unavailable"
                )
                effect = [True, collective_error] if method == "is_initialized" else collective_error
                with patch.object(self.dist, method, side_effect=effect), self.assertRaises(RuntimeError) as raised:
                    _lifecycle._restart(
                        heap_size_bytes=256,
                        replacements=(),
                        preflight_hook=lambda: "state",
                    )

                self.assertIn("SHMEM restart failed during Preflight while converging", str(raised.exception))
                self.assertIn("restart the process", str(raised.exception))
                self.assertNotIn("remains usable", str(raised.exception))
                self.assertIs(raised.exception.__cause__, collective_error)
                self.assertFalse(_lifecycle._runtime_is_healthy())
                self.assertTrue(_lifecycle._runtime_invalidated)
                if method != "all_gather_object":
                    self.dist.all_gather_object.assert_not_called()
                self.torch.npu.synchronize.assert_not_called()
                self.native._shutdown.assert_not_called()

    def test_restart_from_terminal_runtime_keeps_terminal_guidance(self) -> None:
        """Treat existing terminal state, lifecycle drift and quiescence errors as terminal."""
        cases = (
            "already invalidated",
            "distributed state changed",
            "Native state unavailable",
            "Native state changed",
            "Native device changed",
            "quiescence failed",
        )
        for scenario in cases:
            with self.subTest(scenario=scenario):
                self._reset_runtime()
                self.dist.is_initialized.return_value = True
                self.native._debug_state.reset_mock(side_effect=True)
                self.native._debug_state.return_value = {
                    "state": "Ready",
                    "config": {"heap_size_bytes": 128},
                    "device_index": 0,
                    "active_allocations": [],
                }
                self.torch.npu.synchronize.reset_mock(side_effect=True)
                _lifecycle.acquire()

                expected_cause = None
                if scenario == "already invalidated":
                    _lifecycle._runtime_invalidated = True  # pylint: disable=protected-access
                elif scenario == "distributed state changed":
                    self.dist.is_initialized.return_value = False
                elif scenario == "Native state unavailable":
                    expected_cause = RuntimeError("Native state unavailable")
                    self.native._debug_state.side_effect = expected_cause
                elif scenario == "Native state changed":
                    self.native._debug_state.return_value["state"] = "Uninitialized"
                elif scenario == "Native device changed":
                    self.native._debug_state.return_value["device_index"] = 1
                else:
                    expected_cause = RuntimeError("device synchronize failed")
                    self.torch.npu.synchronize.side_effect = expected_cause

                with self.assertRaises(RuntimeError) as raised:
                    _lifecycle._restart(  # pylint: disable=protected-access
                        heap_size_bytes=256,
                        replacements=(),
                        preflight_hook=lambda: "state",
                    )

                self.assertIn("restart the process", str(raised.exception))
                self.assertNotIn("remains usable", str(raised.exception))
                self.assertIs(raised.exception.__cause__, expected_cause)
                self.assertFalse(_lifecycle._runtime_is_healthy())  # pylint: disable=protected-access

    def test_runtime_restart_destructive_failure_is_terminal(self) -> None:
        """An Allocation release failure invalidates the Runtime."""
        _lifecycle.acquire()
        native_error = RuntimeError("finalize failed")
        self.native._shutdown.side_effect = native_error
        with self.assertRaises(RuntimeError) as raised:
            _lifecycle._restart(  # pylint: disable=protected-access
                heap_size_bytes=256,
                replacements=(),
                preflight_hook=lambda: "state",
            )
        self.assertIn("SHMEM restart failed during Shutdown", str(raised.exception))
        self.assertIn("root_ranks=[0, 1]: finalize failed", str(raised.exception))
        self.assertIn("restart the process", str(raised.exception))
        self.assertIs(raised.exception.__cause__, native_error)
        self.assertTrue(_lifecycle._runtime_invalidated)  # pylint: disable=protected-access

    def test_runtime_restart_bootstrap_failure_is_flattened(self) -> None:
        """Bootstrap context appears once and retains the Native cause."""
        _lifecycle.acquire()
        native_error = RuntimeError(
            "[HYPER_PARALLEL_SHMEM_ERROR] error_code=CANN_ERROR operation=get_unique_id "
            "phase=CannCall cann_error_code=-4 message=SHMEM unique ID creation failed: "
            "ACLSHMEM_INNER_ERROR(-4), cann_detail=[[SHM_SHMEM bootstrap.cpp:42] failed]"
        )
        self.native._get_unique_id.side_effect = native_error

        with self.assertRaises(RuntimeError) as raised:
            _lifecycle._restart(  # pylint: disable=protected-access
                heap_size_bytes=256,
                replacements=(),
                preflight_hook=lambda: "state",
            )

        self.assertEqual(
            str(raised.exception),
            "SHMEM restart failed during Bootstrap on root_rank=0: "
            "SHMEM unique ID creation failed: ACLSHMEM_INNER_ERROR(-4). "
            "The SHMEM Runtime is no longer usable; restart the process.",
        )
        self.assertEqual(str(raised.exception).count("SHMEM restart failed"), 1)
        self.assertEqual(str(raised.exception).count("Bootstrap"), 1)
        self.assertEqual(str(raised.exception).count("root_rank=0"), 1)
        self.assertIs(raised.exception.__cause__, native_error)
        self.assertTrue(_lifecycle._runtime_invalidated)  # pylint: disable=protected-access

    def test_runtime_restart_native_failure_has_readable_summary_and_full_cause(self) -> None:
        """The outer error is concise, and its cause retains Native metadata."""
        self.dist.is_initialized.return_value = False
        _lifecycle.acquire()
        native_error = RuntimeError(
            "[HYPER_PARALLEL_SHMEM_ERROR] error_code=CANN_ERROR operation=initialize "
            "phase=CannCall cann_error_code=-3 message=SHMEM initialization failed on NPU device 0: "
            "ACLSHMEM_SMEM_ERROR(-3), requested_heap=24.00 GiB, free_large_page_hbm=12.34 GiB, "
            "total_large_page_hbm=52.41 GiB, requested_heap_exceeds_free_large_page_hbm=true, "
            "cann_detail=[[SHM_SHMEM hybm_vmm_based_segment.cpp:173] HalMemCreate failed]"
        )
        self.native._initialize.side_effect = native_error

        with self.assertRaises(RuntimeError) as raised:
            _lifecycle._restart(  # pylint: disable=protected-access
                heap_size_bytes=24 * 1024**3,
                replacements=(),
                preflight_hook=lambda: "state",
            )

        message = str(raised.exception)
        self.assertIn("SHMEM restart failed during Initialize on root_rank=0", message)
        self.assertIn("ACLSHMEM_SMEM_ERROR(-3)", message)
        self.assertIn("requested_heap=24.00 GiB", message)
        self.assertIn("free_large_page_hbm=12.34 GiB", message)
        self.assertIn("restart the process", message)
        self.assertNotIn("HYPER_PARALLEL_SHMEM_ERROR", message)
        self.assertNotIn("cann_detail", message)
        self.assertIs(raised.exception.__cause__, native_error)
        self.assertTrue(_lifecycle._runtime_invalidated)  # pylint: disable=protected-access
        with self.assertRaisesRegex(RuntimeError, "restart the process"):
            _lifecycle.acquire()
        with self.assertRaisesRegex(RuntimeError, "restart the process"):
            _lifecycle.release()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_release_stages_use_the_acquired_root_group(self) -> None:
        """Feature: Explicit SHMEM Root selection.
        Description: Release a Runtime acquired on a subgroup.
        Expectation: Preflight, quiescence and shutdown all converge on that subgroup.
        """
        subgroup = object()
        _lifecycle.acquire(subgroup)
        self.dist.all_gather_object.reset_mock()

        _lifecycle.release()

        self.assertEqual(self.dist.all_gather_object.call_count, 3)
        self.assertTrue(
            all(item.kwargs["group"] is subgroup for item in self.dist.all_gather_object.call_args_list)
        )


if __name__ == "__main__":
    unittest.main()
