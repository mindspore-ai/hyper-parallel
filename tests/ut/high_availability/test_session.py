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
"""MindIO transaction ordering and callback semantics with CPU-only state."""

import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import torch

from hyper_parallel import DeviceMesh, Shard, distribute_tensor
from hyper_parallel.core.distributed_checkpoint import standard_planner
from hyper_parallel.high_availability.config import RecoveryConfig
from hyper_parallel.high_availability.groups import (
    GroupDescriptor, GroupRegistry, is_recovery_bootstrapping, is_recovery_enabled, is_recovery_reboot,
)
from hyper_parallel.high_availability.session import MetadataMirrorError, RecoverySession
from hyper_parallel.high_availability.state import ModelOptimizerState, TrainState
from hyper_parallel.high_availability.transport import DeviceMetadataBank, decode
from hyper_parallel.core.optimizer.adamw import AdamW
from hyper_parallel.core.optimizer.muon import Muon
from tests.ut.high_availability.test_state import StatefulValue


class TestRecoverySession(unittest.TestCase):
    """Assert externally visible ordering rather than replacing the state codec."""

    def setUp(self) -> None:
        """Bind real CPU state codecs to mocked control-plane endpoints."""
        self.session = RecoverySession.__new__(RecoverySession)
        self.session.rank = self.session.local_rank = 1
        self.session.world_size = 2
        self.session.config = RecoveryConfig(True, "127.0.0.1")
        self.session.sdk = MagicMock()
        self.session.npu = MagicMock()
        self.session.npu.stop_device.return_value = None
        self.session.channel = MagicMock()
        self.session.step = 0
        self.session._running = True
        self.session._in_update = False
        self.session._stopped = False
        self.session._cleanup_callbacks = []
        self.model = torch.nn.Linear(2, 2)
        self.session.model_state = ModelOptimizerState(self.model, AdamW(self.model.parameters()))
        self.cursor = StatefulValue()
        self.session.train_state = TrainState(self.cursor, {})
        self.session.bank = DeviceMetadataBank(1, torch.device("cpu"))
        self.session._live_step = 0

    def test_clean_resets_activation_swap_only_after_device_restart(self) -> None:
        """Feature: Activation buffers are recycled only after the stopped device restarts.

        Description: Observe shard abort, device restart and activation reset during clean.
        Expectation: Reset follows restart; cleanup exits the stopped state without changing tensor safety flags.
        """
        session = self.session
        session._stopped = True
        session.adapter = MagicMock()
        events = []
        session.adapter.abort_iteration.side_effect = lambda: events.append("abort")
        session.npu.restart_device.side_effect = lambda rank: events.append("restart")
        session.adapter.reset_after_device_restart.side_effect = lambda synchronize: events.append("reset")
        self.assertEqual(session._clean(False, None, None), 0)
        session.adapter.reset_after_device_restart.assert_called_once_with(session.npu.synchronize)
        self.assertEqual(events, ["abort", "restart", "reset"])
        self.assertFalse(session._stopped)

    def test_clean_aborts_checkpoint_before_restarting_device(self) -> None:
        """Feature: Async checkpoint cleanup. Description: Clean a stopped survivor.
        Expectation: Local persistence is fenced before device restart and rollback.
        """
        session = self.session
        session._stopped = True
        session.adapter = MagicMock()
        events = []
        session.register_cleanup(lambda: events.append("checkpoint"))
        session.npu.restart_device.side_effect = lambda rank: events.append("restart")
        session._clean(False, None, None)
        self.assertEqual(events, ["checkpoint", "restart"])

    def test_failed_activation_reset_does_not_complete_clean(self) -> None:
        """Feature: Failed activation cleanup does not acknowledge a successful clean.

        Description: Raise from post-restart activation quiescence.
        Expectation: Propagate the error and preserve the stopped flag.
        """
        session = self.session
        session._stopped = True
        session.adapter = MagicMock()
        session.adapter.reset_after_device_restart.side_effect = RuntimeError("activation reset failed")
        with self.assertRaisesRegex(RuntimeError, "activation reset failed"):
            session._clean(False, None, None)
        self.assertTrue(session._stopped)

    def test_update_reports_end_only_after_metadata_mirror(self):
        """Feature: Optimizer update publication ordering.

        Description: Commit an update while observing the SDK and mirror calls.
        Expectation: Metadata mirroring succeeds before End publishes the committed step.
        """
        events = []
        self.session.sdk.tft_start_updating_os.side_effect = lambda *_: events.append("start")
        self.session.channel.mirror_metadata.side_effect = lambda *_: events.append("metadata_ack")
        self.session.sdk.tft_end_updating_os.side_effect = lambda step: events.append(("end", step, self.session.step))
        with self.session.update(1):
            self.cursor.value = 1
            events.append("optimizer_and_scheduler")
        self.assertEqual(events, ["start", "optimizer_and_scheduler", "metadata_ack", ("end", 1, 0)])
        self.assertEqual(self.session.step, 1)
        metadata = decode(self.session.bank.get(1, 1)["payload"].numpy().tobytes())
        self.assertEqual(metadata["state"]["rank_state"]["loader"]["value"], 1)

    def test_hccl_update_uses_barrier_without_a_duplicate_pre_update_device_fence(self) -> None:
        """Feature: One HCCL barrier covers local readiness before the SDK update window.

        Description: Observe barrier, optimizer, post-update fence and metadata commit ordering.
        Expectation: Only the post-update explicit fence remains and valid live state is not published early.
        """
        session = self.session
        events = []
        session.sdk.tft_start_updating_os.side_effect = lambda: events.append("start")
        session.sdk.tft_end_updating_os.side_effect = lambda step: events.append("end")
        session.channel.mirror_metadata.side_effect = lambda step: events.append("mirror")

        def _fence():
            self.assertIsNone(session._live_step)
            events.append("device_sync")

        session.npu.synchronize.side_effect = _fence
        with patch("hyper_parallel.high_availability.session.dist.is_initialized", return_value=True):
            with patch("hyper_parallel.high_availability.session.dist.barrier",
                       side_effect=lambda: events.append("barrier")) as barrier:
                with session.update(1):
                    events.append("optimizer")
        self.assertEqual(events, ["barrier", "start", "optimizer", "device_sync", "mirror", "end"])
        session.npu.synchronize.assert_called_once_with()
        barrier.assert_called_once_with()
        self.assertEqual(session._live_step, 1)

    def test_failed_update_barrier_does_not_enter_sdk_update_window(self) -> None:
        """Feature: Barrier failures remain outside the optimizer update transaction.

        Description: Fail the pre-update HCCL barrier before optimizer work starts.
        Expectation: Keep the old live version and never report SDK Updating or Updated.
        """
        session = self.session
        with patch("hyper_parallel.high_availability.session.dist.is_initialized", return_value=True):
            with patch("hyper_parallel.high_availability.session.dist.barrier",
                       side_effect=RuntimeError("HCCL barrier failed")):
                with self.assertRaisesRegex(RuntimeError, "HCCL barrier failed"):
                    with session.update(1):
                        self.fail("optimizer must not run")
        session.sdk.tft_start_updating_os.assert_not_called()
        session.sdk.tft_end_updating_os.assert_not_called()
        session.npu.synchronize.assert_not_called()
        self.assertEqual(session._live_step, 0)
        self.assertEqual(session.step, 0)
        self.assertFalse(session._in_update)

    def test_failed_post_update_fence_never_publishes_live_state(self) -> None:
        """Feature: The post-update fence protects the recovery version boundary.

        Description: Fail device synchronization after the optimizer returns.
        Expectation: Do not publish a live version, mirror metadata or acknowledge Updated.
        """
        session = self.session
        session.npu.synchronize.side_effect = RuntimeError("device fence failed")
        with patch("hyper_parallel.high_availability.session.dist.is_initialized", return_value=True):
            with patch("hyper_parallel.high_availability.session.dist.barrier"):
                with self.assertRaisesRegex(RuntimeError, "device fence failed"):
                    with session.update(1):
                        pass
        session.sdk.tft_start_updating_os.assert_called_once_with()
        session.sdk.tft_end_updating_os.assert_not_called()
        session.channel.mirror_metadata.assert_not_called()
        self.assertIsNone(session._live_step)
        self.assertEqual(session.step, 0)
        self.assertFalse(session._in_update)

    def test_failed_update_never_reports_end(self):
        with self.assertRaisesRegex(RuntimeError, "optimizer failed"):
            with self.session.update(1):
                raise RuntimeError("optimizer failed")
        self.session.sdk.tft_end_updating_os.assert_not_called()
        self.session.channel.mirror_metadata.assert_not_called()
        self.assertEqual(self.session.step, 0)
        self.assertFalse(self.session._in_update)

    def test_sdk_destroys_controller_and_processor_after_native_group_teardown(self):
        """Feature: SDK-managed shutdown independent of the native training group.

        Description: Finish the native body, reject later device collectives and execute normal SDK cleanup.
        Expectation: The SDK closes Controller then Processor without a HyperParallel exit barrier or duplicate cleanup.
        """
        events = []
        session = self._prepare_run(controller_initialized=True)

        def _decorate(function):
            def _wrapper(*args):
                result = function(*args)
                session.sdk.tft_destroy_controller()
                session.sdk.tft_destroy_processor()
                return result
            return _wrapper

        session.sdk.tft_exception_handler.side_effect = _decorate
        session.sdk.tft_destroy_controller.side_effect = lambda: events.append("controller")
        session.sdk.tft_destroy_processor.side_effect = lambda: events.append("processor")

        def _body():
            events.append("native_destroy")
            barrier.side_effect = AssertionError("training process group is gone")
            return 12

        with patch("hyper_parallel.high_availability.session.dist.barrier") as barrier:
            self.assertEqual(session.run(_body), 12)
        barrier.assert_not_called()
        self.assertEqual(events, ["native_destroy", "controller", "processor"])
        session.sdk.tft_destroy_controller.assert_called_once()
        session.sdk.tft_destroy_processor.assert_called_once()
        self.assertFalse(session._processor_initialized)
        self.assertFalse(session._controller_initialized)

    def _prepare_run(self, controller_initialized: bool = False) -> RecoverySession:
        """Bind the session and initialize its committed state for a local training run."""
        session = self.session
        session.rank = session.local_rank = 0
        session.bank = DeviceMetadataBank(0, torch.device("cpu"))
        session._running = False
        session._processor_initialized = True
        session._controller_initialized = controller_initialized
        session.reboot = False
        session.groups = GroupRegistry()
        def _decorate(function):
            def _wrapper(active):
                result = function(active)
                if active._controller_initialized and not os.getenv("MINDX_TASK_ID"):
                    active.sdk.tft_destroy_controller()
                active.sdk.tft_destroy_processor()
                return result
            return _wrapper

        session.sdk.tft_exception_handler.side_effect = _decorate
        return session

    def test_sdk_shutdown_failure_preserves_the_completed_training_body(self):
        """Feature: Completed training survives an SDK shutdown failure.

        Description: Fail SDK cleanup after the native body has returned.
        Expectation: The error propagates through fallback cleanup without repeating the completed body.
        """
        session = self._prepare_run()
        error = RuntimeError("SDK shutdown failed")
        body = MagicMock(return_value=12)
        completed = []

        def _decorate(function):
            def _wrapper(active):
                completed.append(function(active))
                raise error
            return _wrapper

        session.sdk.tft_exception_handler.side_effect = _decorate
        with patch("hyper_parallel.high_availability.session.dist.barrier"):
            with self.assertRaises(RuntimeError) as caught:
                session.run(body)
        self.assertIs(caught.exception, error)
        self.assertEqual(completed, [12])
        body.assert_called_once_with()
        session.sdk.tft_report_error.assert_not_called()

    def test_initial_progress_follows_native_checkpoint_loading(self):
        """Feature: Initial progress follows the native training-start callbacks.

        Description: Load a nonzero initial step and align progress twice.
        Expectation: Reuse bound state and report the loaded step once without capturing or mirroring metadata.
        """
        session = self._prepare_run()
        state = session.model_state

        def _body():
            session.channel.mirror_metadata.assert_not_called()
            self.cursor.value = 8
            with torch.no_grad():
                state.model.weight.fill_(8)
            session.initialize_training_progress(8)
            session.initialize_training_progress(9)
            self.assertEqual(session.step, 8)
            self.assertEqual(session._live_step, 8)
            self.assertIs(session.model_state, state)
            self.assertTrue(torch.equal(state.parameters["weight"], torch.full_like(state.model.weight, 8)))
            return 12

        with patch.object(session, "_capture_metadata") as capture:
            self.assertEqual(session.run(_body), 12)
        capture.assert_not_called()
        session.channel.mirror_metadata.assert_not_called()
        session.sdk.tft_report_load_ckpt_step.assert_called_once_with(8)

    def test_zero_initial_progress_does_not_publish_recovery_state(self):
        """Feature: Fresh training has no recoverable initial metadata version.

        Description: Initialize progress at zero before the first optimizer update.
        Expectation: No metadata capture, mirror, or checkpoint-step report is performed.
        """
        session = self._prepare_run()
        with patch.object(session, "_capture_metadata") as capture:
            session.run(lambda: session.initialize_training_progress(0))
        capture.assert_not_called()
        session.channel.mirror_metadata.assert_not_called()
        session.sdk.tft_report_load_ckpt_step.assert_not_called()
        self.assertEqual(session.step, 0)
        self.assertFalse(session._initial_progress_pending)

    def test_initial_progress_rejects_replaced_parameters(self):
        """Feature: Initialization must preserve the validated persistent parameters.

        Description: Replace a model parameter after attach but before initial progress synchronization.
        Expectation: Reject the replacement before reporting a checkpoint step or mirroring metadata.
        """
        session = self._prepare_run()

        def _body():
            session.model_state.model.weight = torch.nn.Parameter(session.model_state.model.weight.detach().clone())
            session.initialize_training_progress(8)

        with self.assertRaisesRegex(RuntimeError, "stable persistent"):
            session.run(_body)
        session.sdk.tft_report_load_ckpt_step.assert_not_called()
        session.channel.mirror_metadata.assert_not_called()

    def test_training_exception_releases_processor_before_controller(self):
        """Feature: Recovery runtime cleanup after training failure.

        Description: Propagate a non-recoverable exception from the training body.
        Expectation: Processor destruction precedes owned Controller destruction.
        """
        session = self._prepare_run(controller_initialized=True)
        events = []
        session.sdk.tft_destroy_processor.side_effect = lambda: events.append("processor")
        session.sdk.tft_destroy_controller.side_effect = lambda: events.append("controller")
        error = ValueError("training failed")
        with patch("hyper_parallel.high_availability.session.dist.barrier"):
            with self.assertRaises(ValueError) as caught:
                session.run(MagicMock(side_effect=error))
        self.assertIs(caught.exception, error)
        self.assertEqual(events, ["processor", "controller"])
        self.assertFalse(session._processor_initialized)
        self.assertFalse(session._controller_initialized)
        self.assertFalse(session._running)
        session.close()
        session.sdk.tft_destroy_processor.assert_called_once()
        session.sdk.tft_destroy_controller.assert_called_once()

    def test_sdk_decorator_failure_leaves_processor_available_for_caller_cleanup(self):
        """Feature: Recovery wrapper construction failure.

        Description: Fail before the SDK returns its training wrapper.
        Expectation: The session remains outside running state so its caller can release the Processor.
        """
        session = self._prepare_run()
        session.sdk.tft_exception_handler.side_effect = ValueError("wrapper failed")
        body = MagicMock()
        with self.assertRaisesRegex(ValueError, "wrapper failed"):
            session.run(body)
        body.assert_not_called()
        self.assertFalse(session._running)
        self.assertTrue(session._processor_initialized)
        session.close()
        session.sdk.tft_destroy_processor.assert_called_once()
        self.assertFalse(session._processor_initialized)
        self.assertFalse(session._running)

    def test_sdk_retry_keeps_runtime_alive_until_training_finishes(self):
        """Feature: Processor lifetime during recoverable training errors.

        Description: Retry the training body within the SDK exception loop.
        Expectation: close is deferred until the wrapper finishes using the runtime.
        """
        session = self._prepare_run()

        def _decorate(function):
            def _retry(active):
                with self.assertRaisesRegex(RuntimeError, "recoverable"):
                    function(active)
                active.close()
                active.sdk.tft_destroy_processor.assert_not_called()
                self.assertTrue(active._processor_initialized)
                self.assertTrue(active._running)
                result = function(active)
                active.sdk.tft_destroy_processor()
                return result
            return _retry

        session.sdk.tft_exception_handler.side_effect = _decorate
        body = MagicMock(side_effect=[RuntimeError("recoverable"), 12])
        with patch("hyper_parallel.high_availability.session.dist.barrier"):
            self.assertEqual(session.run(body), 12)
        self.assertEqual(body.call_count, 2)
        session.sdk.tft_destroy_processor.assert_called_once()
        self.assertFalse(session._processor_initialized)
        self.assertFalse(session._running)

    def test_cleanup_failure_chains_training_error_and_preserves_resource_flags(self):
        """Feature: Error chaining during recovery teardown.

        Description: Fail training and then fail Processor destruction.
        Expectation: Cleanup failure propagates with the training error as context and live resources remain retryable.
        """
        session = self._prepare_run(controller_initialized=True)
        session.sdk.tft_destroy_processor.side_effect = RuntimeError("cleanup failed")
        error = ValueError("training failed")
        with patch("hyper_parallel.high_availability.session.dist.barrier"):
            with self.assertRaisesRegex(RuntimeError, "cleanup failed") as caught:
                session.run(MagicMock(side_effect=error))
        self.assertIs(caught.exception.__context__, error)
        self.assertTrue(session._processor_initialized)
        self.assertTrue(session._controller_initialized)
        self.assertFalse(session._running)
        session.sdk.tft_destroy_controller.assert_not_called()
        session.sdk.tft_destroy_processor.side_effect = None
        session.close()
        self.assertFalse(session._processor_initialized)
        self.assertFalse(session._controller_initialized)
        session.sdk.tft_destroy_controller.assert_called_once()

    def test_sdk_controller_cleanup_failure_is_retried_by_close(self):
        """Feature: Controller cleanup fallback after SDK teardown failure.

        Description: Fail SDK Controller destruction after training and let close complete cleanup.
        Expectation: The SDK error propagates, Controller cleanup is retried, and resource flags clear on success.
        """
        session = self._prepare_run(controller_initialized=True)
        error = ValueError("controller cleanup failed")
        session.sdk.tft_destroy_controller.side_effect = [error, None]
        with patch.dict("os.environ"), \
                patch("hyper_parallel.high_availability.session.dist.barrier"):
            os.environ.pop("MINDX_TASK_ID", None)
            with self.assertRaisesRegex(ValueError, "controller cleanup failed") as caught:
                session.run(lambda: 12)
        self.assertIs(caught.exception, error)
        self.assertEqual(session.sdk.tft_destroy_controller.call_count, 2)
        session.sdk.tft_destroy_processor.assert_called_once_with()
        self.assertFalse(session._processor_initialized)
        self.assertFalse(session._controller_initialized)
        self.assertFalse(session._running)

    def test_taskd_controller_is_not_destroyed_by_worker(self):
        """Feature: External Controller lifetime on normal TaskD completion.

        Description: Complete SDK-managed training while TaskD owns the Controller.
        Expectation: Only the local Processor is destroyed and close does not destroy an external Controller.
        """
        session = self._prepare_run()
        with patch.dict("os.environ", {"MINDX_TASK_ID": "test-job"}), \
                patch("hyper_parallel.high_availability.session.dist.barrier"):
            self.assertEqual(session.run(lambda: 12), 12)
        session.sdk.tft_destroy_controller.assert_not_called()
        session.sdk.tft_destroy_processor.assert_called_once_with()
        self.assertFalse(session._processor_initialized)
        self.assertFalse(session._controller_initialized)

    def test_replacement_resumes_tensor_distribution_after_repair(self):
        """Feature: Replacement training communication after repair.

        Description: Reenter training with a sealed replacement group registry.
        Expectation: Tensor distribution communicates normally while initial state publication remains disabled.
        """
        session = self.session
        session._running = False
        session._processor_initialized = True
        session._controller_initialized = False
        session.reboot = True
        session.groups = GroupRegistry()
        session.groups.reboot = True
        session.groups.sealed = True
        mesh = DeviceMesh("cpu", [0, 1], mesh_dim_names=("tp",), _init_backend=False)
        tensor = torch.arange(16).reshape(4, 4)

        def _decorate(function):
            def _retry(active):
                with self.assertRaisesRegex(RuntimeError, "ARF FINISH"):
                    function(active)
                self.assertTrue(active.groups.reboot)
                active.step = active._live_step = 8
                return function(active)
            return _retry

        def _body():
            self.assertEqual(session.step, 8)
            session.initialize_training_progress(session.step)
            session.channel.mirror_metadata.assert_not_called()
            session.sdk.tft_report_load_ckpt_step.assert_not_called()
            return distribute_tensor(tensor, mesh, [Shard(0)], src_data_rank=0)

        session.sdk.tft_exception_handler.side_effect = _decorate
        with patch("hyper_parallel.high_availability.session.dist.barrier"), \
                patch("hyper_parallel.core.dtensor.dtensor._distribute_tensor_with_communication",
                      return_value=tensor[:2]) as communicate:
            result = session.run(_body)
        communicate.assert_called_once()
        self.assertIs(communicate.call_args.args[0], tensor)
        self.assertEqual(communicate.call_args.args[-1], 0)
        torch.testing.assert_close(result.to_local(), tensor[:2])
        self.assertTrue(session.groups.reboot)

    def test_closed_session_cannot_run_training_again(self):
        """Feature: Recovery session lifecycle validation.

        Description: Attempt training with attached state and a closed Processor.
        Expectation: The session rejects the call before invoking the training body.
        """
        session = self.session
        session._running = False
        session._processor_initialized = False
        session._controller_initialized = False
        session.reboot = False
        session.groups = GroupRegistry()
        session.sdk.tft_exception_handler.side_effect = lambda function: function
        body = MagicMock()
        with patch("hyper_parallel.high_availability.session.dist.barrier"):
            with self.assertRaisesRegex(RuntimeError, "initialize.*attach"):
                session.run(body)
        body.assert_not_called()
        session.sdk.tft_exception_handler.assert_not_called()

    def test_metadata_failure_does_not_commit_or_enter_runtime_error_retry(self):
        """Feature: Metadata mirror failure is terminal.

        Description: Fail required mirroring with runtime and non-runtime communication errors.
        Expectation: No step is committed or paused, and the original error is chained outside RuntimeError retry.
        """
        for error in (RuntimeError("SUSPECT REMOTE ERROR"), ValueError("metadata unavailable")):
            with self.subTest(error=type(error).__name__):
                self.session.channel.mirror_metadata.side_effect = error
                with self.assertRaisesRegex(MetadataMirrorError, "step 1; recovery is unavailable") as caught:
                    with self.session.update(1):
                        pass
                self.assertIs(caught.exception.__cause__, error)
                self.assertNotIsInstance(caught.exception, RuntimeError)
                self.session.sdk.tft_end_updating_os.assert_not_called()
                self.session.sdk.tft_pause_train.assert_not_called()
                self.assertEqual(self.session.step, 0)
                self.assertEqual(self.session._live_step, 1)
                self.assertFalse(self.session._in_update)

    def test_stop_observed_at_cpu_boundary(self):
        self.session._stop([], None)
        self.session.npu.stop_device.assert_called_once_with(1)
        with self.assertRaisesRegex(RuntimeError, "FORCE STOP"):
            self.session.pause()

    def test_rollback_invalidates_checkpoint_plans_before_training_resumes(self) -> None:
        """Feature: New and surviving workers rebuild checkpoint plans together.

        Description: Roll back a committed step while a surviving worker has cached save metadata.
        Expectation: Its next planner lookup misses, matching a replacement with an empty cache.
        """
        with patch.dict(standard_planner.StandardSavePlanner.cached_save_result, {}, clear=True):
            planner = standard_planner.StandardSavePlanner()
            planner.configure_planner({"weight": self.model.weight})
            planner.cache_result(MagicMock(), MagicMock())
            self.assertIsNotNone(planner.get_cached())
            with self.session.update(1):
                pass
            self.session._rollback(1, [], None)
            self.assertIsNone(planner.get_cached())
            self.assertEqual(standard_planner.StandardSavePlanner.cached_save_result, {})

    def test_rollback_never_restores_weights_from_cpu(self):
        with self.session.update(1):
            self.cursor.value = 8
        self.cursor.value = 9
        self.session._rollback(1, [], None)
        self.assertEqual(self.cursor.value, 8)
        self.session._live_step = None
        with self.assertRaisesRegex(RuntimeError, "replica repair"):
            self.session._rollback(1, [], None)

    def test_repair_uses_mindio_rank_list_and_releases_temporary_group(self):
        """Feature: MindIO-selected repair participants.

        Description: Execute send and receive callbacks after the global rebuild phase.
        Expectation: Transfers use a temporary group from MindIO and always release it after repair.
        """
        self.session.sdk.RepairType = SimpleNamespace(RT_SEND=SimpleNamespace(value=1),
                                                     RT_RECV_REPAIR=SimpleNamespace(value=2))
        self.session._live_step = 8
        self.session.groups = GroupRegistry()
        for sending in (True, False):
            with self.subTest(sending=sending):
                self.session.rank = 5 if sending else 1
                self.session.channel.reset_mock()
                group = object()
                info = {"repair_type": 1 if sending else 2, "src": [5], "dst": [1], "rank_list": [1, 5]}
                with patch("hyper_parallel.high_availability.session.dist.new_group",
                           return_value=group) as create, \
                        patch("hyper_parallel.high_availability.session.dist.destroy_process_group") as destroy:
                    self.session._repair(8, True, [1], info, [], None)
                create.assert_called_once_with(ranks=[1, 5], backend="hccl", use_local_synchronization=True)
                destroy.assert_called_once_with(group)
                transfer = self.session.channel.send if sending else self.session.channel.receive
                transfer.assert_called_once_with(1 if sending else 5, 8, group)

    def test_repair_reuses_exact_groups_without_destroying_them(self):
        """Feature: Reuse rebuilt communication groups during replica repair.

        Description: Send or receive over an exact registered group or WORLD, including transfer failure.
        Expectation: Select the earliest exact handle and never create or destroy a group on the reuse path.
        """
        self.session.sdk.RepairType = SimpleNamespace(
            RT_SEND=SimpleNamespace(value=1),
            RT_RECV_REPAIR=SimpleNamespace(value=2),
        )
        for registered in (False, True):
            for sending in (False, True):
                for fail in (False, True):
                    with self.subTest(registered=registered, sending=sending, fail=fail):
                        self.session.rank = 1 if sending else 0
                        self.session.world_size = 2
                        self.session._live_step = 8
                        self.session.groups = GroupRegistry()
                        self.session.channel = MagicMock()
                        group = object()
                        if registered:
                            self.session.groups.groups = [
                                GroupDescriptor((0, 1, 2), "hccl", "superset", object()),
                                GroupDescriptor((1, 0), "hccl", "recovery_metadata", group),
                                GroupDescriptor((0, 1), "hccl", "later", object()),
                            ]
                        info = {"repair_type": 1 if sending else 2,
                                "src": [1], "dst": [0], "rank_list": [0, 1]}
                        transfer = self.session.channel.send if sending else self.session.channel.receive
                        if fail:
                            transfer.side_effect = RuntimeError("transfer failed")
                        with (
                            patch(
                                "hyper_parallel.high_availability.session.dist.group",
                                SimpleNamespace(WORLD=object() if registered else group),
                            ),
                            patch.object(self.session.groups, "call_original_new_group") as create,
                            patch("hyper_parallel.high_availability.session.dist.destroy_process_group") as destroy,
                        ):
                            if fail:
                                with self.assertRaisesRegex(RuntimeError, "transfer failed"):
                                    self.session._repair(8, True, [0], info, [], None)
                            else:
                                self.session._repair(8, True, [0], info, [], None)
                            create.assert_not_called()
                            destroy.assert_not_called()
                        transfer.assert_called_once_with(0 if sending else 1, 8, group)

    def test_repair_does_not_reuse_superset_and_releases_failed_temporary_group(self):
        """Feature: Exact membership and temporary repair-group lifetime.

        Description: Fail a transfer with only a registered superset available.
        Expectation: Create an exact temporary group and release it despite the transfer error.
        """
        self.session.sdk.RepairType = SimpleNamespace(
            RT_SEND=SimpleNamespace(value=1),
            RT_RECV_REPAIR=SimpleNamespace(value=2),
        )
        self.session._live_step = 8
        self.session.rank = 1
        self.session.world_size = 4
        self.session.groups = GroupRegistry()
        self.session.groups.groups = [GroupDescriptor((0, 1, 2), "hccl", "superset", object())]
        group = object()
        info = {"repair_type": 1, "src": [1], "dst": [0], "rank_list": [0, 1]}
        self.session.channel.send.side_effect = RuntimeError("transfer failed")
        with patch.object(self.session.groups, "call_original_new_group", return_value=group) as create, \
                patch("hyper_parallel.high_availability.session.dist.destroy_process_group") as destroy:
            with self.assertRaisesRegex(RuntimeError, "transfer failed"):
                self.session._repair(8, True, [0], info, [], None)
            create.assert_called_once_with(ranks=[0, 1], backend="hccl", use_local_synchronization=True)
            destroy.assert_called_once_with(group)

    def test_muon_metadata_refresh_waits_until_global_rollback(self):
        """Feature: No shape collectives during local communication cleanup.

        Description: Rebuild groups, then run rollback after MindIO completes repair.
        Expectation: Optimizer runtime metadata refresh occurs only in rollback.
        """
        self.session.groups = MagicMock()
        self.session.model_state.rebuild_runtime_state = MagicMock()
        with self.session.update(1):
            pass
        self.session._rebuild([1], [], None)
        self.session.model_state.rebuild_runtime_state.assert_not_called()
        self.session._rollback(1, [], None)
        self.session.model_state.rebuild_runtime_state.assert_called_once_with()

    def test_reject_nonconsecutive_and_nested_updates(self):
        with self.assertRaisesRegex(RuntimeError, "consecutive"):
            with self.session.update(2):
                pass
        with self.session.update(1):
            with self.assertRaisesRegex(RuntimeError, "consecutive"):
                with self.session.update(1):
                    pass


class TestGroupRegistry(unittest.TestCase):
    """Recovery must preserve group handles, order and membership."""

    def test_replacement_identity_and_bootstrap_are_scoped_separately(self):
        """Feature: Recovery construction phase and replacement identity.

        Description: Seal a replacement registry and exit its context by exception.
        Expectation: Identity persists while bootstrap and context predicates reset.
        """
        registry = GroupRegistry()
        self.assertFalse(is_recovery_enabled())
        with self.assertRaisesRegex(RuntimeError, "construction failed"):
            with registry:
                self.assertTrue(is_recovery_enabled())
                self.assertFalse(is_recovery_reboot())
                self.assertFalse(is_recovery_bootstrapping())
                registry.reboot = True
                self.assertTrue(is_recovery_reboot())
                self.assertTrue(is_recovery_bootstrapping())
                registry.sealed = True
                self.assertTrue(is_recovery_reboot())
                self.assertFalse(is_recovery_bootstrapping())
                raise RuntimeError("construction failed")
        self.assertTrue(registry.reboot)
        self.assertFalse(is_recovery_enabled())
        self.assertFalse(is_recovery_reboot())
        self.assertFalse(is_recovery_bootstrapping())

    def test_muon_subgroups_rebuild_without_reboot_world_rendezvous(self):
        """Feature: Local Muon subgroup construction.

        Description: Build and rebuild optimizer subgroups using the recorded parent-group order.
        Expectation: Reuse local subgroup handles without collecting all world ranks.
        """
        registry = GroupRegistry()
        registry.reboot = True
        with patch("hyper_parallel.high_availability.groups.dist") as distributed, \
                patch("hyper_parallel.core.optimizer.optimizer.dist", distributed):
            distributed.get_backend.return_value = "hccl"
            distributed.get_world_size.return_value = 8
            distributed.group.WORLD = SimpleNamespace(bound_device_id=None)
            distributed.get_rank.return_value = 0
            distributed.new_group.side_effect = lambda **kwargs: object()
            with registry:
                parent = distributed.new_group(ranks=[0, 2, 4, 6], backend="hccl")
                distributed.new_group(ranks=[1, 3, 5, 7], backend="hccl")
                distributed.get_process_group_ranks.return_value = [0, 2, 4, 6]
                optimizer = object.__new__(Muon)
                optimizer._split_sub_pg_cache = {}
                subgroup = optimizer._get_or_create_sub_pg(parent, 2)
                self.assertIs(optimizer._get_or_create_sub_pg(parent, 2), subgroup)
            self.assertIs(optimizer._get_or_create_sub_pg(parent, 2), subgroup)
            self.assertEqual([descriptor.ranks for descriptor in registry.groups[2:]],
                             [(0, 2), (4, 6), (1, 3), (5, 7)])
            distributed.all_gather_into_tensor.assert_not_called()
            registry.rebuild()
            calls = distributed.reinit_process_group.call_args_list
            self.assertEqual(len(calls), 3)
            self.assertIs(calls[-1].kwargs["group"], subgroup)

    def test_member_groups_reinitialized_in_creation_order(self):
        """Feature: Ordered member-group reconstruction.

        Description: Record shared and distinct group handles before rebuilding twice.
        Expectation: Reinitialize only member handles in creation order, deduplicating identical objects.
        """
        registry = GroupRegistry()
        group_a, group_b = object(), object()
        with patch("hyper_parallel.high_availability.groups.dist") as distributed:
            distributed.get_backend.return_value = "hccl"
            distributed.get_world_size.return_value = 4
            distributed.group.WORLD = SimpleNamespace(bound_device_id=None)
            distributed.get_rank.return_value = 0
            distributed.new_group.side_effect = [group_a, group_b]
            with registry:
                self.assertIs(distributed.new_group(ranks=[0, 2]), group_a)
                self.assertIs(distributed.new_group(ranks=[1, 3]), group_b)
            registry.rebuild()
            self.assertEqual(distributed.reinit_process_group.call_count, 2)
            calls = distributed.reinit_process_group.call_args_list
            self.assertEqual(calls[0].kwargs, {"group": None, "rebuild_link": True})
            self.assertIs(calls[1].kwargs["group"], group_a)

    def test_registered_world_is_not_reinitialized_twice(self) -> None:
        """Feature: Deduplicate the default handle without merging distinct full-world groups.

        Description: Register WORLD alongside a distinct group with the same ranks and rebuild twice.
        Expectation: Each round handles WORLD once and the separate group once.
        """
        registry = GroupRegistry()
        other_group = object()
        with patch("hyper_parallel.high_availability.groups.dist") as distributed:
            distributed.group.WORLD = object()
            distributed.get_backend.return_value = "hccl"
            distributed.get_process_group_ranks.return_value = [0, 1]
            distributed.get_rank.return_value = 0
            registry.register(distributed.group.WORLD)
            registry.register(other_group)
            registry.register(distributed.group.WORLD)
            for _ in range(2):
                distributed.reinit_process_group.reset_mock()
                registry.rebuild()
                calls = distributed.reinit_process_group.call_args_list
                self.assertEqual(len(calls), 2)
                self.assertEqual(calls[0].kwargs, {"group": None, "rebuild_link": True})
                self.assertIs(calls[1].kwargs["group"], other_group)

    def test_rebuild_performs_local_cleanup_without_starting_collectives(self):
        """Feature: MindIO-coordinated communication rebuild.

        Description: Run local process-group reinitialization before acknowledging MindIO's rebuild action.
        Expectation: The callback initiates neither collectives nor a separate Store synchronization.
        """
        registry = GroupRegistry()
        events = []
        with patch("hyper_parallel.high_availability.groups.dist") as distributed:
            distributed.get_rank.return_value = 4
            distributed.get_world_size.return_value = 8
            distributed.reinit_process_group.side_effect = lambda **kwargs: events.append("reinit")
            registry.rebuild()
            self.assertEqual(events, ["reinit"])
            distributed.barrier.assert_not_called()

    def test_rebuild_repeats_local_cleanup_without_persistent_round_state(self):
        """Feature: Repeatable communication-group rebuild.

        Description: Rebuild the same registry twice and create an independent registry.
        Expectation: Both calls perform local cleanup without storing recovery-round counters.
        """
        registry = GroupRegistry()
        with patch("hyper_parallel.high_availability.groups.dist") as distributed:
            distributed.get_rank.return_value = 6
            distributed.get_world_size.return_value = 8
            registry.rebuild()
            registry.rebuild()
        self.assertEqual(distributed.reinit_process_group.call_count, 2)
        self.assertNotIn("generation", vars(registry))
        self.assertNotIn("generation", vars(GroupRegistry()))

    def test_normal_group_creation_keeps_original_torch_arguments(self):
        """Feature: Unmodified group creation outside recovery.

        Description: Call the public Torch factory without entering a recovery registry.
        Expectation: Preserve the original factory and forward all arguments unchanged.
        """
        with patch("hyper_parallel.high_availability.groups.dist.new_group") as create:
            GroupRegistry()
            self.assertIs(torch.distributed.new_group, create)
            torch.distributed.new_group(ranks=[0, 1], backend="gloo", use_local_synchronization=True)
            create.assert_called_once_with(ranks=[0, 1], backend="gloo", use_local_synchronization=True)

    def test_group_creation_rejects_inherited_device_binding(self):
        """Feature: Lazy recovery process-group construction.

        Description: Create a subgroup without device_id while WORLD already binds a device.
        Expectation: Original and replacement workers reject it before native group creation.
        """
        for reboot in (False, True):
            with self.subTest(reboot=reboot):
                registry = GroupRegistry()
                registry.reboot = reboot
                world = SimpleNamespace(bound_device_id=torch.device("cpu"))
                with patch("hyper_parallel.high_availability.groups.dist.group", SimpleNamespace(WORLD=world)), \
                        patch("hyper_parallel.high_availability.groups.dist.new_group") as native:
                    with registry:
                        with self.assertRaisesRegex(ValueError, "without bound device_id"):
                            torch.distributed.new_group(ranks=[0, 1], backend="hccl")
                    native.assert_not_called()
                self.assertEqual(registry.groups, [])

    def test_torch_group_patch_tracks_both_public_entries_and_restores_after_error(self):
        """Feature: Scoped process-wide communication group tracking.

        Description: Create groups through Torch and c10d with positional and keyword ranks, then fail construction.
        Expectation: Both handles are recorded in order and both native entry points are restored on exception.
        """
        registry = GroupRegistry()
        group_a, group_b = object(), object()
        c10d_original = torch.distributed.distributed_c10d.new_group
        with patch("hyper_parallel.high_availability.groups.dist.new_group",
                   side_effect=[group_a, group_b]) as native:
            with self.assertRaisesRegex(RuntimeError, "construction failed"):
                with registry:
                    self.assertIs(torch.distributed.new_group([0, 1], backend="hccl"), group_a)
                    self.assertIs(torch.distributed.distributed_c10d.new_group(ranks=[1, 2], backend="hccl"), group_b)
                    raise RuntimeError("construction failed")
            self.assertIs(torch.distributed.new_group, native)
            self.assertIs(torch.distributed.distributed_c10d.new_group, c10d_original)
            self.assertEqual(native.call_args_list[0].args, ([0, 1],))
        self.assertEqual([entry.ranks for entry in registry.groups], [(0, 1), (1, 2)])
        self.assertFalse(is_recovery_enabled())

    def test_torch_group_patch_covers_child_threads_and_rejects_late_groups(self):
        """Feature: Process-wide tracking during recovery construction and callbacks.

        Description: Create a group in a child thread, seal the catalog, then create an explicit temporary group.
        Expectation: Child-thread groups are tracked, late training groups fail and temporary groups bypass tracking.
        """
        registry = GroupRegistry()
        training_group, repair_group = object(), object()
        with patch("hyper_parallel.high_availability.groups.dist.new_group",
                   side_effect=[training_group, repair_group]) as native:
            with registry:
                self.assertIs(torch.distributed.new_group.__self__, registry)
                self.assertIs(torch.distributed.distributed_c10d.new_group.__self__, registry)
                with ThreadPoolExecutor(max_workers=1) as executor:
                    self.assertFalse(executor.submit(is_recovery_enabled).result())
                    with self.assertRaisesRegex(RuntimeError, "nested recovery group registries"):
                        executor.submit(GroupRegistry().__enter__).result()
                    created = executor.submit(torch.distributed.new_group, [0, 1], backend="hccl").result()
                self.assertIs(created, training_group)
                registry.sealed = True
                with self.assertRaisesRegex(RuntimeError, "before attach"):
                    torch.distributed.new_group([0, 1], backend="hccl")
                temporary = registry.call_original_new_group(
                    ranks=[0, 1], backend="hccl", use_local_synchronization=True)
                self.assertIs(temporary, repair_group)
                native.assert_called_with(ranks=[0, 1], backend="hccl", use_local_synchronization=True)
            self.assertIs(torch.distributed.new_group, native)
        self.assertEqual(len(registry.groups), 1)


class TestProcessorBootstrap(unittest.TestCase):
    """Rank and topology are runtime facts; users supply only the Controller endpoint."""

    def setUp(self) -> None:
        """Keep bookkeeping on CPU while exercising the mocked Ascend startup protocol."""
        patcher = patch("hyper_parallel.high_availability.session.torch",
                        SimpleNamespace(device=lambda *args: torch.device("cpu")))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_post_initialization_failures_release_processor(self):
        """Feature: Partial Processor startup cleanup.

        Description: Fail while starting the Processor or querying its replacement identity.
        Expectation: The initialized Processor is destroyed and the session flag is cleared.
        """
        for method in ("tft_start_processor", "tft_is_reboot_node"):
            with self.subTest(method=method):
                sdk, npu = MagicMock(), MagicMock()
                getattr(sdk, method).side_effect = RuntimeError("startup failed")
                modules = {"mindio_ttp.framework_ttp": sdk, "torch_npu": SimpleNamespace(npu=npu)}
                environment = {"RANK": "1", "LOCAL_RANK": "1", "WORLD_SIZE": "8"}
                with patch.dict("os.environ", environment, clear=True), \
                        patch("hyper_parallel.high_availability.session.dist.is_initialized", return_value=False), \
                        patch("hyper_parallel.high_availability.session.dist.reinit_process_group", create=True), \
                        patch("hyper_parallel.high_availability.session.importlib.import_module",
                              side_effect=modules.__getitem__):
                    session = RecoverySession(RecoveryConfig(True, "127.0.0.1", 18000))
                    with self.assertRaisesRegex(RuntimeError, "startup failed"):
                        session.initialize()
                    sdk.tft_destroy_processor.assert_called_once()
                    self.assertFalse(session._processor_initialized)
                    session.close()
                    sdk.tft_destroy_processor.assert_called_once()

    def test_rank_sources_for_original_and_replacement_workers(self):
        """Feature: Recovery bootstrap rank discovery.

        Description: Construct original and replacement sessions before and after group initialization.
        Expectation: Local device rank comes from the launcher and global identity uses the available source.
        """
        for initialized in (False, True):
            for reboot in (False, True):
                with self.subTest(initialized=initialized, reboot=reboot):
                    sdk, npu = MagicMock(), MagicMock()
                    sdk.tft_is_reboot_node.return_value = reboot
                    modules = {"mindio_ttp.framework_ttp": sdk, "torch_npu": SimpleNamespace(npu=npu)}
                    environment = {"RANK": "9", "LOCAL_RANK": "1", "WORLD_SIZE": "16"}
                    with patch.dict("os.environ", environment, clear=True), \
                            patch("hyper_parallel.high_availability.session.dist.is_initialized",
                                  return_value=initialized), \
                            patch("hyper_parallel.high_availability.session.dist.get_rank",
                                  return_value=13) as get_rank, \
                            patch("hyper_parallel.high_availability.session.dist.get_world_size",
                                  return_value=32) as get_world_size, \
                            patch("hyper_parallel.high_availability.session.dist.reinit_process_group",
                                  create=True), \
                            patch("hyper_parallel.high_availability.session.importlib.import_module",
                                  side_effect=modules.__getitem__):
                        session = RecoverySession(RecoveryConfig(True, "127.0.0.1", 18000))
                        session.initialize()
                        self.assertEqual(session.rank, 13 if initialized else 9)
                        self.assertEqual(session.world_size, 32 if initialized else 16)
                        self.assertEqual(session.local_rank, 1)
                        self.assertEqual(session.bank.rank, session.rank)
                        self.assertEqual(session.bank.device, torch.device("cpu"))
                        self.assertEqual(session.reboot, reboot)
                        npu.set_device.assert_called_once_with(1)
                        self.assertEqual(get_rank.call_count, int(initialized))
                        self.assertEqual(get_world_size.call_count, int(initialized))
                        session.close()

    def test_missing_local_rank_fails_before_initializing_runtime(self):
        """Feature: Launcher metadata validation.

        Description: Construct a session without LOCAL_RANK before or after group initialization.
        Expectation: Bootstrap fails before importing the SDK or selecting a default NPU.
        """
        for initialized in (False, True):
            with self.subTest(initialized=initialized), \
                    patch.dict("os.environ", {"RANK": "9", "WORLD_SIZE": "16"}, clear=True), \
                    patch("hyper_parallel.high_availability.session.dist.is_initialized",
                          return_value=initialized), \
                    patch("hyper_parallel.high_availability.session.dist.get_rank", return_value=9), \
                    patch("hyper_parallel.high_availability.session.dist.get_world_size", return_value=16), \
                    patch("hyper_parallel.high_availability.session.importlib.import_module") as import_module:
                with self.assertRaisesRegex(RuntimeError, "LOCAL_RANK"):
                    RecoverySession(RecoveryConfig(True, "127.0.0.1", 18000))
                import_module.assert_not_called()

    def test_taskd_controller_is_not_started_or_destroyed_by_rank_zero(self):
        """Feature: TaskD owns its Controller lifecycle.

        Description: Initialize and close rank zero with a TaskD-provided endpoint.
        Expectation: The worker neither creates nor destroys the hosted Controller.
        """
        sdk, npu = MagicMock(), MagicMock()
        sdk.tft_is_reboot_node.return_value = False
        modules = {"mindio_ttp.framework_ttp": sdk, "torch_npu": SimpleNamespace(npu=npu)}
        environment = {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "8", "MINDX_TASK_ID": "test-job",
                       "MASTER_ADDR": "192.168.1.2", "TTP_PORT": "18001", "POD_IP": "192.168.1.3"}
        with patch.dict("os.environ", environment, clear=True), \
                patch("hyper_parallel.high_availability.session.dist.is_initialized", return_value=False), \
                patch("hyper_parallel.high_availability.session.dist.reinit_process_group", create=True), \
                patch("hyper_parallel.high_availability.session.importlib.import_module",
                      side_effect=modules.__getitem__):
            session = RecoverySession(RecoveryConfig(True))
            session.initialize()
            sdk.tft_start_processor.assert_called_once_with("192.168.1.2", 18001)
            sdk.tft_init_controller.assert_not_called()
            session.close()
            sdk.tft_destroy_controller.assert_not_called()

    def test_taskd_endpoint_requires_valid_deployment_metadata(self):
        session = RecoverySession.__new__(RecoverySession)
        session.config = RecoveryConfig(True)
        for environment in ({}, {"MASTER_ADDR": "host", "TTP_PORT": "abc"},
                            {"MASTER_ADDR": "host", "TTP_PORT": "65536"}):
            with self.subTest(environment=environment), patch.dict("os.environ", environment, clear=True):
                with self.assertRaisesRegex(ValueError, "TaskD"):
                    session._controller_endpoint()

    def test_rank_zero_starts_controller_and_all_ranks_start_processor(self):
        """Feature: Explicit endpoints use rank-zero Controller hosting.

        Description: Initialize original workers at rank zero and a nonzero rank.
        Expectation: Every worker starts its Processor and only rank zero creates the Controller.
        """
        for rank in (0, 3):
            with self.subTest(rank=rank):
                sdk, npu = MagicMock(), MagicMock()
                sdk.tft_is_reboot_node.return_value = False
                modules = {"mindio_ttp.framework_ttp": sdk, "torch_npu": SimpleNamespace(npu=npu)}
                environment = {"RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": "8"}
                with patch.dict("os.environ", environment), \
                        patch("hyper_parallel.high_availability.session.dist.is_initialized", return_value=False), \
                        patch("hyper_parallel.high_availability.session.dist.reinit_process_group", create=True), \
                        patch("hyper_parallel.high_availability.session.importlib.import_module",
                              side_effect=modules.__getitem__):
                    session = RecoverySession(RecoveryConfig(True, "127.0.0.1", 18000))
                    session.initialize()
                    sdk.tft_init_processor.assert_called_once_with(rank, 8, False, False, "", False, True, False)
                    sdk.tft_start_processor.assert_called_once_with("127.0.0.1", 18000)
                    if rank == 0:
                        sdk.tft_init_controller.assert_called_once_with(0, 8, False, True, False)
                        sdk.tft_start_controller.assert_called_once_with("127.0.0.1", 18000, False, "")
                    else:
                        sdk.tft_init_controller.assert_not_called()
                    session.close()
