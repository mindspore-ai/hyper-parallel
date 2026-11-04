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
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import torch

from hyper_parallel import DeviceMesh, Shard, distribute_tensor
from hyper_parallel.core.fault_tolerance.config import RecoveryConfig
from hyper_parallel.core.fault_tolerance.groups import (
    GroupRegistry, is_recovery_bootstrapping, is_recovery_enabled, is_recovery_reboot, new_group,
)
from hyper_parallel.core.fault_tolerance.session import MetadataMirrorError, RecoverySession
from hyper_parallel.core.fault_tolerance.state import ModelOptimizerState, TrainState
from hyper_parallel.core.fault_tolerance.transport import DeviceMetadataBank, decode
from hyper_parallel.core.optimizer.adamw import AdamW
from hyper_parallel.core.optimizer.muon import Muon
from tests.ut.fault_tolerance.test_state import StatefulValue


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
        self.model = torch.nn.Linear(2, 2)
        self.session.model_state = ModelOptimizerState(self.model, AdamW(self.model.parameters()))
        self.cursor = StatefulValue()
        self.session.train_state = TrainState(self.cursor, {})
        self.session.bank = DeviceMetadataBank(1, torch.device("cpu"))
        self.session._live_step = 0

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

        def _body(active):
            del active
            events.append("native_destroy")
            barrier.side_effect = AssertionError("training process group is gone")
            return 12

        with patch("hyper_parallel.core.fault_tolerance.session.dist.barrier") as barrier:
            self.assertEqual(session.run(_body), 12)
        barrier.assert_not_called()
        self.assertEqual(events, ["native_destroy", "controller", "processor"])
        session.sdk.tft_destroy_controller.assert_called_once()
        session.sdk.tft_destroy_processor.assert_called_once()
        self.assertFalse(session._processor_initialized)
        self.assertFalse(session._controller_initialized)

    def _prepare_run(self, controller_initialized: bool = False) -> RecoverySession:
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
        with patch("hyper_parallel.core.fault_tolerance.session.dist.barrier"):
            with self.assertRaises(RuntimeError) as caught:
                session.run(body)
        self.assertIs(caught.exception, error)
        self.assertEqual(completed, [12])
        body.assert_called_once_with(session)
        session.sdk.tft_report_error.assert_not_called()

    def test_initial_state_is_published_after_native_checkpoint_loading(self):
        """Feature: Initial metadata follows the native training-start callbacks.

        Description: Load a nonzero initial step in the body and publish its state twice.
        Expectation: The loaded step is captured/reported once and later initialization does not overwrite it.
        """
        session = self._prepare_run()

        def _body(active):
            active.channel.mirror_metadata.assert_not_called()
            self.cursor.value = 8
            active.publish_initial_state(8)
            active.publish_initial_state(9)
            self.assertEqual(active.step, 8)
            metadata = active.bank.metadata(active.rank, 8)
            self.assertEqual(metadata["state"]["rank_state"]["loader"]["value"], 8)
            return 12

        with patch("hyper_parallel.core.fault_tolerance.session.dist.barrier"):
            self.assertEqual(session.run(_body), 12)
        session.channel.mirror_metadata.assert_called_once_with(8)
        session.sdk.tft_report_load_ckpt_step.assert_called_once_with(8)

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
        with patch("hyper_parallel.core.fault_tolerance.session.dist.barrier"):
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
        with patch("hyper_parallel.core.fault_tolerance.session.dist.barrier"):
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
        with patch("hyper_parallel.core.fault_tolerance.session.dist.barrier"):
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
                patch("hyper_parallel.core.fault_tolerance.session.dist.barrier"):
            os.environ.pop("MINDX_TASK_ID", None)
            with self.assertRaisesRegex(ValueError, "controller cleanup failed") as caught:
                session.run(lambda active: 12)
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
                patch("hyper_parallel.core.fault_tolerance.session.dist.barrier"):
            self.assertEqual(session.run(lambda active: 12), 12)
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
                active.groups.generation = 1
                return function(active)
            return _retry

        def _body(active):
            self.assertEqual(active.step, 8)
            active.publish_initial_state(active.step)
            active.channel.mirror_metadata.assert_not_called()
            return distribute_tensor(tensor, mesh, [Shard(0)], src_data_rank=0)

        session.sdk.tft_exception_handler.side_effect = _decorate
        with patch("hyper_parallel.core.fault_tolerance.session.dist.barrier"), \
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
        with patch("hyper_parallel.core.fault_tolerance.session.dist.barrier"):
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
        for sending in (True, False):
            with self.subTest(sending=sending):
                self.session.rank = 5 if sending else 1
                self.session.channel.reset_mock()
                group = object()
                info = {"repair_type": 1 if sending else 2, "src": [5], "dst": [1], "rank_list": [1, 5]}
                with patch("hyper_parallel.core.fault_tolerance.session.dist.new_group",
                           return_value=group) as create, \
                        patch("hyper_parallel.core.fault_tolerance.session.dist.destroy_process_group") as destroy:
                    self.session._repair(8, True, [1], info, [], None)
                create.assert_called_once_with(ranks=[1, 5], backend="hccl", use_local_synchronization=True)
                destroy.assert_called_once_with(group)
                transfer = self.session.channel.send if sending else self.session.channel.receive
                transfer.assert_called_once_with(1 if sending else 5, 8, group)

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
        registry = GroupRegistry()
        registry.reboot = True
        with patch("hyper_parallel.core.fault_tolerance.groups.dist") as distributed, \
                patch("hyper_parallel.core.optimizer.optimizer.dist", distributed):
            distributed.get_backend.return_value = "hccl"
            distributed.get_world_size.return_value = 8
            distributed.get_rank.return_value = 0
            distributed.new_group.side_effect = lambda **kwargs: object()
            with registry:
                parent = new_group(ranks=[0, 2, 4, 6], backend="hccl")
                new_group(ranks=[1, 3, 5, 7], backend="hccl")
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
        registry = GroupRegistry()
        group_a, group_b = object(), object()
        with patch("hyper_parallel.core.fault_tolerance.groups.dist") as distributed:
            distributed.get_backend.return_value = "hccl"
            distributed.get_world_size.return_value = 4
            distributed.get_rank.return_value = 0
            distributed.new_group.side_effect = [group_a, group_b]
            with registry:
                self.assertIs(new_group(ranks=[0, 2]), group_a)
                self.assertIs(new_group(ranks=[1, 3]), group_b)
            registry.rebuild()
            self.assertEqual(distributed.reinit_process_group.call_count, 2)
            calls = distributed.reinit_process_group.call_args_list
            self.assertEqual(calls[0].kwargs, {"group": None, "rebuild_link": True})
            self.assertIs(calls[1].kwargs["group"], group_a)
            self.assertEqual(registry.generation, 1)

    def test_rebuild_performs_local_cleanup_without_starting_collectives(self):
        """Feature: MindIO-coordinated communication rebuild.

        Description: Run local process-group reinitialization before acknowledging MindIO's rebuild action.
        Expectation: The callback initiates neither collectives nor a separate Store synchronization.
        """
        registry = GroupRegistry()
        events = []
        with patch("hyper_parallel.core.fault_tolerance.groups.dist") as distributed:
            distributed.get_rank.return_value = 4
            distributed.get_world_size.return_value = 8
            distributed.reinit_process_group.side_effect = lambda **kwargs: events.append("reinit")
            registry.rebuild()
            self.assertEqual(events, ["reinit"])
            distributed.barrier.assert_not_called()

    def test_rebuild_count_is_process_local_diagnostic_state(self):
        """Feature: Local diagnostic rebuild count.

        Description: Rebuild twice in one process, then construct a replacement registry.
        Expectation: The count advances locally and does not require persistence across process replacement.
        """
        registry = GroupRegistry()
        with patch("hyper_parallel.core.fault_tolerance.groups.dist") as distributed:
            distributed.get_rank.return_value = 6
            distributed.get_world_size.return_value = 8
            registry.rebuild()
            registry.rebuild()
        self.assertEqual(registry.generation, 2)
        self.assertEqual(GroupRegistry().generation, 0)

    def test_normal_group_creation_keeps_original_torch_arguments(self):
        with patch("hyper_parallel.core.fault_tolerance.groups.dist.new_group") as create:
            new_group(ranks=[0, 1], backend="gloo", use_local_synchronization=True)
            create.assert_called_once_with(ranks=[0, 1], backend="gloo", use_local_synchronization=True)


class TestProcessorBootstrap(unittest.TestCase):
    """Rank and topology are runtime facts; users supply only the Controller endpoint."""

    def setUp(self) -> None:
        """Keep bookkeeping on CPU while exercising the mocked Ascend startup protocol."""
        patcher = patch("hyper_parallel.core.fault_tolerance.session.torch",
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
                        patch("hyper_parallel.core.fault_tolerance.session.dist.is_initialized", return_value=False), \
                        patch("hyper_parallel.core.fault_tolerance.session.dist.reinit_process_group", create=True), \
                        patch("hyper_parallel.core.fault_tolerance.session.importlib.import_module",
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
                            patch("hyper_parallel.core.fault_tolerance.session.dist.is_initialized",
                                  return_value=initialized), \
                            patch("hyper_parallel.core.fault_tolerance.session.dist.get_rank",
                                  return_value=13) as get_rank, \
                            patch("hyper_parallel.core.fault_tolerance.session.dist.get_world_size",
                                  return_value=32) as get_world_size, \
                            patch("hyper_parallel.core.fault_tolerance.session.dist.reinit_process_group",
                                  create=True), \
                            patch("hyper_parallel.core.fault_tolerance.session.importlib.import_module",
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
                    patch("hyper_parallel.core.fault_tolerance.session.dist.is_initialized",
                          return_value=initialized), \
                    patch("hyper_parallel.core.fault_tolerance.session.dist.get_rank", return_value=9), \
                    patch("hyper_parallel.core.fault_tolerance.session.dist.get_world_size", return_value=16), \
                    patch("hyper_parallel.core.fault_tolerance.session.importlib.import_module") as import_module:
                with self.assertRaisesRegex(RuntimeError, "LOCAL_RANK"):
                    RecoverySession(RecoveryConfig(True, "127.0.0.1", 18000))
                import_module.assert_not_called()

    def test_taskd_controller_is_not_started_or_destroyed_by_rank_zero(self):
        sdk, npu = MagicMock(), MagicMock()
        sdk.tft_is_reboot_node.return_value = False
        modules = {"mindio_ttp.framework_ttp": sdk, "torch_npu": SimpleNamespace(npu=npu)}
        environment = {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "8", "MINDX_TASK_ID": "test-job",
                       "MASTER_ADDR": "192.168.1.2", "TTP_PORT": "18001", "POD_IP": "192.168.1.3"}
        with patch.dict("os.environ", environment, clear=True), \
                patch("hyper_parallel.core.fault_tolerance.session.dist.is_initialized", return_value=False), \
                patch("hyper_parallel.core.fault_tolerance.session.dist.reinit_process_group", create=True), \
                patch("hyper_parallel.core.fault_tolerance.session.importlib.import_module",
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
        for rank in (0, 3):
            with self.subTest(rank=rank):
                sdk, npu = MagicMock(), MagicMock()
                sdk.tft_is_reboot_node.return_value = False
                modules = {"mindio_ttp.framework_ttp": sdk, "torch_npu": SimpleNamespace(npu=npu)}
                environment = {"RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": "8"}
                with patch.dict("os.environ", environment), \
                        patch("hyper_parallel.core.fault_tolerance.session.dist.is_initialized", return_value=False), \
                        patch("hyper_parallel.core.fault_tolerance.session.dist.reinit_process_group", create=True), \
                        patch("hyper_parallel.core.fault_tolerance.session.importlib.import_module",
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
