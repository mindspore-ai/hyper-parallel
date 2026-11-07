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

import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import torch

from hyper_parallel.core.fault_tolerance.config import RecoveryConfig
from hyper_parallel.core.fault_tolerance.groups import GroupRegistry, new_group
from hyper_parallel.core.fault_tolerance.session import RecoverySession
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

    def test_snapshot_mirror_ack_then_end_and_publish(self):
        events = []
        self.session.sdk.tft_start_updating_os.side_effect = lambda *_: events.append("start")
        self.session.channel.mirror_metadata.side_effect = lambda *_: events.append("metadata_ack")
        self.session.sdk.tft_end_updating_os.side_effect = lambda step: events.append(("end", step, self.session.step))
        with self.session.update(1):
            self.cursor.value = 1
            events.append("optimizer_and_scheduler")
        self.assertEqual(events, ["start", "optimizer_and_scheduler", ("end", 1, 0), "metadata_ack"])
        self.assertEqual(self.session.step, 1)
        metadata = decode(self.session.bank.get(1, 1)["payload"].numpy().tobytes())
        self.assertEqual(metadata["state"]["loader"]["value"], 1)

    def test_failed_update_never_reports_end(self):
        with self.assertRaisesRegex(RuntimeError, "optimizer failed"):
            with self.session.update(1):
                raise RuntimeError("optimizer failed")
        self.session.sdk.tft_end_updating_os.assert_not_called()
        self.session.channel.mirror_metadata.assert_not_called()
        self.assertEqual(self.session.step, 0)
        self.assertFalse(self.session._in_update)

    def test_processors_disconnect_before_sdk_controller_cleanup(self):
        events = []
        session = self.session
        session._running = False
        session._initialized = True
        session._owns_controller = False
        session.reboot = False
        session.groups = GroupRegistry()

        def _decorate(function):
            def _wrapper(*args):
                result = function(*args)
                events.append("sdk_controller_cleanup")
                return result
            return _wrapper

        session.sdk.tft_exception_handler.side_effect = _decorate
        session.sdk.tft_destroy_processor.side_effect = lambda: events.append("disconnect")
        with patch("hyper_parallel.core.fault_tolerance.session.dist.barrier",
                   side_effect=lambda: events.append("barrier")):
            self.assertEqual(session.run(lambda active: 12), 12)
        self.assertLess(events.index("disconnect"), len(events) - 2)
        self.assertEqual(events[-2:], ["barrier", "sdk_controller_cleanup"])
        session.sdk.tft_destroy_processor.assert_called_once()

    def test_metadata_failure_keeps_completed_device_step_without_claiming_cpu_rollback(self):
        self.session.channel.mirror_metadata.side_effect = RuntimeError("metadata unavailable")
        with self.assertRaisesRegex(RuntimeError, "metadata unavailable"):
            with self.session.update(1):
                pass
        self.session.sdk.tft_end_updating_os.assert_called_once_with(1)
        self.assertEqual(self.session._live_step, 1)

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

    def test_normal_group_creation_keeps_original_torch_arguments(self):
        with patch("hyper_parallel.core.fault_tolerance.groups.dist.new_group") as create:
            new_group(ranks=[0, 1], backend="gloo", use_local_synchronization=True)
            create.assert_called_once_with(ranks=[0, 1], backend="gloo", use_local_synchronization=True)


class TestProcessorBootstrap(unittest.TestCase):
    """Rank and topology are runtime facts; users supply only the Controller endpoint."""

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
