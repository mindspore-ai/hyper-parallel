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
"""Unit tests for batched pipeline P2P process-group initialization."""
import unittest
from types import SimpleNamespace
from unittest import mock

from hyper_parallel.core.pipeline_parallel import scheduler as scheduler_module
from hyper_parallel.core.pipeline_parallel.scheduler import ScheduleInterleaved1F1B


class TestBatchP2PGroupInitialization(unittest.TestCase):
    """Verify batch transports initialize and use their PP process group."""

    @staticmethod
    def _make_schedule(batch_p2p: bool) -> ScheduleInterleaved1F1B:
        """Build the minimal schedule state needed by initialization tests."""
        schedule = object.__new__(ScheduleInterleaved1F1B)
        schedule._batch_p2p = batch_p2p
        schedule._batch_p2p_group = mock.sentinel.pp_group
        schedule._batch_p2p_group_initialized = False
        schedule.stages = [SimpleNamespace(stage_index=0, pp_group=mock.sentinel.pp_group)]
        schedule.real_stage_num = 2
        schedule.exec_order = {0: []}
        schedule.micro_batch_num = 1
        schedule._send_handles = []
        schedule._boundary_issued = set()
        schedule._pending_boundary = {}
        schedule.sync_shared_parameters_grad = mock.Mock()
        return schedule

    def test_batch_transport_prepares_group_once(self) -> None:
        """Two runs of one batch schedule prepare its process group only once."""
        schedule = self._make_schedule(batch_p2p=True)
        with mock.patch.object(scheduler_module.platform, "prepare_batch_p2p_group") as prepare_group:
            schedule.run_microbatches([], [], [])
            schedule.run_microbatches([], [], [])

        prepare_group.assert_called_once_with(mock.sentinel.pp_group)
        self.assertTrue(schedule._batch_p2p_group_initialized)

    def test_plain_transport_does_not_prepare_group(self) -> None:
        """Plain P2P preserves its existing no-preparation startup path."""
        schedule = self._make_schedule(batch_p2p=False)
        with mock.patch.object(scheduler_module.platform, "prepare_batch_p2p_group") as prepare_group:
            schedule.run_microbatches([], [], [])

        prepare_group.assert_not_called()
        self.assertFalse(schedule._batch_p2p_group_initialized)

    def test_single_rank_batch_transport_does_not_prepare_group(self) -> None:
        """A batch schedule without a cross-rank edge needs no preparation."""
        schedule = self._make_schedule(batch_p2p=True)
        schedule.real_stage_num = 1
        with mock.patch.object(scheduler_module.platform, "prepare_batch_p2p_group") as prepare_group:
            schedule.run_microbatches([], [], [])

        prepare_group.assert_not_called()
        self.assertFalse(schedule._batch_p2p_group_initialized)

    def test_batch_transport_passes_pipeline_group_to_p2p_ops(self) -> None:
        """Batched operations use the same PP group prepared before launch."""
        schedule = self._make_schedule(batch_p2p=True)
        specs = [("isend", mock.sentinel.tensor, 1)]
        with mock.patch.object(
                scheduler_module.platform, "p2p_op", return_value=mock.sentinel.op) as p2p_op:
            with mock.patch.object(
                    scheduler_module.platform, "batch_isend_irecv",
                    return_value=mock.sentinel.handle) as batch_isend_irecv:
                handles = schedule._batched_issue(specs)

        p2p_op.assert_called_once_with(
            "isend", mock.sentinel.tensor, 1, group=mock.sentinel.pp_group)
        batch_isend_irecv.assert_called_once_with([mock.sentinel.op])
        self.assertEqual(handles, [mock.sentinel.handle])

    def test_batch_transport_rejects_different_pipeline_groups(self) -> None:
        """Local virtual stages cannot use distinct communicators in batch mode."""
        schedule = self._make_schedule(batch_p2p=True)
        schedule.stages.append(SimpleNamespace(stage_index=2, pp_group=mock.sentinel.other_group))

        with self.assertRaisesRegex(ValueError, "stages 0 and 2 use different groups"):
            schedule._resolve_batch_p2p_group()

    def test_failed_preparation_does_not_mark_group_initialized(self) -> None:
        """A failed preparation is retried instead of being treated as initialized."""
        schedule = self._make_schedule(batch_p2p=True)
        with mock.patch.object(
                scheduler_module.platform, "prepare_batch_p2p_group",
                side_effect=RuntimeError("preparation failed")):
            with self.assertRaisesRegex(RuntimeError, "preparation failed"):
                schedule.run_microbatches([], [], [])

        self.assertFalse(schedule._batch_p2p_group_initialized)
