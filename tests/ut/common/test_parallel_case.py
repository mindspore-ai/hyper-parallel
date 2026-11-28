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

"""Check Ascend ST launch batching without creating worker processes."""
import os
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

import psutil
import pytest

from tests.common import distributed_launcher, parallel_case
from tests.common.ascend_level0_cases import (
    run_pp_batch_cold_start_and_hsdp,
    run_pp_multi_stream_cold_start_and_composites,
    run_reentrant_exclude_suite,
)
from tests.common.parallel_case import TorchCase, _runtime_ascend_count, parallel_run_ascend_batches


class TestParallelRunAscendBatches(unittest.TestCase):
    """Protect the minimum device budget and the assignment order."""

    @patch.object(parallel_case, "parallel_run")
    def test_split_cases_when_only_four_devices_are_visible(self, run):
        """Do not hand out devices outside the runner's enabled Ascend IDs."""
        cases = [TorchCase("worker.py", "test_first", num_proc=4), TorchCase("worker.py", "test_second", num_proc=4)]
        devices = (2, 3, 6, 7)
        with patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "2,3,6,7"}, clear=True):
            parallel_run_ascend_batches(cases)
        self.assertEqual(run.call_args_list, [
            call([cases[0]], global_num_proc=4, device_ids=devices),
            call([cases[1]], global_num_proc=4, device_ids=devices),
        ])

    @patch.object(Path, "glob")
    @patch.object(parallel_case, "parallel_run")
    def test_use_logical_ids_when_container_device_names_use_physical_ids(self, run, glob):
        """Use zero-based CANN IDs when the device mask is unset."""
        glob.return_value = [Path(f"/dev/davinci{index}") for index in range(4, 8)]
        cases = [TorchCase("worker.py", "test_first", num_proc=2), TorchCase("worker.py", "test_second", num_proc=2)]
        with patch.dict(os.environ, {}, clear=True):
            parallel_run_ascend_batches(cases)
        run.assert_called_once_with(cases, global_num_proc=4, device_ids=(0, 1, 2, 3))

    @patch.object(parallel_case, "parallel_run")
    def test_insufficient_budget_does_not_report_partial_coverage(self, run):
        """A required four-rank case cannot be replaced by a two-rank case."""
        cases = [TorchCase("worker.py", "test_four", num_proc=4), TorchCase("worker.py", "test_two", num_proc=2)]
        with patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3", "HYPER_PARALLEL_TEST_CARD_COUNT": "2"},
                        clear=True):
            with self.assertRaisesRegex(pytest.skip.Exception, "Requires 4 Ascend devices"):
                parallel_run_ascend_batches(cases)
            run.assert_not_called()

    @patch.object(parallel_case, "parallel_run")
    @patch.object(Path, "exists", return_value=True)
    @patch.object(Path, "glob", return_value=[])
    @patch.object(parallel_case.subprocess, "run")
    def test_manager_only_container_uses_runtime_count(self, probe, glob, exists, run) -> None:
        """A container with eight logical NPUs can expose only its driver management node."""
        _runtime_ascend_count.cache_clear()
        self.addCleanup(_runtime_ascend_count.cache_clear)
        probe.return_value = Mock(returncode=0, stdout="ASCEND_DEVICE_COUNT=8\ndriver log\n")
        cases = [TorchCase("worker.py", "test_four", num_proc=4)]
        with patch.dict(os.environ, {"HYPER_PARALLEL_TEST_CARD_COUNT": "4"}, clear=True):
            parallel_run_ascend_batches(cases)
        with patch.dict(os.environ, {}, clear=True):
            parallel_run_ascend_batches(cases)
        self.assertEqual(run.call_args_list, [
            call(cases, global_num_proc=4, device_ids=tuple(range(4))),
            call(cases, global_num_proc=8, device_ids=tuple(range(8))),
        ])
        glob.assert_called_with("davinci[0-9]*")
        exists.assert_called()
        probe.assert_called_once()
        self.assertEqual(probe.call_args.kwargs["timeout"], 30)
        _runtime_ascend_count.cache_clear()
        probe.return_value = Mock(returncode=1, stdout="", stderr="driver unavailable")
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(RuntimeError, "driver unavailable"):
            parallel_run_ascend_batches(cases)

    @patch.object(parallel_case, "parallel_run")
    def test_activation_runs_every_group_that_fits(self, run) -> None:
        """One, two, and four-rank items report their own outcome on smaller runners."""
        workers_dir = Path("workers") / "activation_memory"
        expected = {
            1: [
                "test_reentrant_checkpoint_exclude_swap",
                "test_checkpoint_exclude_save_chain_swap_level0",
                "test_rmsnorm_matmul_checkpoint_exclude_level0",
                "test_adjacent_save_regions_execute_once_and_replay_cached_outputs",
                "test_memory_lifecycle",
                ("test_operator_numerics", "test_rng_autocast"),
            ],
            2: ["test_reentrant_swap_stage_split_npu", "test_reentrant_pp2", "test_fsdp_checkpoint_exclude_level0"],
            4: ["test_reentrant_pp4"],
        }
        for budget in (1, 2, 3, 4, 8):
            visible = ",".join(str(device) for device in range(budget))
            mask = {"ASCEND_RT_VISIBLE_DEVICES": visible}
            with self.subTest(budget=budget), patch.dict(os.environ, mask, clear=True):
                for ranks, selectors in expected.items():
                    run.reset_mock()
                    if ranks > budget:
                        with self.assertRaisesRegex(pytest.skip.Exception, f"Requires {ranks} Ascend devices"):
                            run_reentrant_exclude_suite(workers_dir, ranks)
                        run.assert_not_called()
                        continue
                    run_reentrant_exclude_suite(workers_dir, ranks)
                    workers = [case for batch in run.call_args_list for case in batch.args[0]]
                    self.assertEqual([case.case_name for case in workers], selectors)
                    self.assertTrue(all(case.num_proc == ranks for case in workers))


class TestPipelineColdStart(unittest.TestCase):
    """Each transport starts on its own WORLD and enforces the distributed deadline."""

    @patch.object(distributed_launcher, "allocate_port", return_value=12340)
    @patch.object(distributed_launcher.psutil, "Popen")
    def test_first_selector_and_timeout_cleanup(self, popen, allocate) -> None:
        """A hung cold start fails its launch and terminates all tracked descendants."""
        worker_dir = Path("workers") / "pipeline_parallel"
        for launcher, transport, followups in (
            (run_pp_batch_cold_start_and_hsdp, "batch", (
                "_test_pp_composite.py::test_pp_hsdp_vpp_interleaved_1f1b_matches_reference",
                "_test_pp_composite.py::test_pp_hsdp_vpp_dxdw_split_matches_reference",
            )),
            (run_pp_multi_stream_cold_start_and_composites, "multi_stream", (
                "reentrant_checkpoint_worker.py::test_reentrant_pp8",
                "_test_pp_fsdp_tp_composite.py::test_pp_fsdp_tp_1f1b_composite_matches_reference",
            )),
        ):
            with self.subTest(transport=transport):
                popen.reset_mock()
                allocate.reset_mock()
                rank = Mock(pid=9001)
                job = popen.return_value
                job.children.return_value = [rank]
                job.wait.side_effect = psutil.TimeoutExpired(420)
                cold = f"test_vpp_dynamic_{transport}_p2p_cold_start"
                with patch.object(distributed_launcher.time, "monotonic", return_value=10.0):
                    with patch.object(psutil, "wait_procs", side_effect=[([], [rank]), ([], [rank])]) as wait:
                        with self.assertRaisesRegex(AssertionError, f"{cold}.*remaining PIDs: \\[9001\\]"):
                            launcher(worker_dir)
                        self.assertEqual(wait.call_args_list, [
                            call([job, rank], timeout=5), call([rank], timeout=5),
                        ])
                allocate.assert_called_once_with()
                popen.assert_called_once()
                cmd = popen.call_args.args[0]
                self.assertIn("--nproc-per-node=8", cmd)
                nodes = [arg for arg in cmd if "::" in arg]
                expected_nodes = (f"vpp_schedule.py::{cold}", *followups)
                self.assertEqual(nodes, [str(worker_dir / selector) for selector in expected_nodes])
                timeout = job.wait.call_args_list[0].kwargs["timeout"]
                self.assertEqual(timeout, 420)
                job.children.assert_called_once_with(recursive=True)
                job.terminate.assert_called_once_with()
                rank.terminate.assert_called_once_with()
                job.kill.assert_not_called()
                rank.kill.assert_called_once_with()
