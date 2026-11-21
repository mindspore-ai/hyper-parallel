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
from unittest.mock import call, patch

import pytest

from tests.common import parallel_case
from tests.common.parallel_case import TorchCase, parallel_run_ascend_batches


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
