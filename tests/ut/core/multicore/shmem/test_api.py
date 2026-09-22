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
"""Unit tests for the private SHMEM Python capability boundary."""

import io
import unittest
from contextlib import redirect_stderr
from types import SimpleNamespace
from unittest.mock import Mock, patch

from hyper_parallel.core.multicore.shmem import _api, _debug


class _Tensor:
    """Minimal Tensor-shaped value used by Allocation diagnostics."""

    dtype = "torch.uint8"
    shape = (8,)

    @staticmethod
    def data_ptr() -> int:
        """Return a stable fake Allocation address."""
        return 0x10000

    @staticmethod
    def element_size() -> int:
        """Return the fake dtype size in Byte."""
        return 1

    @staticmethod
    def numel() -> int:
        """Return the fake Tensor element count."""
        return 8


class TestShmemApiDiagnostics(unittest.TestCase):
    """Validate that opt-in diagnostics remain useful and best-effort."""

    def test_debug_state_reports_reference_count_after_runtime_state(self) -> None:
        """Expose the local lifecycle count next to the Native Runtime state."""
        native = SimpleNamespace(
            _debug_state=Mock(return_value={"state": "Ready", "root_rank": 0})
        )
        with (
            patch.object(_debug, "_load_native", return_value=native),
            patch.object(_debug, "_reference_count", return_value=2),
        ):
            state = _debug.debug_state()

        self.assertEqual(state["reference_count"], 2)
        self.assertIsInstance(state["reference_count"], int)
        self.assertEqual(list(state)[:2], ["state", "reference_count"])
        self.assertEqual(repr(state).splitlines()[:2], ["state: Ready", "reference_count: 2"])

    def test_allocation_log_records_only_direct_callsite(self) -> None:
        """Render the first SHMEM-external caller without a complete stack chain."""
        output = io.StringIO()
        with patch.dict("os.environ", {"HYPER_PARALLEL_SHMEM_LOG_LEVEL": "0"}), \
             patch.object(_debug, "_site_rank", return_value=3), redirect_stderr(output):
            _debug._log_allocation_site(_Tensor())  # pylint: disable=protected-access

        message = output.getvalue()
        self.assertIn("[rank 3][DEBUG]", message)
        self.assertIn("allocation_base=0x10000", message)
        self.assertIn("bytes=8", message)
        self.assertIn("test_allocation_log_records_only_direct_callsite", message)
        self.assertNotIn(" <- ", message)

    def test_allocation_log_failure_does_not_change_empty_result(self) -> None:
        """Return a successful Allocation even when optional stack extraction fails."""
        tensor = _Tensor()
        torch = SimpleNamespace(get_default_dtype=Mock(return_value="torch.float32"))
        native = SimpleNamespace(_empty=Mock(return_value=tensor))
        with patch.dict("os.environ", {"HYPER_PARALLEL_SHMEM_LOG_LEVEL": "0"}), \
             patch.object(_api, "_torch_modules", return_value=(torch, object())), \
             patch.object(_api, "_load_native", return_value=native), \
             patch.object(_debug.traceback, "extract_stack", side_effect=RuntimeError("diagnostic failure")):
            result = _api.empty(8, dtype="torch.uint8")

        self.assertIs(result, tensor)
        native._empty.assert_called_once_with((8,), "torch.uint8", None)


if __name__ == "__main__":
    unittest.main()
