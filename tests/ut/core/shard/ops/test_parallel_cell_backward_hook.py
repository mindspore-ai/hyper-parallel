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
"""Unit tests for CellBackwardHook distributed op."""

import os
import unittest
from unittest.mock import MagicMock, patch

import pytest

# pylint: disable=wrong-import-position

pytest.importorskip("mindspore")

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

from tests.ut.platform.mindspore._ensure_mindspore_platform import (  # noqa: E402
    ensure_mindspore_platform_for_shard_and_dtensor,
)

ensure_mindspore_platform_for_shard_and_dtensor()

from hyper_parallel.core.shard.ops.parallel_cell_backward_hook import (  # noqa: E402
    CellBackwardHookDistributedOp,
)
from hyper_parallel.core.dtensor.dtensor import DTensor  # noqa: E402


class TestCellBackwardHookDistributedOp(unittest.TestCase):
    """Unit tests for CellBackwardHookDistributedOp.wrap_output."""

    def setUp(self):
        self.op = CellBackwardHookDistributedOp("CellBackwardHook")

    def test_wrap_output_preserves_local_tensor_slots(self):
        """
        Feature: CellBackwardHook output wrapping
        Description: Mixed outputs contain one DTensor-backed slot and one plain local tensor slot
        Expectation: The DTensor-backed slot is wrapped, while the local tensor is passed through unchanged
        """
        output_layout = MagicMock()
        output_layout.mesh = "mesh"
        output_layout.alias_placements = ("shard",)
        py_output = ("distributed_output", "local_mask")

        with patch.object(DTensor, "from_local", return_value="wrapped_output") as mock_from_local:
            wrapped = self.op.wrap_output(py_output, (output_layout, None))

        self.assertEqual(wrapped, ("wrapped_output", "local_mask"))
        mock_from_local.assert_called_once_with(
            "distributed_output", output_layout.mesh, output_layout.alias_placements
        )

    def test_wrap_output_scalar_local_slot_passthrough(self):
        """
        Feature: CellBackwardHook output wrapping
        Description: A single local tensor output is associated with a None layout
        Expectation: The output is returned unchanged and DTensor.from_local is not called
        """
        py_output = "local_only_output"

        with patch.object(DTensor, "from_local") as mock_from_local:
            wrapped = self.op.wrap_output(py_output, (None,))

        self.assertEqual(wrapped, "local_only_output")
        mock_from_local.assert_not_called()


if __name__ == "__main__":
    unittest.main()
