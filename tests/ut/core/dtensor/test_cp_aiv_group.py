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
"""Unit tests for assigning the CP communication group to AIV."""

import sys
from types import ModuleType
import unittest
from unittest.mock import MagicMock, patch

from hyper_parallel.core.dtensor import _utils
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh, _DEVICE_MESH_MAP
from hyper_parallel.core.utils.communication import EXISTING_COMM_GROUPS


class TestCpAivProcessGroup(unittest.TestCase):
    """Verify CP-specific HCCL options during process-group creation."""

    def setUp(self) -> None:
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def tearDown(self) -> None:
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def test_hccl_options_select_aiv_for_cp(self) -> None:
        """The CP group requests HCCL expansion mode 3 (AIV)."""
        fake_distributed = ModuleType("torch_npu._C._distributed_c10d")

        class FakeOptions:
            """Minimal ProcessGroupHCCL options fake."""

            def __init__(self) -> None:
                self.hccl_config = {}

        class FakeProcessGroupHCCL:
            """Minimal ProcessGroupHCCL fake."""

            Options = FakeOptions

        fake_distributed.ProcessGroupHCCL = FakeProcessGroupHCCL
        fake_c_extension = ModuleType("torch_npu._C")
        fake_c_extension.__path__ = []
        fake_c_extension._distributed_c10d = fake_distributed
        fake_torch_npu = ModuleType("torch_npu")
        fake_torch_npu.__path__ = []
        fake_torch_npu._C = fake_c_extension

        fake_modules = {
            "torch_npu": fake_torch_npu,
            "torch_npu._C": fake_c_extension,
            "torch_npu._C._distributed_c10d": fake_distributed,
        }
        with patch.dict(sys.modules, fake_modules):
            options = _utils.get_cp_hccl_process_group_options()

        self.assertEqual(options.hccl_config, {"hccl_op_expansion_mode": 3})

    @patch("hyper_parallel.core.dtensor.device_mesh.dist")
    @patch("hyper_parallel.core.dtensor.device_mesh._utils")
    def test_device_mesh_applies_aiv_options_only_to_cp_axis(
            self,
            mock_utils: MagicMock,
            mock_dist: MagicMock,
    ) -> None:
        """Only the named CP axis receives AIV process-group options."""
        cp_options = MagicMock(name="cp_options")
        mock_dist.get_rank.return_value = 0
        mock_utils.get_cp_hccl_process_group_options.return_value = cp_options
        mock_utils.split_group.return_value = MagicMock(name="group")

        DeviceMesh(
            "npu",
            [[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
            mesh_dim_names=("dp", "cp", "tp"),
        )

        mock_utils.get_cp_hccl_process_group_options.assert_called_once_with()
        calls_with_options = [
            split_call
            for split_call in mock_utils.split_group.call_args_list
            if "pg_options" in split_call.kwargs
        ]
        self.assertEqual(len(calls_with_options), 1)
        self.assertIs(calls_with_options[0].kwargs["pg_options"], cp_options)


if __name__ == "__main__":
    unittest.main()
