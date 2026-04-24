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
"""Unit tests for MindSpore runtime patches installed by hyper_parallel."""

import os
import unittest

import pytest

pytest.importorskip("mindspore")

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

import mindspore as ms
from mindspore import Tensor
from mindspore._c_expression import ParamInfo

import hyper_parallel  # pylint: disable=unused-import


class TestMindSporePatches(unittest.TestCase):
    """Verify import-time MindSpore monkey patches."""

    def test_parameter_param_info_patch_breaks_back_reference(self):
        """Importing hyper_parallel should keep ``ParamInfo.obj`` cleared."""
        param = ms.Parameter(Tensor([1.0], ms.float32), name="weight")
        param_info = ParamInfo()

        param.param_info = param_info

        self.assertIsNone(param_info.obj)
        self.assertIs(param.param_info, param_info)


if __name__ == "__main__":
    unittest.main()
