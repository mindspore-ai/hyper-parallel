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
"""UT for :mod:`hyper_parallel.platform.mindspore.parameter_param_info_patch`."""
import os
import unittest

import pytest

pytest.importorskip("mindspore")

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"
from tests.ut.platform.mindspore._ensure_mindspore_platform import (  # noqa: E402
    ensure_mindspore_platform_default,
)

ensure_mindspore_platform_default()

from hyper_parallel.platform.mindspore.parameter_param_info_patch import (  # noqa: E402
    patch_mindspore_parameter_param_info_cycle_if_needed,
)

patch_mindspore_parameter_param_info_cycle_if_needed()

import mindspore as ms
from mindspore import Tensor
from mindspore._c_expression import ParamInfo

import hyper_parallel  # noqa: F401 pylint: disable=unused-import


class TestParameterParamInfoPatch(unittest.TestCase):
    """Verify import-time MindSpore monkey patches for ``ParamInfo`` cycles."""

    def test_parameter_param_info_patch_breaks_back_reference(self):
        """``Parameter.param_info`` assignment should clear ``ParamInfo.obj`` when patch is active."""
        param = ms.Parameter(Tensor([1.0], ms.float32), name="weight")
        param_info = ParamInfo()

        param.param_info = param_info

        self.assertIsNone(param_info.obj)
        self.assertIs(param.param_info, param_info)


if __name__ == "__main__":
    unittest.main()
