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
"""test checkpoint dcp API"""

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

DCP_API = "dcp_api.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_dcp_api_group1():
    """
    Feature: parallel run case in checkpoint
    Description:
        1.test_dcp_api_with_dtensor_and_tensor_and_scalar
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(DCP_API, "test_dcp_api_with_dtensor_and_tensor_and_scalar", 12253, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_dcp_api_group2():
    """
    Feature: parallel run case in checkpoint
    Description:
        1.test_dcp_api_with_full_tensor
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(DCP_API, "test_dcp_api_with_full_tensor", 12255, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level3", card_mark="allcards", essential_mark="essential")
def test_dcp_api_group3():
    """
    Feature: parallel run case in checkpoint
    Description:
        1.test_dcp_api_save_8card_load_4card
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(DCP_API, "test_dcp_api_save_8card_load_4card", 12254, 4),
    ])
