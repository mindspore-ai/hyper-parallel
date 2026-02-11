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

from tests.torch.utils import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level3", card_mark="allcards", essential_mark="essential")
def test_dcp_api_with_dtensor_and_scalar():
    """
    Feature: Test checkpoint save and load API with DTensor state_dict.
    Description: Test save and load function with state_dict containing DTensors on 8-card setup.
    Expectation: Run success.
    """
    master_port = 12252
    file_name = "dcp_api.py"
    case_name = "test_dcp_api_with_dtensor_and_scalar"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_dcp_api_with_dtensor_and_scalar_with_more_params():
    """
    Feature: Test checkpoint save and load API with DTensor state_dict using different mesh_shape and layouts.
    Description: Test save and load function with state_dict containing DTensors on 8-card setup with mesh_shape (4, 2).
    Expectation: Run success.
    """
    master_port = 12253
    file_name = "dcp_api.py"
    case_name = "test_dcp_api_with_dtensor_and_scalar_with_more_params"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level3", card_mark="allcards", essential_mark="essential")
def test_dcp_api_save_8card_load_4card():
    """
    Feature: Test checkpoint save with 8-card cluster and load with 4-card cluster.
    Description: Test save function with state_dict containing DTensors and scalars on 8-card setup,
                 then load with 4-card setup. This tests resharding capability.
    Expectation: Run success.
    """
    master_port = 12254
    file_name = "dcp_api.py"
    case_name = "test_dcp_api_save_8card_load_4card"
    torchrun_case(file_name, case_name, master_port)
