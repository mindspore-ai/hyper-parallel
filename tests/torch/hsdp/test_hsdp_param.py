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
"""test hybrid shard data parallel"""
from tests.torch.utils import torchrun_case
from tests.common.mark_utils import arg_mark

def test_hsdp_param_1d_mesh_without_dtensor():
    """
    Feature: HSDPParam.
    Description: Test HSDPParam with 1D-mesh and Parameter is not DTensor
    Expectation: assertion pass.
    """
    master_port = 12342
    file_name = "_test_hsdp_param.py"
    case_name = "test_hsdp_param_1d_mesh_without_dtensor"
    torchrun_case(file_name, case_name, master_port)

def test_hsdp_param_2d_mesh_without_dtensor():
    """
    Feature: HSDPParam.
    Description: Test HSDPParam with 2D-mesh and Parameter is not DTensor
    Expectation: assertion pass.
    """
    master_port = 12342
    file_name = "_test_hsdp_param.py"
    case_name = "test_hsdp_param_2d_mesh_without_dtensor"
    torchrun_case(file_name, case_name, master_port)