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
"""test device mesh"""
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

DEVICE_MESH = "device_mesh.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_device_mesh_group1():
    """
    Feature: parallel run case in process_group
    Description:
        1.test_device_mesh_from_1d_group_valid
        2.test_device_mesh_from_2d_group_use_list_valid
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(DEVICE_MESH, "test_device_mesh_from_1d_group_valid", 10124, 2),
        TorchCase(DEVICE_MESH, "test_device_mesh_from_2d_group_use_list_valid", 10127, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_device_mesh_group2():
    """
    Feature: parallel run case in process_group
    Description:
        1.test_device_mesh_from_2d_group_valid
        2.test_device_mesh_from_3d_group_valid
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(DEVICE_MESH, "test_device_mesh_from_2d_group_valid", 10125, 4),
        TorchCase(DEVICE_MESH, "test_device_mesh_from_3d_group_valid", 10126, 4),
    ])
