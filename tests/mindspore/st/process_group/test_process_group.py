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
"""test process group"""
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase

DEVICE_MESH = "device_mesh.py"
PROCESS_GROUP = "process_group.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_process_group_group1():
    """
    Feature: parallel run case in process_group
    Description:
        1.test_device_mesh_from_1d_group_valid
        2.test_device_mesh_slice_invalid_without_mesh_dim_names
        3.test_device_mesh_get_group_invalid_without_init_backend
        4.test_device_mesh_invalid_different_mesh_dim_names
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(DEVICE_MESH, "test_device_mesh_from_1d_group_valid", 10114, 2, 2),
        MindSporeCase(DEVICE_MESH, "test_device_mesh_slice_invalid_without_mesh_dim_names", 10117, 2, 2),
        MindSporeCase(DEVICE_MESH, "test_device_mesh_get_group_invalid_without_init_backend", 10118, 2, 2),
        MindSporeCase(DEVICE_MESH, "test_device_mesh_invalid_different_mesh_dim_names", 10119, 2, 2)
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_process_group_group2():
    """
    Feature: parallel run case in process_group
    Description:
        1.test_device_mesh_from_3d_group_valid
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(DEVICE_MESH, "test_device_mesh_from_3d_group_valid", 10116, 8, 8),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_process_group_group3():
    """
    Feature: parallel run case in process_group
    Description:
        1.test_device_mesh_from_2d_group_valid
        2.test_process_group
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(DEVICE_MESH, "test_device_mesh_from_2d_group_valid", 10115, 4, 4),
        MindSporeCase(PROCESS_GROUP, "test_process_group", 10111, 4, 4)
    ])
