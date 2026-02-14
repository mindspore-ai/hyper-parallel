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
"""test_process_group.py"""
from tests.common.mark_utils import arg_mark
from tests.mindspore.st.utils import msrun_case


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_device_mesh_from_1d_group_valid():
    """
    Feature: Test case test_device_mesh_from_1d_group_valid in device_mesh.py.
    Description: Initial environments for mindspore, and execute case test_device_mesh_from_1d_group_valid in
        device_mesh.py.
    Expectation: Run success.
    """
    glog_v = 3
    file_name = "device_mesh.py"
    case_name = "test_device_mesh_from_1d_group_valid"
    master_port = 10114
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_device_mesh_from_2d_group_valid():
    """
    Feature: Test case test_device_mesh_from_2d_group_valid in device_mesh.py.
    Description: Initial environments for mindspore, and execute case test_device_mesh_from_2d_group_valid in
        device_mesh.py.
    Expectation: Run success.
    """
    glog_v = 3
    file_name = "device_mesh.py"
    case_name = "test_device_mesh_from_2d_group_valid"
    master_port = 10115
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_device_mesh_from_3d_group_valid():
    """
    Feature: Test case test_device_mesh_from_3d_group_valid in device_mesh.py.
    Description: Initial environments for mindspore, and execute case test_device_mesh_from_3d_group_valid in
        device_mesh.py.
    Expectation: Run success.
    """
    glog_v = 3
    file_name = "device_mesh.py"
    case_name = "test_device_mesh_from_3d_group_valid"
    master_port = 10116
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_device_mesh_slice_invalid_without_mesh_dim_names():
    """
    Feature: Test case test_device_mesh_slice_invalid_without_mesh_dim_names in device_mesh.py.
    Description: Initial environments for mindspore, and execute case
        test_device_mesh_slice_invalid_without_mesh_dim_names in device_mesh.py.
    Expectation: Run success.
    """
    glog_v = 3
    file_name = "device_mesh.py"
    case_name = "test_device_mesh_slice_invalid_without_mesh_dim_names"
    master_port = 10117
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_device_mesh_get_group_invalid_without_init_backend():
    """
    Feature: Test case test_device_mesh_get_group_invalid_without_init_backend in device_mesh.py.
    Description: Initial environments for mindspore, and execute case
        test_device_mesh_get_group_invalid_without_init_backend in device_mesh.py.
    Expectation: Run success.
    """
    glog_v = 3
    file_name = "device_mesh.py"
    case_name = "test_device_mesh_get_group_invalid_without_init_backend"
    master_port = 10118
    msrun_case(glog_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_device_mesh_invalid_different_mesh_dim_names():
    """
    Feature: Test case test_device_mesh_invalid_different_mesh_dim_names in device_mesh.py.
    Description: Initial environments for mindspore, and execute case
        test_device_mesh_invalid_different_mesh_dim_names in device_mesh.py.
    Expectation: Run success.
    """
    glog_v = 3
    file_name = "device_mesh.py"
    case_name = "test_device_mesh_invalid_different_mesh_dim_names"
    master_port = 10119
    msrun_case(glog_v, file_name, case_name, master_port)
