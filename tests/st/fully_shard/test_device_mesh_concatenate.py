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
"""torchrun entry for DeviceMesh.concatenate STs."""

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

_TEST_DEVICE_MESH_CONCATENATE = "_test_device_mesh_concatenate.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_device_mesh_concatenate_group1():
    """
    Feature: Pytorch DeviceMesh.concatenate with root and flattened dims.
    Description:
        1.test_device_mesh_concatenate_supports_root_and_flattened_dims
        2.test_device_mesh_concatenate_rejects_out_of_root_order
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_DEVICE_MESH_CONCATENATE, "test_device_mesh_concatenate_supports_root_and_flattened_dims", 12365, 4),
        TorchCase(_TEST_DEVICE_MESH_CONCATENATE, "test_device_mesh_concatenate_rejects_out_of_root_order", 12366, 4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_device_mesh_concatenate_group1_gloo():
    """
    Feature: Pytorch DeviceMesh.concatenate with root and flattened dims.
    Description:
        1.test_device_mesh_concatenate_supports_root_and_flattened_dims
        2.test_device_mesh_concatenate_rejects_out_of_root_order
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_DEVICE_MESH_CONCATENATE, "test_device_mesh_concatenate_supports_root_and_flattened_dims", num_proc=4),
        TorchCase(_TEST_DEVICE_MESH_CONCATENATE, "test_device_mesh_concatenate_rejects_out_of_root_order", num_proc=4),
    ])
