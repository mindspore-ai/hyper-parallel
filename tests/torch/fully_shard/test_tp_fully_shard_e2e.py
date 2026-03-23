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
"""torchrun entry for TP + fully_shard end-to-end tests"""

from tests.common.mark_utils import arg_mark
from tests.torch.utils import torchrun_case


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_tp_plus_fully_shard_loss_and_grad_match_standalone():
    """
    Feature: fully_shard with TP-sharded DTensor parameters.
    Description: Run one end-to-end TP + FSDP training case and compare loss/grad with standalone.
    Expectation: Distributed loss and local shard gradients match standalone slices.
    """
    master_port = 12364
    file_name = "_test_tp_fully_shard_e2e.py"
    case_name = "test_tp_plus_fully_shard_loss_and_grad_match_standalone"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_tp_plus_fully_shard_rectangular_inputs_match_standalone():
    """
    Feature: fully_shard with TP-sharded DTensor parameters.
    Description: Run a rectangular-shape TP + FSDP end-to-end training case.
    Expectation: Distributed loss and local shard gradients match standalone slices.
    """
    master_port = 12365
    file_name = "_test_tp_fully_shard_e2e.py"
    case_name = "test_tp_plus_fully_shard_rectangular_inputs_match_standalone"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_tp_plus_fully_shard_wide_output_match_standalone():
    """
    Feature: fully_shard with TP-sharded DTensor parameters.
    Description: Run a wide-output TP + FSDP end-to-end training case.
    Expectation: Distributed loss and local shard gradients match standalone slices.
    """
    master_port = 12366
    file_name = "_test_tp_fully_shard_e2e.py"
    case_name = "test_tp_plus_fully_shard_wide_output_match_standalone"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_tp4_plus_fully_shard_loss_and_grad_match_standalone():
    """
    Feature: fully_shard with TP-sharded DTensor parameters.
    Description: Run a tp_size=4 TP + FSDP end-to-end training case when mesh size permits.
    Expectation: Supported environments match standalone loss and gradient slices.
    """
    master_port = 12367
    file_name = "_test_tp_fully_shard_e2e.py"
    case_name = "test_tp4_plus_fully_shard_loss_and_grad_match_standalone"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_tp_plus_fully_shard_on_3d_root_mesh_match_standalone():
    """
    Feature: fully_shard with TP-sharded DTensor parameters on a 3D root mesh.
    Description: Run a dp x tp x ep = 2 x 2 x 2 end-to-end training case.
    Expectation: Distributed loss and local shard gradients match standalone slices.
    """
    master_port = 12368
    file_name = "_test_tp_fully_shard_e2e.py"
    case_name = "test_tp_plus_fully_shard_on_3d_root_mesh_match_standalone"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_tp_plus_fully_shard_rectangular_3d_root_mesh_match_standalone():
    """
    Feature: fully_shard with TP-sharded DTensor parameters on a 3D root mesh.
    Description: Run a rectangular-shape dp x tp x ep = 2 x 2 x 2 end-to-end training case.
    Expectation: Distributed loss and local shard gradients match standalone slices.
    """
    master_port = 12369
    file_name = "_test_tp_fully_shard_e2e.py"
    case_name = "test_tp_plus_fully_shard_rectangular_3d_root_mesh_match_standalone"
    torchrun_case(file_name, case_name, master_port)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_hsdp_plus_tp_on_3d_root_mesh_match_standalone():
    """
    Feature: fully_shard with 2D HSDP mesh and 1D TP mesh.
    Description: Run a dp x fsdp x tp = 2 x 2 x 2 end-to-end training case.
    Expectation: Distributed loss and local shard gradients match standalone slices.
    """
    master_port = 12370
    file_name = "_test_tp_fully_shard_e2e.py"
    case_name = "test_hsdp_plus_tp_on_3d_root_mesh_match_standalone"
    torchrun_case(file_name, case_name, master_port)
