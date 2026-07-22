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
"""launch _test_optim_state_dict.py cases for fully_shard."""
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

_TEST_OPTIM_SD = "_test_optim_state_dict.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_optim_state_dict_group1():
    """
    Feature: get/set_optim_state_dict for HSDP
    Description:
        1.test_o1_optim_state_dict_fqn_roundtrip
        2.test_o2_optim_state_dict_full_cpu
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_OPTIM_SD, "test_o1_optim_state_dict_fqn_roundtrip", 12410, 4),
        TorchCase(_TEST_OPTIM_SD, "test_o2_optim_state_dict_full_cpu", 12411, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_optim_state_dict_group2():
    """
    Feature: get/set_optim_state_dict for HSDP
    Description:
        1.test_o3_optim_state_dict_broadcast_from_rank0
        2.test_o4_optim_state_dict_flatten
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_OPTIM_SD, "test_o3_optim_state_dict_broadcast_from_rank0", 12412, 4),
        TorchCase(_TEST_OPTIM_SD, "test_o4_optim_state_dict_flatten", 12413, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_optim_state_dict_group3():
    """
    Feature: get/set_optim_state_dict for HSDP
    Description:
        1.test_o5_optim_state_dict_strict_false
        2.test_o9_hsdp_local_shape_correctness
        3.test_o10_full_cpu_restore_to_device
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_OPTIM_SD, "test_o5_optim_state_dict_strict_false", 12414, 4),
        TorchCase(_TEST_OPTIM_SD, "test_o9_hsdp_local_shape_correctness", 12415, 4),
        TorchCase(_TEST_OPTIM_SD, "test_o10_full_cpu_restore_to_device", 12416, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_optim_state_dict_group4():
    """
    Feature: DCP save/load with optim state dict
    Description:
        1.test_o6_dcp_save_load_nested
        2.test_o7_dcp_save_load_flatten
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_OPTIM_SD, "test_o6_dcp_save_load_nested", 12417, 4),
        TorchCase(_TEST_OPTIM_SD, "test_o7_dcp_save_load_flatten", 12418, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_optim_state_dict_group5():
    """
    Feature: DCP load into new optimizer (empty state regression)
    Description:
        1.test_o8_dcp_load_new_optimizer
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_OPTIM_SD, "test_o8_dcp_load_new_optimizer", 12419, 4),
    ])
