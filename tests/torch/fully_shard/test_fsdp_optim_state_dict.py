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
"""launch _test_fsdp_optim_state_dict.py cases for pure FSDP (1-D mesh)."""
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

_TEST_FSDP_SD = "_test_fsdp_optim_state_dict.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_fsdp_optim_state_dict_group1():
    """
    Feature: get/set_optim_state_dict for pure FSDP (1-D mesh)
    Description:
        1.test_f1_fsdp_optim_state_dict_fqn_roundtrip
        2.test_f2_fsdp_optim_state_dict_full_cpu
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_FSDP_SD, "test_f1_fsdp_optim_state_dict_fqn_roundtrip", 13410, 4),
        TorchCase(_TEST_FSDP_SD, "test_f2_fsdp_optim_state_dict_full_cpu", 13411, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_fsdp_optim_state_dict_group2():
    """
    Feature: get/set_optim_state_dict for pure FSDP (1-D mesh)
    Description:
        1.test_f3_fsdp_optim_state_dict_broadcast_from_rank0
        2.test_f4_fsdp_optim_state_dict_flatten
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_FSDP_SD, "test_f3_fsdp_optim_state_dict_broadcast_from_rank0", 13412, 4),
        TorchCase(_TEST_FSDP_SD, "test_f4_fsdp_optim_state_dict_flatten", 13413, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_fsdp_optim_state_dict_group3():
    """
    Feature: get/set_optim_state_dict for pure FSDP (1-D mesh)
    Description:
        1.test_f5_fsdp_optim_state_dict_strict_false
        2.test_f8_fsdp_local_shape_correctness
        3.test_f9_fsdp_full_cpu_restore_to_device
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_FSDP_SD, "test_f5_fsdp_optim_state_dict_strict_false", 13414, 4),
        TorchCase(_TEST_FSDP_SD, "test_f8_fsdp_local_shape_correctness", 13415, 4),
        TorchCase(_TEST_FSDP_SD, "test_f9_fsdp_full_cpu_restore_to_device", 13416, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_fsdp_optim_state_dict_group4():
    """
    Feature: DCP save/load with optim state dict for pure FSDP
    Description:
        1.test_f6_fsdp_dcp_save_load_nested
        2.test_f7_fsdp_dcp_load_new_optimizer
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_FSDP_SD, "test_f6_fsdp_dcp_save_load_nested", 13417, 4),
        TorchCase(_TEST_FSDP_SD, "test_f7_fsdp_dcp_load_new_optimizer", 13418, 4),
    ])
