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
"""launch _test_cp_optim_state_dict.py cases for CP+FSDP optim state dict."""
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

_TEST_CP_SD = "_test_cp_optim_state_dict.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_cp_optim_state_dict_group1():
    """
    Feature: get/set_optim_state_dict for CP+FSDP
    Description:
        1.test_c1_cp_fsdp_optim_state_dict_fqn_roundtrip
        2.test_c2_cp_fsdp_optim_state_dict_full_cpu_broadcast
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_CP_SD, "test_c1_cp_fsdp_optim_state_dict_fqn_roundtrip", 13710, 4),
        TorchCase(_TEST_CP_SD, "test_c2_cp_fsdp_optim_state_dict_full_cpu_broadcast", 13711, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_cp_optim_state_dict_group2():
    """
    Feature: get/set_optim_state_dict for CP+FSDP
    Description:
        1.test_c3_cp_fsdp_optim_state_dict_flatten
        2.test_c4_cp_fsdp_local_shape_correctness
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_CP_SD, "test_c3_cp_fsdp_optim_state_dict_flatten", 13712, 4),
        TorchCase(_TEST_CP_SD, "test_c4_cp_fsdp_local_shape_correctness", 13713, 4),
    ])
