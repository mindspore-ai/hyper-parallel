# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""launch _test_state_dict.py cases for fully_shard."""
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

_TEST_STATE_DICT = "_test_state_dict.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_state_dict_group1():
    """
    Feature: parallel run case in fully_shard
    Description:
        1.test_t6_get_model_sd_sharded
        2.test_t7_get_model_sd_full_cpu
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_STATE_DICT, "test_t6_get_model_sd_sharded", 12376, 4),
        TorchCase(_TEST_STATE_DICT, "test_t7_get_model_sd_full_cpu", 12377, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_state_dict_group2():
    """
    Feature: parallel run case in fully_shard
    Description:
        1.test_t8_get_model_sd_ignore_frozen
        2.test_t9_get_model_sd_sharded_cpu
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_STATE_DICT, "test_t8_get_model_sd_ignore_frozen", 12378, 4),
        TorchCase(_TEST_STATE_DICT, "test_t9_get_model_sd_sharded_cpu", 12379, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_state_dict_group3():
    """
    Feature: parallel run case in fully_shard
    Description:
        1.test_t11_meta_load_backward
        2.test_t10_to_dtype_if_needed (CPU-only unit test, no device required)
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_STATE_DICT, "test_t11_meta_load_backward", 12381, 4),
        TorchCase(_TEST_STATE_DICT, "test_t10_to_dtype_if_needed", 12402, 1),
    ])
