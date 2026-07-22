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
"""Launcher for single-card optimizer state dict unit tests."""

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

UT_FILE = "_test_optim_state_dict_ut.py"


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_optim_state_dict_ut_group1():
    """
    Feature: optim state dict single-card UT
    Description:
        1.test_u1_fqn_keys_no_integer_ids
        2.test_u2_param_groups_initial_lr_empty_state
        3.test_u3_adamw_roundtrip
        4.test_u4_sgd_roundtrip
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(UT_FILE, "test_u1_fqn_keys_no_integer_ids", num_proc=1),
        TorchCase(UT_FILE, "test_u2_param_groups_initial_lr_empty_state", num_proc=1),
        TorchCase(UT_FILE, "test_u3_adamw_roundtrip", num_proc=1),
        TorchCase(UT_FILE, "test_u4_sgd_roundtrip", num_proc=1),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_optim_state_dict_ut_group2():
    """
    Feature: optim state dict single-card UT
    Description:
        1.test_u5_strict_false
        2.test_u6_flatten_roundtrip_dotted_fqn
        3.test_u7_empty_param_group_flatten_error
        4.test_u8_chained_optimizer_rejection
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(UT_FILE, "test_u5_strict_false", num_proc=1),
        TorchCase(UT_FILE, "test_u6_flatten_roundtrip_dotted_fqn", num_proc=1),
        TorchCase(UT_FILE, "test_u7_empty_param_group_flatten_error", num_proc=1),
        TorchCase(UT_FILE, "test_u8_chained_optimizer_rejection", num_proc=1),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_optim_state_dict_ut_group3():
    """
    Feature: optim state dict single-card UT
    Description:
        1.test_u9_unflatten_strict_true_inconsistent_fields
        2.test_u10_unflatten_strict_false_inconsistent_fields
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(UT_FILE, "test_u9_unflatten_strict_true_inconsistent_fields", num_proc=1),
        TorchCase(UT_FILE, "test_u10_unflatten_strict_false_inconsistent_fields", num_proc=1),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_optim_state_dict_ut_group4():
    """
    Feature: optim state dict single-card UT
    Description:
        1.test_u11_set_restores_param_groups_fields
        2.test_u12_strict_true_missing_fqns
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(UT_FILE, "test_u11_set_restores_param_groups_fields", num_proc=1),
        TorchCase(UT_FILE, "test_u12_strict_true_missing_fqns", num_proc=1),
    ])
