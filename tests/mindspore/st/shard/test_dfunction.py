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
"""System tests for DFunction distributed dispatch (MindSpore)."""
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase

DIST_FUNCTION = "_dfunction.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_dfunction_ms_group1():
    """
    Feature: DFunction DTensor forward output and numerical correctness (MindSpore)
    Description:
        1. test_dtensor_forward_output_is_dtensor
        2. test_dtensor_forward_numerical_correctness
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(DIST_FUNCTION, "test_dtensor_forward_output_is_dtensor", 19001, 4, 4),
        MindSporeCase(DIST_FUNCTION, "test_dtensor_forward_numerical_correctness", 19002, 4, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_dfunction_ms_group2():
    """
    Feature: DFunction single-card vs multi-card forward parity and ReLU (MindSpore)
    Description:
        1. test_single_vs_multi_card_forward_parity
        2. test_relu_forward_correctness
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(DIST_FUNCTION, "test_single_vs_multi_card_forward_parity", 19003, 4, 4),
        MindSporeCase(DIST_FUNCTION, "test_relu_forward_correctness", 19004, 4, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_dfunction_ms_group3():
    """
    Feature: DFunction column-parallel and row-parallel Linear (MindSpore)
    Description:
        1. test_linear_colwise_dispatch_new
        2. test_linear_rowwise_get_expand_impl
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(DIST_FUNCTION, "test_linear_colwise_dispatch_new", 19005, 4, 4),
        MindSporeCase(DIST_FUNCTION, "test_linear_rowwise_get_expand_impl", 19006, 4, 4),
    ])
