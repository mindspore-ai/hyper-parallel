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
"""System tests for DFunction distributed dispatch."""
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

DIST_FUNCTION = "_dfunction.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_dfunction_group1():
    """
    Feature: DFunction distributed dispatch
    Description:
        1. test_dtensor_forward_output_is_dtensor
        2. test_dtensor_forward_numerical_correctness
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(DIST_FUNCTION, "test_dtensor_forward_output_is_dtensor", 13001, 4),
        TorchCase(DIST_FUNCTION, "test_dtensor_forward_numerical_correctness", 13002, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_dfunction_group2():
    """
    Feature: DFunction layout cache and backward
    Description:
        1. test_layout_cache_hit
        2. test_dtensor_backward_gradient_correctness
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(DIST_FUNCTION, "test_layout_cache_hit", 13003, 4),
        TorchCase(DIST_FUNCTION, "test_dtensor_backward_gradient_correctness", 13004, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_dfunction_group3():
    """
    Feature: DFunction single-card vs multi-card numerical parity
    Description:
        1. test_single_vs_multi_card_forward_parity
        2. test_single_vs_multi_card_backward_parity
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(DIST_FUNCTION, "test_single_vs_multi_card_forward_parity", 13006, 4),
        TorchCase(DIST_FUNCTION, "test_single_vs_multi_card_backward_parity", 13007, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_dfunction_group4():
    """
    Feature: DFunction with custom ReLU and column-parallel Linear
    Description:
        1. test_relu_forward_backward
        2. test_linear_colwise_dispatch_new
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(DIST_FUNCTION, "test_relu_forward_backward", 13008, 4),
        TorchCase(DIST_FUNCTION, "test_linear_colwise_dispatch_new", 13009, 4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_dfunction_group4_gloo():
    """
    Feature: DFunction with custom ReLU and column-parallel Linear
    Description:
        1. test_relu_forward_backward
        2. test_linear_colwise_dispatch_new
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(DIST_FUNCTION, "test_relu_forward_backward", num_proc=4),
        TorchCase(DIST_FUNCTION, "test_linear_colwise_dispatch_new", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_dfunction_group5():
    """
    Feature: DFunction row-parallel Linear with get_expand_impl
    Description:
        1. test_linear_rowwise_get_expand_impl
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(DIST_FUNCTION, "test_linear_rowwise_get_expand_impl", 13010, 4),
    ])
