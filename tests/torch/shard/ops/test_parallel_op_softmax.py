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
"""Test runner for softmax distributed ST (PyTorch)."""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

IMPL_FILE = str(Path(__file__).resolve().parent / "_test_parallel_op_softmax.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_softmax_group1():
    """
    Feature: parallel run case in _test_parallel_op_softmax
    Description:
        1. test_softmax_hybrid_parallel —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_softmax_hybrid_parallel", num_proc=8),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_softmax_group1_gloo():
    """
    Feature: parallel run case in _test_parallel_op_softmax
    Description:
        1. test_softmax_hybrid_parallel —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_softmax_hybrid_parallel", num_proc=8),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_softmax_group2():
    """
    Feature: parallel run case in _test_parallel_op_softmax
    Description:
        1. test_softmax_all_replicated —
        2. test_softmax_data_parallel —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_softmax_all_replicated", num_proc=4),
        TorchCase(IMPL_FILE, "test_softmax_data_parallel", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_softmax_group2_gloo():
    """
    Feature: parallel run case in _test_parallel_op_softmax
    Description:
        1. test_softmax_all_replicated —
        2. test_softmax_data_parallel —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_softmax_all_replicated", num_proc=4),
        TorchCase(IMPL_FILE, "test_softmax_data_parallel", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_softmax_group3():
    """
    Feature: parallel run case in _test_parallel_op_softmax
    Description:
        1. test_softmax_model_parallel —
        2. test_softmax_negative_dim —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_softmax_model_parallel", num_proc=4),
        TorchCase(IMPL_FILE, "test_softmax_negative_dim", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_softmax_group3_gloo():
    """
    Feature: parallel run case in _test_parallel_op_softmax
    Description:
        1. test_softmax_model_parallel —
        2. test_softmax_negative_dim —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_softmax_model_parallel", num_proc=4),
        TorchCase(IMPL_FILE, "test_softmax_negative_dim", num_proc=4),
    ])
