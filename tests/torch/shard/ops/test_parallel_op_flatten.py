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
"""Test runner for flatten distributed ST (PyTorch)."""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

IMPL_FILE = str(Path(__file__).resolve().parent / "_test_parallel_op_flatten.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_flatten_group1():
    """
    Feature: parallel run case in _test_parallel_op_flatten
    Description:
        1. test_flatten_2d_to_1d — - Flatten a 2D tensor to 1D (start_dim=0, end_dim=1).
        2. test_flatten_all_dims —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_flatten_2d_to_1d", num_proc=4),
        TorchCase(IMPL_FILE, "test_flatten_all_dims", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_flatten_group1_gloo():
    """
    Feature: parallel run case in _test_parallel_op_flatten
    Description:
        1. test_flatten_2d_to_1d — - Flatten a 2D tensor to 1D (start_dim=0, end_dim=1).
        2. test_flatten_all_dims —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_flatten_2d_to_1d", num_proc=4),
        TorchCase(IMPL_FILE, "test_flatten_all_dims", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_flatten_group2():
    """
    Feature: parallel run case in _test_parallel_op_flatten
    Description:
        1. test_flatten_default_args — - Apply flatten() without explicit start_dim and end_dim.
        2. test_flatten_middle_dims — - Flatten dimensions 1 and 2 of a 4D distributed tensor.
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_flatten_default_args", num_proc=4),
        TorchCase(IMPL_FILE, "test_flatten_middle_dims", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_flatten_group2_gloo():
    """
    Feature: parallel run case in _test_parallel_op_flatten
    Description:
        1. test_flatten_default_args — - Apply flatten() without explicit start_dim and end_dim.
        2. test_flatten_middle_dims — - Flatten dimensions 1 and 2 of a 4D distributed tensor.
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_flatten_default_args", num_proc=4),
        TorchCase(IMPL_FILE, "test_flatten_middle_dims", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_flatten_group3():
    """
    Feature: parallel run case in _test_parallel_op_flatten
    Description:
        1. test_flatten_negative_dims — - Flatten dimensions using negative indices (-2, -1) on a 3D distributed tensor.
        2. test_flatten_scalar — - Apply flatten(0, -1) to a distributed scalar tensor.
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_flatten_negative_dims", num_proc=4),
        TorchCase(IMPL_FILE, "test_flatten_scalar", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_flatten_group3_gloo():
    """
    Feature: parallel run case in _test_parallel_op_flatten
    Description:
        1. test_flatten_negative_dims — - Flatten dimensions using negative indices (-2, -1) on a 3D distributed tensor.
        2. test_flatten_scalar — - Apply flatten(0, -1) to a distributed scalar tensor.
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_flatten_negative_dims", num_proc=4),
        TorchCase(IMPL_FILE, "test_flatten_scalar", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_flatten_group4():
    """
    Feature: parallel run case in _test_parallel_op_flatten
    Description:
        1. test_flatten_single_dim — - Flatten a single dimension (e.g., start_dim=1, end_dim=1).
        2. test_flatten_unsharded — - Flatten dimensions 1 and 2 of a 3D distributed tensor.
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_flatten_single_dim", num_proc=4),
        TorchCase(IMPL_FILE, "test_flatten_unsharded", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_flatten_group4_gloo():
    """
    Feature: parallel run case in _test_parallel_op_flatten
    Description:
        1. test_flatten_single_dim — - Flatten a single dimension (e.g., start_dim=1, end_dim=1).
        2. test_flatten_unsharded — - Flatten dimensions 1 and 2 of a 3D distributed tensor.
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_flatten_single_dim", num_proc=4),
        TorchCase(IMPL_FILE, "test_flatten_unsharded", num_proc=4),
    ])
