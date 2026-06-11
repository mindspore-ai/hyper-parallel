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
"""Test runner for index_select distributed ST (PyTorch)."""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

IMPL_FILE = str(Path(__file__).resolve().parent / "_test_parallel_op_index_select.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_index_select_group1():
    """
    Feature: parallel run case in _test_parallel_op_index_select
    Description:
        1. test_index_select_2d_dim0 —
        2. test_index_select_2d_dim1 —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_index_select_2d_dim0", num_proc=4),
        TorchCase(IMPL_FILE, "test_index_select_2d_dim1", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_index_select_group1_gloo():
    """
    Feature: parallel run case in _test_parallel_op_index_select
    Description:
        1. test_index_select_2d_dim0 —
        2. test_index_select_2d_dim1 —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_index_select_2d_dim0", num_proc=4),
        TorchCase(IMPL_FILE, "test_index_select_2d_dim1", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_index_select_group2():
    """
    Feature: parallel run case in _test_parallel_op_index_select
    Description:
        1. test_index_select_3d —
        2. test_index_select_3d_dim1 —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_index_select_3d", num_proc=4),
        TorchCase(IMPL_FILE, "test_index_select_3d_dim1", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_index_select_group2_gloo():
    """
    Feature: parallel run case in _test_parallel_op_index_select
    Description:
        1. test_index_select_3d —
        2. test_index_select_3d_dim1 —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_index_select_3d", num_proc=4),
        TorchCase(IMPL_FILE, "test_index_select_3d_dim1", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_index_select_group3():
    """
    Feature: parallel run case in _test_parallel_op_index_select
    Description:
        1. test_index_select_basic —
        2. test_index_select_duplicate_indices_sharded —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_index_select_basic", num_proc=4),
        TorchCase(IMPL_FILE, "test_index_select_duplicate_indices_sharded", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_index_select_group3_gloo():
    """
    Feature: parallel run case in _test_parallel_op_index_select
    Description:
        1. test_index_select_basic —
        2. test_index_select_duplicate_indices_sharded —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_index_select_basic", num_proc=4),
        TorchCase(IMPL_FILE, "test_index_select_duplicate_indices_sharded", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_index_select_group4():
    """
    Feature: parallel run case in _test_parallel_op_index_select
    Description:
        1. test_index_select_fully_replicated —
        2. test_index_select_negative_dim —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_index_select_fully_replicated", num_proc=4),
        TorchCase(IMPL_FILE, "test_index_select_negative_dim", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_index_select_group4_gloo():
    """
    Feature: parallel run case in _test_parallel_op_index_select
    Description:
        1. test_index_select_fully_replicated —
        2. test_index_select_negative_dim —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_index_select_fully_replicated", num_proc=4),
        TorchCase(IMPL_FILE, "test_index_select_negative_dim", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_index_select_group5():
    """
    Feature: parallel run case in _test_parallel_op_index_select
    Description:
        1. test_index_select_negative_sharded_dim —
        2. test_index_select_out_of_order_sharded —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_index_select_negative_sharded_dim", num_proc=4),
        TorchCase(IMPL_FILE, "test_index_select_out_of_order_sharded", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_index_select_group5_gloo():
    """
    Feature: parallel run case in _test_parallel_op_index_select
    Description:
        1. test_index_select_negative_sharded_dim —
        2. test_index_select_out_of_order_sharded —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_index_select_negative_sharded_dim", num_proc=4),
        TorchCase(IMPL_FILE, "test_index_select_out_of_order_sharded", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_index_select_group6():
    """
    Feature: parallel run case in _test_parallel_op_index_select
    Description:
        1. test_index_select_sharded_dim0_2d —
        2. test_index_select_sharded_dim1_2d —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_index_select_sharded_dim0_2d", num_proc=4),
        TorchCase(IMPL_FILE, "test_index_select_sharded_dim1_2d", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_index_select_group6_gloo():
    """
    Feature: parallel run case in _test_parallel_op_index_select
    Description:
        1. test_index_select_sharded_dim0_2d —
        2. test_index_select_sharded_dim1_2d —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_index_select_sharded_dim0_2d", num_proc=4),
        TorchCase(IMPL_FILE, "test_index_select_sharded_dim1_2d", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_index_select_group7():
    """
    Feature: parallel run case in _test_parallel_op_index_select
    Description:
        1. test_index_select_sharded_dim2_3d —
        2. test_index_select_single_element —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_index_select_sharded_dim2_3d", num_proc=4),
        TorchCase(IMPL_FILE, "test_index_select_single_element", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_index_select_group7_gloo():
    """
    Feature: parallel run case in _test_parallel_op_index_select
    Description:
        1. test_index_select_sharded_dim2_3d —
        2. test_index_select_single_element —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_index_select_sharded_dim2_3d", num_proc=4),
        TorchCase(IMPL_FILE, "test_index_select_single_element", num_proc=4),
    ])
