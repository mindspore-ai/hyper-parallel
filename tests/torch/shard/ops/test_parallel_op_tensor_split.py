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
"""Test runner for tensor_split distributed ST (PyTorch)."""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

IMPL_FILE = str(Path(__file__).resolve().parent / "_test_parallel_op_tensor_split.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_tensor_split_group1():
    """
    Feature: parallel run case in _test_parallel_op_tensor_split
    Description:
        1. test_tensor_split_1d_tensor_indices —
        2. test_tensor_split_3d_sections —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_tensor_split_1d_tensor_indices", num_proc=4),
        TorchCase(IMPL_FILE, "test_tensor_split_3d_sections", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_tensor_split_group1_gloo():
    """
    Feature: parallel run case in _test_parallel_op_tensor_split
    Description:
        1. test_tensor_split_1d_tensor_indices —
        2. test_tensor_split_3d_sections —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_tensor_split_1d_tensor_indices", num_proc=4),
        TorchCase(IMPL_FILE, "test_tensor_split_3d_sections", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_tensor_split_group2():
    """
    Feature: parallel run case in _test_parallel_op_tensor_split
    Description:
        1. test_tensor_split_4d_multi_shard —
        2. test_tensor_split_by_indices_unsharded —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_tensor_split_4d_multi_shard", num_proc=4),
        TorchCase(IMPL_FILE, "test_tensor_split_by_indices_unsharded", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_tensor_split_group2_gloo():
    """
    Feature: parallel run case in _test_parallel_op_tensor_split
    Description:
        1. test_tensor_split_4d_multi_shard —
        2. test_tensor_split_by_indices_unsharded —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_tensor_split_4d_multi_shard", num_proc=4),
        TorchCase(IMPL_FILE, "test_tensor_split_by_indices_unsharded", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_tensor_split_group3():
    """
    Feature: parallel run case in _test_parallel_op_tensor_split
    Description:
        1. test_tensor_split_by_sections_unsharded —
        2. test_tensor_split_default_dim —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_tensor_split_by_sections_unsharded", num_proc=4),
        TorchCase(IMPL_FILE, "test_tensor_split_default_dim", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_tensor_split_group3_gloo():
    """
    Feature: parallel run case in _test_parallel_op_tensor_split
    Description:
        1. test_tensor_split_by_sections_unsharded —
        2. test_tensor_split_default_dim —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_tensor_split_by_sections_unsharded", num_proc=4),
        TorchCase(IMPL_FILE, "test_tensor_split_default_dim", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_tensor_split_group4():
    """
    Feature: parallel run case in _test_parallel_op_tensor_split
    Description:
        1. test_tensor_split_list_indices —
        2. test_tensor_split_negative_dim —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_tensor_split_list_indices", num_proc=4),
        TorchCase(IMPL_FILE, "test_tensor_split_negative_dim", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_tensor_split_group4_gloo():
    """
    Feature: parallel run case in _test_parallel_op_tensor_split
    Description:
        1. test_tensor_split_list_indices —
        2. test_tensor_split_negative_dim —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_tensor_split_list_indices", num_proc=4),
        TorchCase(IMPL_FILE, "test_tensor_split_negative_dim", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_tensor_split_group5():
    """
    Feature: parallel run case in _test_parallel_op_tensor_split
    Description:
        1. test_tensor_split_out_of_bounds_indices —
        2. test_tensor_split_replicated —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_tensor_split_out_of_bounds_indices", num_proc=4),
        TorchCase(IMPL_FILE, "test_tensor_split_replicated", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_tensor_split_group5_gloo():
    """
    Feature: parallel run case in _test_parallel_op_tensor_split
    Description:
        1. test_tensor_split_out_of_bounds_indices —
        2. test_tensor_split_replicated —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_tensor_split_out_of_bounds_indices", num_proc=4),
        TorchCase(IMPL_FILE, "test_tensor_split_replicated", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_tensor_split_group6():
    """
    Feature: parallel run case in _test_parallel_op_tensor_split
    Description:
        1. test_tensor_split_uneven_sections —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_tensor_split_uneven_sections", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_tensor_split_group6_gloo():
    """
    Feature: parallel run case in _test_parallel_op_tensor_split
    Description:
        1. test_tensor_split_uneven_sections —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_tensor_split_uneven_sections", num_proc=4),
    ])
