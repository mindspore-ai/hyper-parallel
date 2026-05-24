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
"""Test runner for min distributed ST (PyTorch)."""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

IMPL_FILE = str(Path(__file__).resolve().parent / "_test_parallel_op_min.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_min_group1():
    """
    Feature: parallel run case in _test_parallel_op_min
    Description:
        1. test_min_3d_element_wise —
        2. test_min_3d_reduce_negative_dim —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_min_3d_element_wise", num_proc=4),
        TorchCase(IMPL_FILE, "test_min_3d_reduce_negative_dim", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_min_group1_gloo():
    """
    Feature: parallel run case in _test_parallel_op_min
    Description:
        1. test_min_3d_element_wise —
        2. test_min_3d_reduce_negative_dim —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_min_3d_element_wise", num_proc=4),
        TorchCase(IMPL_FILE, "test_min_3d_reduce_negative_dim", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_min_group2():
    """
    Feature: parallel run case in _test_parallel_op_min
    Description:
        1. test_min_3d_reduce_sharded_dim —
        2. test_min_4_cards —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_min_3d_reduce_sharded_dim", num_proc=4),
        TorchCase(IMPL_FILE, "test_min_4_cards", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_min_group2_gloo():
    """
    Feature: parallel run case in _test_parallel_op_min
    Description:
        1. test_min_3d_reduce_sharded_dim —
        2. test_min_4_cards —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_min_3d_reduce_sharded_dim", num_proc=4),
        TorchCase(IMPL_FILE, "test_min_4_cards", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_min_group3():
    """
    Feature: parallel run case in _test_parallel_op_min
    Description:
        1. test_min_dim_reduce_replicated —
        2. test_min_dim_reduce_sharded —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_min_dim_reduce_replicated", num_proc=4),
        TorchCase(IMPL_FILE, "test_min_dim_reduce_sharded", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_min_group3_gloo():
    """
    Feature: parallel run case in _test_parallel_op_min
    Description:
        1. test_min_dim_reduce_replicated —
        2. test_min_dim_reduce_sharded —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_min_dim_reduce_replicated", num_proc=4),
        TorchCase(IMPL_FILE, "test_min_dim_reduce_sharded", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_min_group4():
    """
    Feature: parallel run case in _test_parallel_op_min
    Description:
        1. test_min_element_wise —
        2. test_min_global_reduce —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_min_element_wise", num_proc=4),
        TorchCase(IMPL_FILE, "test_min_global_reduce", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_min_group4_gloo():
    """
    Feature: parallel run case in _test_parallel_op_min
    Description:
        1. test_min_element_wise —
        2. test_min_global_reduce —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_min_element_wise", num_proc=4),
        TorchCase(IMPL_FILE, "test_min_global_reduce", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_min_group5():
    """
    Feature: parallel run case in _test_parallel_op_min
    Description:
        1. test_min_keepdim —
        2. test_min_keepdim_negative_dim —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_min_keepdim", num_proc=4),
        TorchCase(IMPL_FILE, "test_min_keepdim_negative_dim", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_min_group5_gloo():
    """
    Feature: parallel run case in _test_parallel_op_min
    Description:
        1. test_min_keepdim —
        2. test_min_keepdim_negative_dim —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_min_keepdim", num_proc=4),
        TorchCase(IMPL_FILE, "test_min_keepdim_negative_dim", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_min_group6():
    """
    Feature: parallel run case in _test_parallel_op_min
    Description:
        1. test_min_1d_mesh_element_wise —
        2. test_min_1d_mesh_global_reduce —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_min_1d_mesh_element_wise", num_proc=2),
        TorchCase(IMPL_FILE, "test_min_1d_mesh_global_reduce", num_proc=2),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_min_group6_gloo():
    """
    Feature: parallel run case in _test_parallel_op_min
    Description:
        1. test_min_1d_mesh_element_wise —
        2. test_min_1d_mesh_global_reduce —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_min_1d_mesh_element_wise", num_proc=2),
        TorchCase(IMPL_FILE, "test_min_1d_mesh_global_reduce", num_proc=2),
    ])
