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
"""Test runner for conv3d distributed ST (PyTorch)."""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

IMPL_FILE = str(Path(__file__).resolve().parent / "_test_parallel_op_conv3d.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_conv3d_group1():
    """
    Feature: parallel run case in _test_parallel_op_conv3d
    Description:
        1. test_distributed_conv3d_column_parallel —
        2. test_distributed_conv3d_data_parallel —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_distributed_conv3d_column_parallel", num_proc=4),
        TorchCase(IMPL_FILE, "test_distributed_conv3d_data_parallel", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_conv3d_group1_gloo():
    """
    Feature: parallel run case in _test_parallel_op_conv3d
    Description:
        1. test_distributed_conv3d_column_parallel —
        2. test_distributed_conv3d_data_parallel —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_distributed_conv3d_column_parallel", num_proc=4),
        TorchCase(IMPL_FILE, "test_distributed_conv3d_data_parallel", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_conv3d_group2():
    """
    Feature: parallel run case in _test_parallel_op_conv3d
    Description:
        1. test_distributed_conv3d_dp_cp —
        2. test_distributed_conv3d_dp_rp —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_distributed_conv3d_dp_cp", num_proc=4),
        TorchCase(IMPL_FILE, "test_distributed_conv3d_dp_rp", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_conv3d_group2_gloo():
    """
    Feature: parallel run case in _test_parallel_op_conv3d
    Description:
        1. test_distributed_conv3d_dp_cp —
        2. test_distributed_conv3d_dp_rp —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_distributed_conv3d_dp_cp", num_proc=4),
        TorchCase(IMPL_FILE, "test_distributed_conv3d_dp_rp", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_conv3d_group3():
    """
    Feature: parallel run case in _test_parallel_op_conv3d
    Description:
        1. test_distributed_conv3d_groups_cp —
        2. test_distributed_conv3d_groups_cp_with_bias —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_distributed_conv3d_groups_cp", num_proc=4),
        TorchCase(IMPL_FILE, "test_distributed_conv3d_groups_cp_with_bias", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_conv3d_group3_gloo():
    """
    Feature: parallel run case in _test_parallel_op_conv3d
    Description:
        1. test_distributed_conv3d_groups_cp —
        2. test_distributed_conv3d_groups_cp_with_bias —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_distributed_conv3d_groups_cp", num_proc=4),
        TorchCase(IMPL_FILE, "test_distributed_conv3d_groups_cp_with_bias", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_conv3d_group4():
    """
    Feature: parallel run case in _test_parallel_op_conv3d
    Description:
        1. test_distributed_conv3d_groups_dp —
        2. test_distributed_conv3d_row_parallel —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_distributed_conv3d_groups_dp", num_proc=4),
        TorchCase(IMPL_FILE, "test_distributed_conv3d_row_parallel", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_conv3d_group4_gloo():
    """
    Feature: parallel run case in _test_parallel_op_conv3d
    Description:
        1. test_distributed_conv3d_groups_dp —
        2. test_distributed_conv3d_row_parallel —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_distributed_conv3d_groups_dp", num_proc=4),
        TorchCase(IMPL_FILE, "test_distributed_conv3d_row_parallel", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_conv3d_group5():
    """
    Feature: parallel run case in _test_parallel_op_conv3d
    Description:
        1. test_distributed_conv3d_spatial_h —
        2. test_distributed_conv3d_spatial_parallel —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_distributed_conv3d_spatial_h", num_proc=4),
        TorchCase(IMPL_FILE, "test_distributed_conv3d_spatial_parallel", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_conv3d_group5_gloo():
    """
    Feature: parallel run case in _test_parallel_op_conv3d
    Description:
        1. test_distributed_conv3d_spatial_h —
        2. test_distributed_conv3d_spatial_parallel —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_distributed_conv3d_spatial_h", num_proc=4),
        TorchCase(IMPL_FILE, "test_distributed_conv3d_spatial_parallel", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_conv3d_group6():
    """
    Feature: parallel run case in _test_parallel_op_conv3d
    Description:
        1. test_distributed_conv3d_spatial_w —
        2. test_distributed_conv3d_with_bias —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_distributed_conv3d_spatial_w", num_proc=4),
        TorchCase(IMPL_FILE, "test_distributed_conv3d_with_bias", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_conv3d_group6_gloo():
    """
    Feature: parallel run case in _test_parallel_op_conv3d
    Description:
        1. test_distributed_conv3d_spatial_w —
        2. test_distributed_conv3d_with_bias —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_distributed_conv3d_spatial_w", num_proc=4),
        TorchCase(IMPL_FILE, "test_distributed_conv3d_with_bias", num_proc=4),
    ])
