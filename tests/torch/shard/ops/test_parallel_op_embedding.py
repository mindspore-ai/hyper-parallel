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
"""Test runner for embedding distributed ST (PyTorch)."""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

IMPL_FILE = str(Path(__file__).resolve().parent / "_test_parallel_op_embedding.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_embedding_group1():
    """
    Feature: parallel run case in _test_parallel_op_embedding
    Description:
        1. test_embedding_args_positional —
        2. test_embedding_column_parallel —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_embedding_args_positional", num_proc=4),
        TorchCase(IMPL_FILE, "test_embedding_column_parallel", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_embedding_group1_gloo():
    """
    Feature: parallel run case in _test_parallel_op_embedding
    Description:
        1. test_embedding_args_positional —
        2. test_embedding_column_parallel —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_embedding_args_positional", num_proc=4),
        TorchCase(IMPL_FILE, "test_embedding_column_parallel", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_embedding_group2():
    """
    Feature: parallel run case in _test_parallel_op_embedding
    Description:
        1. test_embedding_cp_padding_and_scale —
        2. test_embedding_data_parallel —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_embedding_cp_padding_and_scale", num_proc=4),
        TorchCase(IMPL_FILE, "test_embedding_data_parallel", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_embedding_group2_gloo():
    """
    Feature: parallel run case in _test_parallel_op_embedding
    Description:
        1. test_embedding_cp_padding_and_scale —
        2. test_embedding_data_parallel —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_embedding_cp_padding_and_scale", num_proc=4),
        TorchCase(IMPL_FILE, "test_embedding_data_parallel", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_embedding_group3():
    """
    Feature: parallel run case in _test_parallel_op_embedding
    Description:
        1. test_embedding_dp_cp —
        2. test_embedding_dp_rp —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_embedding_dp_cp", num_proc=4),
        TorchCase(IMPL_FILE, "test_embedding_dp_rp", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_embedding_group3_gloo():
    """
    Feature: parallel run case in _test_parallel_op_embedding
    Description:
        1. test_embedding_dp_cp —
        2. test_embedding_dp_rp —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_embedding_dp_cp", num_proc=4),
        TorchCase(IMPL_FILE, "test_embedding_dp_rp", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_embedding_group4():
    """
    Feature: parallel run case in _test_parallel_op_embedding
    Description:
        1. test_embedding_row_parallel —
        2. test_embedding_rp_padding —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_embedding_row_parallel", num_proc=4),
        TorchCase(IMPL_FILE, "test_embedding_rp_padding", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_embedding_group4_gloo():
    """
    Feature: parallel run case in _test_parallel_op_embedding
    Description:
        1. test_embedding_row_parallel —
        2. test_embedding_rp_padding —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_embedding_row_parallel", num_proc=4),
        TorchCase(IMPL_FILE, "test_embedding_rp_padding", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_embedding_group5():
    """
    Feature: parallel run case in _test_parallel_op_embedding
    Description:
        1. test_embedding_sequence_parallel —
        2. test_embedding_sp_cp —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_embedding_sequence_parallel", num_proc=4),
        TorchCase(IMPL_FILE, "test_embedding_sp_cp", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_embedding_group5_gloo():
    """
    Feature: parallel run case in _test_parallel_op_embedding
    Description:
        1. test_embedding_sequence_parallel —
        2. test_embedding_sp_cp —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_embedding_sequence_parallel", num_proc=4),
        TorchCase(IMPL_FILE, "test_embedding_sp_cp", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_embedding_group6():
    """
    Feature: parallel run case in _test_parallel_op_embedding
    Description:
        1. test_embedding_sp_rp —
        2. test_embedding_weight_2d_sharding —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_embedding_sp_rp", num_proc=4),
        TorchCase(IMPL_FILE, "test_embedding_weight_2d_sharding", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_embedding_group6_gloo():
    """
    Feature: parallel run case in _test_parallel_op_embedding
    Description:
        1. test_embedding_sp_rp —
        2. test_embedding_weight_2d_sharding —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_embedding_sp_rp", num_proc=4),
        TorchCase(IMPL_FILE, "test_embedding_weight_2d_sharding", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_embedding_group7():
    """
    Feature: parallel run case in _test_parallel_op_embedding
    Description:
        1. test_embedding_weight_row_parallel_only — isolated with replicated input.
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_embedding_weight_row_parallel_only", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_embedding_group7_gloo():
    """
    Feature: parallel run case in _test_parallel_op_embedding
    Description:
        1. test_embedding_weight_row_parallel_only — isolated with replicated input.
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_embedding_weight_row_parallel_only", num_proc=4),
    ])
