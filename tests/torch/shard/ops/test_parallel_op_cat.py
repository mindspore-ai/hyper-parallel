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
"""Test runner for cat distributed ST (PyTorch)."""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

IMPL_FILE = str(Path(__file__).resolve().parent / "_test_parallel_op_cat.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_cat_group1():
    """
    Feature: parallel run case in _test_parallel_op_cat
    Description:
        1. test_cat_3d_complex —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_cat_3d_complex", num_proc=8),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_cat_group1_gloo():
    """
    Feature: parallel run case in _test_parallel_op_cat
    Description:
        1. test_cat_3d_complex —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_cat_3d_complex", num_proc=8),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_cat_group2():
    """
    Feature: parallel run case in _test_parallel_op_cat
    Description:
        1. test_cat_4d_tensor —
        2. test_cat_5d_mixed_placements —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_cat_4d_tensor", num_proc=4),
        TorchCase(IMPL_FILE, "test_cat_5d_mixed_placements", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_cat_group2_gloo():
    """
    Feature: parallel run case in _test_parallel_op_cat
    Description:
        1. test_cat_4d_tensor —
        2. test_cat_5d_mixed_placements —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_cat_4d_tensor", num_proc=4),
        TorchCase(IMPL_FILE, "test_cat_5d_mixed_placements", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_cat_group3():
    """
    Feature: parallel run case in _test_parallel_op_cat
    Description:
        1. test_cat_basic —
        2. test_cat_mismatched_shapes —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_cat_basic", num_proc=4),
        TorchCase(IMPL_FILE, "test_cat_mismatched_shapes", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_cat_group3_gloo():
    """
    Feature: parallel run case in _test_parallel_op_cat
    Description:
        1. test_cat_basic —
        2. test_cat_mismatched_shapes —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_cat_basic", num_proc=4),
        TorchCase(IMPL_FILE, "test_cat_mismatched_shapes", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_cat_group4():
    """
    Feature: parallel run case in _test_parallel_op_cat
    Description:
        1. test_cat_multiple_tensors —
        2. test_cat_shard_last_cat_first —
        3. test_cat_singleton_dimension —
        4. test_cat_with_empty —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_cat_multiple_tensors", num_proc=2),
        TorchCase(IMPL_FILE, "test_cat_shard_last_cat_first", num_proc=2),
        TorchCase(IMPL_FILE, "test_cat_singleton_dimension", num_proc=2),
        TorchCase(IMPL_FILE, "test_cat_with_empty", num_proc=2),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_cat_group4_gloo():
    """
    Feature: parallel run case in _test_parallel_op_cat
    Description:
        1. test_cat_multiple_tensors —
        2. test_cat_shard_last_cat_first —
        3. test_cat_singleton_dimension —
        4. test_cat_with_empty —
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_cat_multiple_tensors", num_proc=2),
        TorchCase(IMPL_FILE, "test_cat_shard_last_cat_first", num_proc=2),
        TorchCase(IMPL_FILE, "test_cat_singleton_dimension", num_proc=2),
        TorchCase(IMPL_FILE, "test_cat_with_empty", num_proc=2),
    ])
