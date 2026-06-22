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
"""test parallel op getitem"""
from pathlib import Path
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

IMPL_FILE = str(Path(__file__).resolve().parent / "_test_parallel_op_getitem.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_getitem_group1():
    """
    Feature: parallel run case in shard
    Description:
        1.test_getitem_basic_int_replicated
        2.test_getitem_basic_slice_keep_dim
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_getitem_basic_int_replicated", num_proc=4),
        TorchCase(IMPL_FILE, "test_getitem_basic_slice_keep_dim", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_getitem_group1_gloo():
    """
    Feature: parallel run case in shard
    Description:
        1.test_getitem_basic_int_replicated
        2.test_getitem_basic_slice_keep_dim
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_getitem_basic_int_replicated", num_proc=4),
        TorchCase(IMPL_FILE, "test_getitem_basic_slice_keep_dim", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_getitem_group2():
    """
    Feature: parallel run case in shard
    Description:
        1.test_getitem_basic_newaxis
        2.test_getitem_basic_ellipsis
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_getitem_basic_newaxis", num_proc=4),
        TorchCase(IMPL_FILE, "test_getitem_basic_ellipsis", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_getitem_group2_gloo():
    """
    Feature: parallel run case in shard
    Description:
        1.test_getitem_basic_newaxis
        2.test_getitem_basic_ellipsis
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_getitem_basic_newaxis", num_proc=4),
        TorchCase(IMPL_FILE, "test_getitem_basic_ellipsis", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_getitem_group3():
    """
    Feature: parallel run case in shard
    Description:
        1.test_getitem_advanced_single_list
        2.test_getitem_mixed_basic
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_getitem_advanced_single_list", num_proc=4),
        TorchCase(IMPL_FILE, "test_getitem_mixed_basic", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_getitem_group3_gloo():
    """
    Feature: parallel run case in shard
    Description:
        1.test_getitem_advanced_single_list
        2.test_getitem_mixed_basic
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_getitem_advanced_single_list", num_proc=4),
        TorchCase(IMPL_FILE, "test_getitem_mixed_basic", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_getitem_group4():
    """
    Feature: parallel run case in shard
    Description:
        1.test_getitem_chained
        2.test_getitem_advanced_keep_shard_outside
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_getitem_chained", num_proc=4),
        TorchCase(IMPL_FILE, "test_getitem_advanced_keep_shard_outside", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_getitem_group4_gloo():
    """
    Feature: parallel run case in shard
    Description:
        1.test_getitem_chained
        2.test_getitem_advanced_keep_shard_outside
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_getitem_chained", num_proc=4),
        TorchCase(IMPL_FILE, "test_getitem_advanced_keep_shard_outside", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_getitem_group5():
    """
    Feature: parallel run case in shard
    Description:
        1.test_getitem_basic_tuple_int
        2.test_getitem_zero_size_slice
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_getitem_basic_tuple_int", num_proc=4),
        TorchCase(IMPL_FILE, "test_getitem_zero_size_slice", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_getitem_group5_gloo():
    """
    Feature: parallel run case in shard
    Description:
        1.test_getitem_basic_tuple_int
        2.test_getitem_zero_size_slice
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_getitem_basic_tuple_int", num_proc=4),
        TorchCase(IMPL_FILE, "test_getitem_zero_size_slice", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_getitem_group6():
    """
    Feature: parallel run case in shard
    Description:
        1.test_getitem_advanced_paired
        2.test_getitem_advanced_multi_d_index
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_getitem_advanced_paired", num_proc=4),
        TorchCase(IMPL_FILE, "test_getitem_advanced_multi_d_index", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_getitem_group6_gloo():
    """
    Feature: parallel run case in shard
    Description:
        1.test_getitem_advanced_paired
        2.test_getitem_advanced_multi_d_index
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_getitem_advanced_paired", num_proc=4),
        TorchCase(IMPL_FILE, "test_getitem_advanced_multi_d_index", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_getitem_group7():
    """
    Feature: parallel run case in shard
    Description:
        1.test_getitem_advanced_consecutive
        2.test_getitem_advanced_split
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_getitem_advanced_consecutive", num_proc=4),
        TorchCase(IMPL_FILE, "test_getitem_advanced_split", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_getitem_group7_gloo():
    """
    Feature: parallel run case in shard
    Description:
        1.test_getitem_advanced_consecutive
        2.test_getitem_advanced_split
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_getitem_advanced_consecutive", num_proc=4),
        TorchCase(IMPL_FILE, "test_getitem_advanced_split", num_proc=4),
    ])
