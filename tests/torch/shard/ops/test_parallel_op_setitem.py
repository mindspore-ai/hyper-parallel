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
"""test parallel op setitem"""
from pathlib import Path
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

IMPL_FILE = str(Path(__file__).resolve().parent / "_test_parallel_op_setitem.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_setitem_group1():
    """
    Feature: parallel run case in shard
    Description:
        1.test_setitem_scalar
        2.test_setitem_tensor_replicated
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_setitem_scalar", num_proc=4),
        TorchCase(IMPL_FILE, "test_setitem_tensor_replicated", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_setitem_group1_gloo():
    """
    Feature: parallel run case in shard
    Description:
        1.test_setitem_scalar
        2.test_setitem_tensor_replicated
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_setitem_scalar", num_proc=4),
        TorchCase(IMPL_FILE, "test_setitem_tensor_replicated", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_setitem_group2():
    """
    Feature: parallel run case in shard
    Description:
        1.test_setitem_shard_kept_dim
        2.test_setitem_advanced
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_setitem_shard_kept_dim", num_proc=4),
        TorchCase(IMPL_FILE, "test_setitem_advanced", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_setitem_group2_gloo():
    """
    Feature: parallel run case in shard
    Description:
        1.test_setitem_shard_kept_dim
        2.test_setitem_advanced
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_setitem_shard_kept_dim", num_proc=4),
        TorchCase(IMPL_FILE, "test_setitem_advanced", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_setitem_group3():
    """
    Feature: parallel run case in shard
    Description:
        1.test_setitem_view_inplace_zero_
        2.test_setitem_view_inplace_add_
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_setitem_view_inplace_zero_", num_proc=4),
        TorchCase(IMPL_FILE, "test_setitem_view_inplace_add_", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_setitem_group3_gloo():
    """
    Feature: parallel run case in shard
    Description:
        1.test_setitem_view_inplace_zero_
        2.test_setitem_view_inplace_add_
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_setitem_view_inplace_zero_", num_proc=4),
        TorchCase(IMPL_FILE, "test_setitem_view_inplace_add_", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_setitem_group4():
    """
    Feature: parallel run case in shard
    Description:
        1.test_setitem_int_view_inplace_zero_
        2.test_setitem_int_view_inplace_add_
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_setitem_int_view_inplace_zero_", num_proc=4),
        TorchCase(IMPL_FILE, "test_setitem_int_view_inplace_add_", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_setitem_group4_gloo():
    """
    Feature: parallel run case in shard
    Description:
        1.test_setitem_int_view_inplace_zero_
        2.test_setitem_int_view_inplace_add_
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_setitem_int_view_inplace_zero_", num_proc=4),
        TorchCase(IMPL_FILE, "test_setitem_int_view_inplace_add_", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_setitem_group5():
    """
    Feature: parallel run case in shard
    Description:
        1.test_setitem_broadcast_tensor_value
        2.test_setitem_broadcast_tensor_shard_kept
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_setitem_broadcast_tensor_value", num_proc=4),
        TorchCase(IMPL_FILE, "test_setitem_broadcast_tensor_shard_kept", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_setitem_group5_gloo():
    """
    Feature: parallel run case in shard
    Description:
        1.test_setitem_broadcast_tensor_value
        2.test_setitem_broadcast_tensor_shard_kept
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_setitem_broadcast_tensor_value", num_proc=4),
        TorchCase(IMPL_FILE, "test_setitem_broadcast_tensor_shard_kept", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_setitem_group6():
    """
    Feature: parallel run case in shard
    Description:
        1.test_setitem_global_tensor_dim0_shard
        2.test_setitem_broadcast_tensor_1d_dim0_shard
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_setitem_global_tensor_dim0_shard", num_proc=4),
        TorchCase(IMPL_FILE, "test_setitem_broadcast_tensor_1d_dim0_shard", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_setitem_group6_gloo():
    """
    Feature: parallel run case in shard
    Description:
        1.test_setitem_global_tensor_dim0_shard
        2.test_setitem_broadcast_tensor_1d_dim0_shard
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_setitem_global_tensor_dim0_shard", num_proc=4),
        TorchCase(IMPL_FILE, "test_setitem_broadcast_tensor_1d_dim0_shard", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_setitem_group7():
    """
    Feature: parallel run case in shard
    Description:
        1.test_setitem_broadcast_tensor_2d_dim0_shard
        2.test_setitem_global_tensor_dim1_shard
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_setitem_broadcast_tensor_2d_dim0_shard", num_proc=4),
        TorchCase(IMPL_FILE, "test_setitem_global_tensor_dim1_shard", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_setitem_group7_gloo():
    """
    Feature: parallel run case in shard
    Description:
        1.test_setitem_broadcast_tensor_2d_dim0_shard
        2.test_setitem_global_tensor_dim1_shard
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_setitem_broadcast_tensor_2d_dim0_shard", num_proc=4),
        TorchCase(IMPL_FILE, "test_setitem_global_tensor_dim1_shard", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_setitem_group8():
    """
    Feature: parallel run case in shard
    Description:
        1.test_setitem_broadcast_tensor_dim1_shard
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_setitem_broadcast_tensor_dim1_shard", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_setitem_group8_gloo():
    """
    Feature: parallel run case in shard
    Description:
        1.test_setitem_broadcast_tensor_dim1_shard
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_setitem_broadcast_tensor_dim1_shard", num_proc=4),
    ])
