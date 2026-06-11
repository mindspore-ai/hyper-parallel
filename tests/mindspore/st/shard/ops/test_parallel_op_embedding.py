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
"""Test runner for embedding distributed ST (MindSpore)."""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase

IMPL_FILE = str(Path(__file__).resolve().parent / "_test_parallel_op_embedding.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_embedding_group1():
    """
    Feature: parallel run case in _test_parallel_op_embedding
    Description:
        1. test_distributed_embedding_data_parallel — Embedding data parallel
        2. test_distributed_embedding_column_parallel — Embedding column parallel
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_distributed_embedding_data_parallel", worker_num=4, local_worker_num=4,
            glog_v=2),
        MindSporeCase(IMPL_FILE, "test_distributed_embedding_column_parallel", worker_num=4, local_worker_num=4,
        glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_embedding_group2():
    """
    Feature: parallel run case in _test_parallel_op_embedding
    Description:
        1. test_distributed_embedding_row_parallel — Embedding row parallel
        2. test_distributed_embedding_dp_cp — Embedding data parallel + column parallel
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_distributed_embedding_row_parallel", worker_num=4, local_worker_num=4, glog_v=2),
        MindSporeCase(IMPL_FILE, "test_distributed_embedding_dp_cp", worker_num=4, local_worker_num=4, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_embedding_group3():
    """
    Feature: parallel run case in _test_parallel_op_embedding
    Description:
        1. test_distributed_embedding_dp_rp — Embedding data parallel + row parallel
        2. test_distributed_embedding_sequence_parallel — Embedding sequence parallel
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_distributed_embedding_dp_rp", worker_num=4, local_worker_num=4, glog_v=2),
        MindSporeCase(IMPL_FILE, "test_distributed_embedding_sequence_parallel", worker_num=4, local_worker_num=4,
        glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_embedding_group4():
    """
    Feature: parallel run case in _test_parallel_op_embedding
    Description:
        1. test_distributed_embedding_sp_cp — Embedding sequence parallel + column parallel
        2. test_distributed_embedding_sp_rp — Embedding sequence parallel + row parallel
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_distributed_embedding_sp_cp", worker_num=4, local_worker_num=4, glog_v=2),
        MindSporeCase(IMPL_FILE, "test_distributed_embedding_sp_rp", worker_num=4, local_worker_num=4, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_embedding_group5():
    """
    Feature: parallel run case in _test_parallel_op_embedding
    Description:
        1. test_distributed_embedding_weight_2d_sharding — Embedding weight 2D sharding
        2. test_distributed_embedding_weight_row_parallel_only — Embedding weight row parallel only
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_distributed_embedding_weight_2d_sharding", worker_num=4, local_worker_num=4,
            glog_v=2),
        MindSporeCase(IMPL_FILE, "test_distributed_embedding_weight_row_parallel_only", worker_num=4,
        local_worker_num=4, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_embedding_group6():
    """
    Feature: parallel run case in _test_parallel_op_embedding
    Description:
        1. test_distributed_embedding_cp_padding_and_scale — Embedding column parallel with padding_idx &
        scale_grad_by_freq
        2. test_distributed_embedding_rp_padding — Embedding row parallel with padding_idx
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_distributed_embedding_cp_padding_and_scale", worker_num=4, local_worker_num=4,
        glog_v=2),
        MindSporeCase(IMPL_FILE, "test_distributed_embedding_rp_padding", worker_num=4, local_worker_num=4, glog_v=2),
    ])
