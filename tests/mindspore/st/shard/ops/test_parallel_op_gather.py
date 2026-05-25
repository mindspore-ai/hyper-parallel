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
"""Test runner for gather_d and gather_nd distributed ST (MindSpore)."""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase

IMPL_FILE = str(Path(__file__).resolve().parent / "_test_parallel_op_gather.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_gather_group1():
    """
    Feature: parallel run case in _test_parallel_op_gather
    Description:
        1. test_gatherd_data_parallel_dim0_1 — GatherD batch-sharded input, replicated index
        2. test_gatherd_data_parallel_dim1_2 — GatherD sequence-sharded input, replicated index
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_gatherd_data_parallel_dim0_1", worker_num=4, local_worker_num=4, glog_v=2),
        MindSporeCase(IMPL_FILE, "test_gatherd_data_parallel_dim1_2", worker_num=4, local_worker_num=4, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_gather_group2():
    """
    Feature: parallel run case in _test_parallel_op_gather
    Description:
        1. test_gatherd_row_parallel_3 — GatherD row-parallel
        2. test_gatherd_input_multi_shard_4 — GatherD multi-shard input
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_gatherd_row_parallel_3", worker_num=4, local_worker_num=4, glog_v=2),
        MindSporeCase(IMPL_FILE, "test_gatherd_input_multi_shard_4", worker_num=4, local_worker_num=4, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_gather_group3():
    """
    Feature: parallel run case in _test_parallel_op_gather
    Description:
        1. test_gatherd_3d_mesh_input_multi_shard_5 — GatherD 3D-mesh gather-axis sharding
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_gatherd_3d_mesh_input_multi_shard_5", worker_num=8, local_worker_num=8,
            glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_gather_group4():
    """
    Feature: parallel run case in _test_parallel_op_gather
    Description:
        1. test_gatherd_3d_mesh_matched_non_dim_shard_6 — GatherD 3D-mesh matched non-dim shard
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_gatherd_3d_mesh_matched_non_dim_shard_6", worker_num=8, local_worker_num=8,
        glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_gather_group5():
    """
    Feature: parallel run case in _test_parallel_op_gather
    Description:
        1. test_gathernd_partial_model_parallel_1 — GatherNd indices sharded by tp
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_gathernd_partial_model_parallel_1", worker_num=8, local_worker_num=8, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_gather_group6():
    """
    Feature: parallel run case in _test_parallel_op_gather
    Description:
        1. test_gathernd_partial_data_parallel_2 — GatherNd indices sharded by dp
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_gathernd_partial_data_parallel_2", worker_num=8, local_worker_num=8, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_gather_group7():
    """
    Feature: parallel run case in _test_parallel_op_gather
    Description:
        1. test_gathernd_params_plain_tensor — GatherNd with plain Tensor params
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_gathernd_params_plain_tensor", worker_num=8, local_worker_num=8, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_gather_group8():
    """
    Feature: parallel run case in _test_parallel_op_gather
    Description:
        1. test_gathernd_k1_trailing_dims_shard_3 — GatherNd K=1 trailing dims shard
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_gathernd_k1_trailing_dims_shard_3", worker_num=8, local_worker_num=8, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_gather_group9():
    """
    Feature: parallel run case in _test_parallel_op_gather
    Description:
        1. test_gathernd_k2_trailing_dim_shard_4 — GatherNd K=2 trailing dim shard
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_gathernd_k2_trailing_dim_shard_4", worker_num=8, local_worker_num=8, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_gather_group10():
    """
    Feature: parallel run case in _test_parallel_op_gather
    Description:
        1. test_gathernd_k3_no_trailing_dims_5 — GatherNd K=3 no trailing dims
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_gathernd_k3_no_trailing_dims_5", worker_num=8, local_worker_num=8, glog_v=2),
    ])
