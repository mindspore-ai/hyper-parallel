# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""Test runner for expand_dims and expand_dims_view distributed ST (MindSpore)."""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase

IMPL_FILE = str(Path(__file__).resolve().parent / "_test_parallel_op_expand_dims.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_expand_dims_group1():
    """
    Feature: parallel run case in _test_parallel_op_expand_dims
    Description:
        1. test_expanddims_data_parallel_1 — ExpandDims data parallel
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_expanddims_data_parallel_1", worker_num=8, local_worker_num=8, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_expand_dims_group2():
    """
    Feature: parallel run case in _test_parallel_op_expand_dims
    Description:
        1. test_expanddims_model_parallel_2 — ExpandDims model parallel
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_expanddims_model_parallel_2", worker_num=8, local_worker_num=8, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_expand_dims_group3():
    """
    Feature: parallel run case in _test_parallel_op_expand_dims
    Description:
        1. test_expanddims_hybrid_parallel_3 — ExpandDims hybrid parallel
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_expanddims_hybrid_parallel_3", worker_num=8, local_worker_num=8, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_expand_dims_group4():
    """
    Feature: parallel run case in _test_parallel_op_expand_dims
    Description:
        1. test_expanddims_insert_middle_4 — ExpandDims insert axis in middle
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_expanddims_insert_middle_4", worker_num=8, local_worker_num=8, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_expand_dims_group5():
    """
    Feature: parallel run case in _test_parallel_op_expand_dims
    Description:
        1. test_expanddims_negative_axis_5 — ExpandDims negative axis
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_expanddims_negative_axis_5", worker_num=8, local_worker_num=8, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_expand_dims_group6():
    """
    Feature: parallel run case in _test_parallel_op_expand_dims
    Description:
        1. test_expanddims_all_replicated_6 — ExpandDims all replicated
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_expanddims_all_replicated_6", worker_num=8, local_worker_num=8, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_expand_dims_group7():
    """
    Feature: parallel run case in _test_parallel_op_expand_dims
    Description:
        1. test_expanddimsview_data_parallel_1 — ExpandDimsView data parallel
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_expanddimsview_data_parallel_1", worker_num=8, local_worker_num=8, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_expand_dims_group8():
    """
    Feature: parallel run case in _test_parallel_op_expand_dims
    Description:
        1. test_expanddimsview_model_parallel_2 — ExpandDimsView model parallel
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_expanddimsview_model_parallel_2", worker_num=8, local_worker_num=8, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_expand_dims_group9():
    """
    Feature: parallel run case in _test_parallel_op_expand_dims
    Description:
        1. test_expanddimsview_hybrid_parallel_3 — ExpandDimsView hybrid parallel
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_expanddimsview_hybrid_parallel_3", worker_num=8, local_worker_num=8, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_expand_dims_group10():
    """
    Feature: parallel run case in _test_parallel_op_expand_dims
    Description:
        1. test_expanddimsview_insert_middle_4 — ExpandDimsView insert axis in middle
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_expanddimsview_insert_middle_4", worker_num=8, local_worker_num=8, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_expand_dims_group11():
    """
    Feature: parallel run case in _test_parallel_op_expand_dims
    Description:
        1. test_expanddimsview_negative_axis_5 — ExpandDimsView negative axis
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_expanddimsview_negative_axis_5", worker_num=8, local_worker_num=8, glog_v=2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_expand_dims_group12():
    """
    Feature: parallel run case in _test_parallel_op_expand_dims
    Description:
        1. test_expanddimsview_all_replicated_6 — ExpandDimsView all replicated
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_expanddimsview_all_replicated_6", worker_num=8, local_worker_num=8, glog_v=2),
    ])
