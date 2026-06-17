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
"""Chained matmul Partial propagation ST test runner."""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase

IMPL_FILE = str(Path(__file__).resolve().parent / "_test_parallel_matmul_partial_propagation.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_matmul_partial_propagation():
    """
    Feature: Chained matmul Partial propagation — matmul and linear under TP and DP×TP.
    Description:
        test_chained_matmul_partial_propagation_tp2: (x @ A.T) @ (B.T) under TP=2.
        test_chained_matmul_dp_tp_partial_propagation: (x @ A.T) @ (B.T) under dp2×tp2.
        test_linear_partial_propagation_tp2: matmul → ops.dense chain under TP=2.
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(
            IMPL_FILE, "test_chained_matmul_partial_propagation_tp2",
            worker_num=2, local_worker_num=2, glog_v=2,
        ),
        MindSporeCase(
            IMPL_FILE, "test_chained_matmul_dp_tp_partial_propagation",
            worker_num=4, local_worker_num=4, glog_v=2,
        ),
        MindSporeCase(
            IMPL_FILE, "test_linear_partial_propagation_tp2",
            worker_num=2, local_worker_num=2, glog_v=2,
        ),
    ])
