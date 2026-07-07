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
"""msrun launchers for pure fully_shard precision ST (MindSpore).

Cases compare in-process (no external checkpoint / MNIST). Each launcher runs one
parallel_run wave that fills the 8-card budget; parallel_run reports failures per case
name, so grouping cases keeps per-case localization.
"""
import os

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import MindSporeCase, parallel_run

_PRECISION_IMPL = os.path.join(os.path.dirname(__file__), "_fully_shard_precision.py")
_LIST_PRECISION_IMPL = os.path.join(os.path.dirname(__file__), "_precision_fully_shard_list.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_ms_fully_shard_precision_on_dp_mesh():
    """
    Feature: fully_shard precision on a 1D dp mesh.
    Description: Run the gradient-accumulation, recompute, comm_fusion, and combined cases in one 8-card wave.
    Expectation: Every case's per-rank loss and gradient shards match the single-card reference.
    """
    parallel_run([
        MindSporeCase(_PRECISION_IMPL, name, worker_num=2, local_worker_num=2)
        for name in (
            "test_ms_fully_shard_with_gradient_accumulation",
            "test_ms_fully_shard_with_recompute",
            "test_ms_fully_shard_with_comm_fusion",
            "test_ms_fully_shard_with_recompute_and_comm_fusion",
        )
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_ms_fully_shard_precision_on_hsdp_mesh():
    """
    Feature: fully_shard precision on a 2D HSDP (dp x op) mesh.
    Description: Run the recompute and recompute+comm_fusion cases in one 8-card wave.
    Expectation: Every case's per-rank loss and gradient shards match the single-card reference.
    """
    parallel_run([
        MindSporeCase(_PRECISION_IMPL, name, worker_num=4, local_worker_num=4)
        for name in (
            "test_ms_hsdp_with_recompute",
            "test_ms_hsdp_with_recompute_and_comm_fusion",
        )
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_ms_fully_shard_precision_with_list_units():
    """
    Feature: fully_shard(list) precision with reshard_after_forward=False.
    Description: Run the grouped-list-unit case and its prefetch+recompute variant in one 8-card wave.
    Expectation: Loss and dense1 gradient shard match the single-card reference.
    """
    parallel_run([
        MindSporeCase(_LIST_PRECISION_IMPL, name, worker_num=4, local_worker_num=4)
        for name in (
            "test_ms_fully_shard_list_unit",
            "test_ms_fully_shard_list_unit_with_recompute",
        )
    ])
