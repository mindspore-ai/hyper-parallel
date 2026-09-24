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
"""Pytest launcher for ExpertParallel token dispatcher performance comparison.

Port allocation:
  10520  EP dispatcher performance: AllToAllTokenDispatcher vs DeredundencyTokenDispatcher
"""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase


_WORKER = str(Path(__file__).resolve().parent / "_test_ep_dispatcher_performance.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="unessential")
def test_ep_dispatcher_performance_compare():
    """
    Feature: ExpertParallel token dispatcher performance comparison.
    Description:
        Launches a 4-card EP MoE workload and compares average forward+backward
        step time for ``all_to_all`` and ``deredundency`` token dispatchers.
    Expectation: Run success and print per-dispatcher average step time on rank0.
    """
    parallel_run([
        TorchCase(_WORKER, "test_ep_dispatcher_performance_compare_npu", 10520, 4),
    ])
