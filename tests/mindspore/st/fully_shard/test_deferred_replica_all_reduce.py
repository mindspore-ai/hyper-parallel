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
"""Launch the deferred replica all-reduce regression on eight Ascend cards."""
import os

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import MindSporeCase, parallel_run


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_deferred_replica_all_reduce() -> None:
    """
    Feature: HSDP deferred replica gradient synchronization.
    Description: EP-like groups, shard sizes 1/2, four different micro-batches,
        SUM/AVG and FP32 main_grad. Check analytic gradients and one all-reduce.
    Expectation: Every micro-batch contributes once and replicas stay identical.
    """
    worker = os.path.join(os.path.dirname(__file__), "_test_deferred_replica_all_reduce.py")
    parallel_run([MindSporeCase(worker, "test_deferred_replica_all_reduce", worker_num=8, local_worker_num=8)])
