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
"""test_moe.py"""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.multicore_test_env import prepare_multicore_test_environment, without_inherited_rank_environment
from tests.common.parallel_case import MindSporeCase, parallel_run

_WORKER = str(Path(__file__).resolve().parent / "mega_moe.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_mega_moe_group_tp2ep2():
    """
    Feature: MoE-FFN forward and backward operators with MindSpore (TP=2, EP=2).
    Description: Run the two 2-card precision cases concurrently on disjoint devices.
                 The forward case covers dispatch -> GMM1 -> SwiGLU -> GMM2 -> combine;
                 the backward case covers act_grad GMM -> SwiGLU_bwd -> gate_grad GMM.
    Expectation: Both cases run successfully and satisfy their configured precision tolerance.
    """
    prepare_multicore_test_environment()
    with without_inherited_rank_environment():
        parallel_run(
            [
                MindSporeCase(
                    _WORKER,
                    "test_mega_moe_tp2ep2",
                    worker_num=2,
                    local_worker_num=2,
                    glog_v=2,
                ),
                MindSporeCase(
                    _WORKER,
                    "test_mega_moe_grad_tp2ep2",
                    worker_num=2,
                    local_worker_num=2,
                    glog_v=2,
                ),
            ],
            global_num_proc=4,
        )
