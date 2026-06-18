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
from tests.mindspore.st.utils import msrun_case

_WORKER = str(Path(__file__).resolve().parent / "mega_moe.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_mega_moe_tp2ep2():
    """
    Feature: MoE-FFN forward operator with MindSpore (TP=2, EP=2).
    Description: Test mega_moe operator: dispatch -> GMM1 -> SwiGLU -> GMM2 -> combine.
                 Verifies down_proj_y against a reference implementation.
    Expectation: Run success and precision within rtol=atol=2e-3.
    """
    glog_v = 2
    case_name = "test_mega_moe_tp2ep2"
    master_port = 10200
    worker_num = 2
    msrun_case(glog_v, _WORKER, case_name, master_port, worker_num, worker_num)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_mega_moe_grad_tp2ep2():
    """
    Feature: MoE-FFN backward operator with MindSpore (TP=2, EP=2).
    Description: Test mega_moe_grad operator: act_grad GMM -> SwiGLU_bwd -> gate_grad GMM.
                 Verifies act_grad_y, grad_gate, and gate_dx against reference implementations.
    Expectation: Run success and precision within rtol=atol=2e-3.
    """
    glog_v = 2
    case_name = "test_mega_moe_grad_tp2ep2"
    master_port = 10201
    worker_num = 2
    msrun_case(glog_v, _WORKER, case_name, master_port, worker_num, worker_num)
