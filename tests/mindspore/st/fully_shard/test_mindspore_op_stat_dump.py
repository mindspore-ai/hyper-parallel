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
"""Launch MindSpore communication dump validation on FSDP + TP-MoE + Muon."""

from tests.common.mark_utils import arg_mark
from tests.mindspore.st.utils import msrun_case

_FILE_NAME = "_test_mindspore_op_stat_dump.py"


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_fsdp_tp_moe_muon_async_dump_matches_sync() -> None:
    """
    Feature: Deferred MindSpore communication output dump.
    Description: Run FSDP + TP-MoE + test-local Muon on four ranks, then compare
        synchronous and asynchronous AllReduce output statistics.
    Expectation: The wait-hook output CRC32 and L2 norm match the synchronous reference.
    """
    msrun_case(
        3,
        _FILE_NAME,
        "test_fsdp_tp_moe_muon_async_dump_matches_sync",
        18634,
        worker_num=4,
        local_worker_num=4,
    )
