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
"""Launch MindSpore fully_shard comm_fusion=False backward overlap ST."""
from tests.common.mark_utils import arg_mark
from tests.mindspore.st.utils import msrun_case

_FILE_NAME = "_test_fully_shard_backward_overlap.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_ms_hsdp_backward_overlap_functional():
    """
    Feature: comm_fusion=False HSDP backward RS/AR layer overlap (MindSpore).
    Description: Launch an 8-card msrun case on HSDP mesh (2x4) with per-layer
        fully_shard, prefetch, and comm_fusion=False. Verify fused async all-reduce
        is issued and deterministic grads match across repeated runs.
    Expectation: Run success.
    """
    msrun_case(
        2,
        _FILE_NAME,
        "test_ms_hsdp_backward_overlap_functional",
        18536,
        worker_num=8,
        local_worker_num=8,
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_ms_hsdp_backward_overlap_performance():
    """
    Feature: comm_fusion=False HSDP backward overlap performance guard (MindSpore).
    Description: Launch the same 8-card HSDP setup, warmup, then measure backward
        latency with NPU sync. Assert overlap all-reduce is exercised and backward
        median stays below a conservative CI ceiling.
    Expectation: Run success.
    """
    msrun_case(
        2,
        _FILE_NAME,
        "test_ms_hsdp_backward_overlap_performance",
        18537,
        worker_num=8,
        local_worker_num=8,
    )
