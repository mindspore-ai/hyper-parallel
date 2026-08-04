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
"""Launch MindSpore TP/EP + FSDP backward overlap performance ST."""
import os

from tests.common.mark_utils import arg_mark
from tests.common.distributed_launcher import msrun_case

_FILE_NAME = os.path.join(
    os.path.dirname(__file__),
    "_test_tp_fsdp_backward_overlap_perf.py",
)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_ms_pure_hsdp_backward_overlap_perf_baseline():
    """
    Feature: pure HSDP backward overlap performance baseline (MindSpore).
    Description: Launch 8-card msrun for per-layer HSDP (2x4) backward timing baseline.
    Expectation: Run success.
    """
    msrun_case(
        2,
        _FILE_NAME,
        "test_ms_pure_hsdp_backward_overlap_perf_baseline",
        18538,
        worker_num=8,
        local_worker_num=8,
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_ms_tp_fsdp_2d_backward_overlap_perf():
    """
    Feature: TP + FSDP (2D mesh) backward overlap performance (MindSpore).
    Description: Launch 8-card msrun for TP-sharded model + fully_shard on DP mesh (4x2).
    Expectation: Run success.
    """
    msrun_case(
        2,
        _FILE_NAME,
        "test_ms_tp_fsdp_2d_backward_overlap_perf",
        18539,
        worker_num=8,
        local_worker_num=8,
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_ms_hsdp_tp_3d_backward_overlap_perf():
    """
    Feature: HSDP + TP (3D mesh) backward overlap performance (MindSpore).
    Description: Launch 8-card msrun for HSDP+TP composite mesh (2x2x2) backward timing
        and output_buffer copy-path diagnostics.
    Expectation: Run success.
    """
    msrun_case(
        2,
        _FILE_NAME,
        "test_ms_hsdp_tp_3d_backward_overlap_perf",
        18540,
        worker_num=8,
        local_worker_num=8,
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_ms_dp_tp_ep_3d_backward_overlap_perf():
    """
    Feature: DP + TP + EP + FSDP backward overlap performance (MindSpore).
    Description: Launch 8-card msrun for (dp, tp, ep)=(2, 2, 2) composite backward timing
        and output_buffer copy-path diagnostics.
    Expectation: Run success.
    """
    msrun_case(
        2,
        _FILE_NAME,
        "test_ms_dp_tp_ep_3d_backward_overlap_perf",
        18541,
        worker_num=8,
        local_worker_num=8,
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_ms_hsdp_tp_vs_pure_hsdp_backward_perf():
    """
    Feature: TP + FSDP backward overlap slowdown guard (MindSpore).
    Description: Launch 8-card msrun comparing pure HSDP vs HSDP+TP backward latency and
        reporting output_buffer copy-path counters for the review scenario.
    Expectation: Run success.
    """
    msrun_case(
        2,
        _FILE_NAME,
        "test_ms_hsdp_tp_vs_pure_hsdp_backward_perf",
        18542,
        worker_num=8,
        local_worker_num=8,
    )
