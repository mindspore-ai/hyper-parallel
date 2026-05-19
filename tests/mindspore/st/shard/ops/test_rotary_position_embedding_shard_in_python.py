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
"""Test runner for RotaryPositionEmbedding distributed ST (MindSpore)."""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase

IMPL_FILE = str(
    Path(__file__).resolve().parent / "rotary_position_embedding_shard_in_python.py"
)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_rpe_ms_group1():
    """
    Feature: parallel run case in rotary_position_embedding_shard_in_python (2-card, 1-D mesh)
    Description:
        1. test_rpe_replicated — all replicated on dp mesh, fwd vs standalone
        2. test_rpe_dp_b — B-dim DP (x Shard(0), cos/sin broadcast Replicate),
           fwd + grad_x vs standalone
        3. test_rpe_tp_n — N-dim TP (x/cos/sin Shard(1)), fwd vs standalone
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_rpe_replicated", 19300, 2, 2, 2),
        MindSporeCase(IMPL_FILE, "test_rpe_dp_b", 19301, 2, 2, 2),
        MindSporeCase(IMPL_FILE, "test_rpe_tp_n", 19302, 2, 2, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_rpe_ms_group2():
    """
    Feature: parallel run case in rotary_position_embedding_shard_in_python (4-card, 2-D mesh)
    Description:
        1. test_rpe_dp_tp — 2-D (dp=2, tp=2) mesh; x/cos/sin (Shard(0), Shard(1));
           fwd + backward (grad_x/cos/sin) vs standalone
        2. test_rpe_dp_sp — 2-D (dp=2, sp=2) mesh; x (Shard(0), Shard(2)),
           cos/sin broadcast (1,1,S,D) with (Replicate, Shard(2)); fwd vs standalone
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_rpe_dp_tp", 19303, 4, 4, 2),
        MindSporeCase(IMPL_FILE, "test_rpe_dp_sp", 19304, 4, 4, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_rpe_ms_group3():
    """
    Feature: parallel run case in rotary_position_embedding_shard_in_python (4-card, 2-D mesh)
    Description:
        1. test_rpe_tp_sp — 2-D (tp=2, sp=2) mesh; x/cos/sin full shape (Shard(1), Shard(2));
           fwd vs standalone
        2. test_rpe_dp_tp_cos_full — 2-D (dp=2, tp=2) mesh; x (Shard(0), Shard(1)),
           cos/sin broadcast (1,1,S,D) fully replicated; validates broadcast semantics
           when x is sharded on both B and N; fwd vs standalone
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_rpe_tp_sp", 19305, 4, 4, 2),
        MindSporeCase(IMPL_FILE, "test_rpe_dp_tp_cos_full", 19306, 4, 4, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_rpe_ms_group4():
    """
    Feature: parallel run case in rotary_position_embedding_shard_in_python (8-card, 3-D mesh)
    Description:
        1. test_rpe_dp_tp_sp — 3-D (dp=2, tp=2, sp=2) mesh; x (Shard(0), Shard(1), Shard(2)),
           cos/sin broadcast (1,1,S,D) with (Replicate, Replicate, Shard(2));
           B/N/S all sharded simultaneously; fwd vs standalone
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(IMPL_FILE, "test_rpe_dp_tp_sp", 19307, 8, 8, 2),
    ])
