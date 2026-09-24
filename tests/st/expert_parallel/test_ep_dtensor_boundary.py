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
"""Pytest launchers for EP DTensor boundary integration tests.

Each test function uses ``parallel_run`` to spawn ``torchrun`` workers that
execute the actual distributed logic in ``_test_ep_dtensor_boundary.py``.
"""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase


_WORKER = str(Path(__file__).resolve().parent / "_test_ep_dtensor_boundary.py")


def _run_group(*cases):
    """Launch a group of worker cases with ``parallel_run``.

    Args:
        *cases: Each element is a tuple ``(case_name, master_port, num_proc)``.
    """
    parallel_run([
        TorchCase(_WORKER, case_name, master_port, num_proc)
        for case_name, master_port, num_proc in cases
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="unessential")
def test_ep_dtensor_boundary():
    """
    Feature: EP DTensor boundary integration tests.
    Description:
        Two cases run in parallel within a single group (6 cards total):
          - EP-only plain tensor regression (2 cards)
          - SP(2) x EP(2) DTensor Shard(1) boundary forward/backward (4 cards)
        The SP x EP test passes DTensor to MoE via PrepareModuleInputOutput
        (use_local_input=True), and MoE outputs DTensor (use_local_output=False,
        output_layouts=Shard(1)). Forward and backward numerical correctness
        vs single-card baseline. Gradient is plain tensor due to Stage 1
        use_local_input=True.
    Expectation: Run success.
    """
    _run_group(
        ("test_ep_only_plain_tensor", 10520, 2),
        ("test_sp_ep_dtensor_boundary", 10522, 4),
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="unessential")
def test_ep_dtensor_boundary_sp_ep_shared_expert():
    """
    Feature: SP x EP DTensor boundary with shared expert.
    Description:
        4-card SP(2) x EP(2) with shared expert and DTensor Shard(1) input.
        Covers the ``out = out + shared_out`` path in MoE.forward where routed
        output and shared-expert output are summed as plain tensors, then
        wrapped back to DTensor Shard(1) by the post-hook.
        Forward and backward numerical correctness vs single-card baseline.
        Gradient is plain tensor due to Stage 1 use_local_input=True.
    Expectation: Run success.
    """
    _run_group(
        ("test_sp_ep_dtensor_boundary_with_shared_expert", 10524, 4),
    )
