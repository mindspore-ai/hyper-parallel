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
"""Pytest launchers for TP + FSDP hybrid parallelism NPU distributed tests.

Uses the real ``ColwiseParallel`` / ``RowwiseParallel`` with ``fully_shard`` on NPU.

Port allocation:
  10600–10601  4-card TP+FSDP (forward + backward)
"""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

_WORKER = str(Path(__file__).resolve().parent / "_test_tp_hybrid_distributed.py")


def _run_group(*cases):
    """Launch a group of worker cases with ``parallel_run``."""
    parallel_run([
        TorchCase(_WORKER, case_name, master_port, num_proc)
        for case_name, master_port, num_proc in cases
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_tp_fsdp_hybrid_4card():
    """
    Feature: parallel_run launcher for 4-card TP + FSDP hybrid tests
    Description:
        1. test_tp_fsdp_mlp_forward_precision_npu
        2. test_tp_fsdp_mlp_backward_gradient_npu
    Expectation: Run success.
    """
    _run_group(
        ("test_tp_fsdp_mlp_forward_precision_npu", 10600, 4),
        ("test_tp_fsdp_mlp_backward_gradient_npu", 10601, 4),
    )
