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
"""Pytest launchers for TP + FSDP / TP + CP hybrid parallelism NPU distributed tests.

Uses the real ``ColwiseParallel`` / ``RowwiseParallel`` combined with ``fully_shard``
and ``ContextParallel`` to verify hybrid parallel composition on NPU hardware.

Port allocation:
  10600–10601  4-card TP+FSDP (forward + backward)
  10602–10603  8-card TP+CP (forward + backward)
"""
from pathlib import Path

import pytest

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


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_tp_cp_hybrid_4card():
    """
    Feature: parallel_run launcher for 4-card TP + CP hybrid tests
    Description:
        1. test_tp_cp_transformer_forward_precision_npu
    Expectation: Run success.

    Skip reason: ContextParallel Ulysses mode assumes the attention module
    receives separate Q, K, V as positional args (qkv_indices=(0,1,2)),
    but SimpleAttention.forward(self, x) takes a single input and computes
    Q/K/V internally.  Additionally, ``parallelize_module(..., {"": style})``
    does not match any child module (fnmatch(name, "") is always False).
    Tracked in design issue docs/tensor_parallel_cp_design_issue.md.
    """
    pytest.skip(
        "ContextParallel Ulysses mode incompatible with SimpleAttention structure; "
        "see docs/tensor_parallel_cp_design_issue.md"
    )
    _run_group(
        ("test_tp_cp_transformer_forward_precision_npu", 10602, 4),
    )
