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
"""Pytest launcher for ``NoParallel`` NPU distributed tests.

Single ``parallel_run`` wave: two cases × 2 ranks (ports 10560–10561, sum = 4).

Worker implementations live in ``_test_no_parallel_distributed.py`` and compare
against a **CPU single-device** reference (float32).

Coverage:

1. Replicated Linear forward precision
2. Input redistribution from Shard (SequenceParallel → NoParallel)

Dropped scenarios (covered elsewhere or redundant):

* LayerNorm forward — same replicated forward path as Linear; see
  ``test_tp_sequence_parallel_distributed`` / UT ``test_no_parallel.py``.
* Output redistribution, backward, and SP→NoParallel→Row composition — exercised
  in ``test_tp_styles_distributed`` and ``tests/ut/core/tensor_parallel``.
"""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

_WORKER = str(Path(__file__).resolve().parent / "_test_no_parallel_distributed.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_no_parallel_two_card():
    """
    Feature: two ``NoParallel`` scenarios on **2** NPU ranks in one wave
    Description:
        1. test_no_parallel_linear_forward_precision_npu
        2. test_no_parallel_redistribute_sharded_input_npu
    Expectation: both worker cases succeed.
    """
    parallel_run([
        TorchCase(_WORKER, "test_no_parallel_linear_forward_precision_npu", 10560, 2),
        TorchCase(_WORKER, "test_no_parallel_redistribute_sharded_input_npu", 10561, 2),
    ])
