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
"""Pytest launchers for TP + FSDP hybrid and related 4-card NPU tests.

Packs two 4-rank workers into one wave (sum ranks = 8):

  * TP + FSDP MLP forward+backward
  * SequenceParallel LayerNorm forward+backward

Port allocation: 10800 (hybrid), 10801 (sequence parallel).
"""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

_HYBRID_WORKER = str(Path(__file__).resolve().parent / "_test_tp_hybrid_distributed.py")
_SEQ_WORKER = str(Path(__file__).resolve().parent / "_test_tp_sequence_parallel_distributed.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_tp_fsdp_hybrid_4card():
    """
    Feature: 4-card TP+FSDP hybrid + SequenceParallel in one wave
    Description:
        1. test_tp_fsdp_mlp_fwd_bwd_precision_npu
        2. test_sequence_parallel_layernorm_fwd_bwd_vs_cpu_4gpu
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_HYBRID_WORKER, "test_tp_fsdp_mlp_fwd_bwd_precision_npu", 10800, 4),
        TorchCase(_SEQ_WORKER, "test_sequence_parallel_layernorm_fwd_bwd_vs_cpu_4gpu", 10801, 4),
    ])
