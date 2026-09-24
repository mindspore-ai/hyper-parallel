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
"""Pytest launchers for MC2 fused TP Linear NPU distributed tests.

Port allocation (``sum(num_proc) <= 8`` per wave):

  Wave 1 (level0): 10900–10901, col/row fwd+bwd × 2 ranks
  Wave 2 (level1): 10910–10912, mlp fwd+bwd / fp16 / seq-dim1 × 2 ranks
"""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

_WORKER = str(Path(__file__).resolve().parent / "_test_mc2_distributed.py")


def _run_group(*cases):
    parallel_run([
        TorchCase(_WORKER, case_name, master_port, num_proc)
        for case_name, master_port, num_proc in cases
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_mc2_two_card_wave_one():
    """
    Feature: MC2 colwise/rowwise forward+backward on 2 NPUs
    Description:
        1. test_mc2_colwise_linear_fwd_bwd_precision_npu
        2. test_mc2_rowwise_linear_fwd_bwd_precision_npu
    Expectation: Run success.
    """
    _run_group(
        ("test_mc2_colwise_linear_fwd_bwd_precision_npu", 10900, 2),
        ("test_mc2_rowwise_linear_fwd_bwd_precision_npu", 10901, 2),
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_mc2_two_card_wave_two():
    """
    Feature: MC2 MLP / fp16 / seq-dim1 extended precision on 2 NPUs
    Description:
        1. test_mc2_mlp_col_row_fwd_bwd_precision_npu
        2. test_mc2_colwise_linear_forward_fp16_npu
        3. test_mc2_colwise_seq_dim1_forward_precision_npu
    Expectation: Run success.
    """
    _run_group(
        ("test_mc2_mlp_col_row_fwd_bwd_precision_npu", 10910, 2),
        ("test_mc2_colwise_linear_forward_fp16_npu", 10911, 2),
        ("test_mc2_colwise_seq_dim1_forward_precision_npu", 10912, 2),
    )
