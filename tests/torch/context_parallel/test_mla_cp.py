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
"""Lightweight launchers for dense MLA CP; no framework imports at collection."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark

_WORKER = str(Path(__file__).with_name("_test_mla_cp.py"))


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_mla_npu_8_cards() -> None:
    """
    Feature: MLA Ulysses CP8 on NPU.
    Description: Run both strategies with BF16 native FA, packed sequences and explicit masks.
    Expectation: Outputs and gradients match the native non-CP reference within the existing BF16 tolerances.
    """
    torchrun_case(_WORKER, "test_mla_npu_bfloat16", num_proc=8)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_mla_tp_cp_npu_8_cards() -> None:
    """
    Feature: MLA TP/CP composition with FSDP and SP.
    Description: Run TP2 CP4 and two decoder layers under DP2 TP2 CP2 with FSDP and SP on and off.
    Expectation: Local outputs, global losses and reconstructed parameter gradients match the reference.
    """
    torchrun_case(_WORKER, "test_mla_tp_cp_npu_bfloat16", num_proc=8)
    torchrun_case(_WORKER, "test_mla_model_dp2_tp2_cp2_fsdp", num_proc=8)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="onecard", essential_mark="essential")
def test_mla_npu_reference() -> None:
    """
    Feature: Independent single-card fused RoPE and attention validation.
    Description: Compare BF16 and FP16 kernels with explicit CPU math, including packed and fully masked rows.
    Expectation: Kernel outputs and gradients satisfy the existing elementwise tolerances.
    """
    torchrun_case(_WORKER, "test_mla_npu_single_card", num_proc=1)
