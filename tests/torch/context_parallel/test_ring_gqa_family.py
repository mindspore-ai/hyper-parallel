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
"""Launcher for a two-rank Ring GQA NPU forward/backward comparison."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark

_WORKER = str(Path(__file__).with_name("_test_ring_gqa_family.py"))


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_ring_gqa_family_npu_2_cards() -> None:
    """Compare Ring GQA output, input gradients and reduced parameter gradients."""
    torchrun_case(_WORKER, "test_ring_gqa_family_bfloat16", num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_ring_gqa_ulysses_ring_npu_4_cards() -> None:
    """Compare the Ulysses + Ring member against full-sequence GQA."""
    torchrun_case(_WORKER, "test_ring_gqa_ulysses_ring_bfloat16", num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_ring_gqa_packed_npu_2_cards() -> None:
    """Compare packed TND Ring GQA across a document boundary on two ranks."""
    torchrun_case(_WORKER, "test_ring_gqa_packed_tnd_bfloat16", num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_ring_gqa_pruned_zigzag_npu_2_cards() -> None:
    """Compare document-pruned, zigzag-balanced Ring GQA on two ranks."""
    torchrun_case(_WORKER, "test_ring_gqa_pruned_zigzag_bfloat16", num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_ring_gqa_pruned_zigzag_npu_4_cards() -> None:
    """Exercise multi-hop document-subring transport and gradient return."""
    torchrun_case(_WORKER, "test_ring_gqa_pruned_zigzag_bfloat16", num_proc=4)
