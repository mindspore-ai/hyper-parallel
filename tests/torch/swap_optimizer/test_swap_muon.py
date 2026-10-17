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
"""Launch swap Muon system tests."""

from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

SWAP_MUON = str(Path(__file__).resolve().parent / "swap_muon.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_swap_muon_1():
    """Test swap muon."""
    parallel_run([
        TorchCase(SWAP_MUON, "test_swap_muon_parameter_align", None, 1),
        TorchCase(SWAP_MUON, "test_swap_muon_checkpoint_host_state", None, 1),
        TorchCase(SWAP_MUON, "test_swap_muon_first_step_is_lazy_and_offloaded", None, 1),
        TorchCase(SWAP_MUON, "test_swap_muon_packed_swap", None, 1),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_swap_muon_2():
    """Test swap muon."""
    parallel_run([
        TorchCase(SWAP_MUON, "test_swap_muon_hsdp_logical_schedule_4p", None, 4),
        TorchCase(SWAP_MUON, "test_swap_muon_hsdp_packed_logical_schedule_4p", None, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_swap_muon_3():
    """Test swap muon."""
    parallel_run([
        TorchCase(SWAP_MUON, "test_swap_muon_swap_times_parameter_align", None, 1),
        TorchCase(SWAP_MUON, "test_swap_muon_post_update_fn_align", None, 1),
        TorchCase(SWAP_MUON, "test_swap_muon_mixed_precision_reset", None, 1),
        TorchCase(SWAP_MUON, "test_swap_muon_packed_mixed_dtype", None, 1),
        TorchCase(SWAP_MUON, "test_swap_muon_packed_mixed_precision_reset", None, 1),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_swap_muon_4():
    """Test swap muon."""
    parallel_run([
        TorchCase(SWAP_MUON, "test_swap_muon_hsdp_mixed_precision_reset_4p", None, 4),
        TorchCase(SWAP_MUON, "test_swap_muon_hsdp_packed_mixed_precision_reset_4p", None, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_swap_muon_hsdp_checkpoint_8p():
    """
    Feature: HSDP checkpoint lifecycle for swap Muon.
    Description:
        1. test_swap_muon_hsdp_checkpoint_8p (replicate=2, shard=4 mesh)
    Expectation: Replicated state broadcasts and returns to host storage cleanly.
    """
    parallel_run([
        TorchCase(SWAP_MUON, "test_swap_muon_hsdp_checkpoint_8p", None, 8),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_swap_muon_hsdp_logical_schedule_8p():
    """Verify global partitions, real collective order and resumed numerical parity."""
    parallel_run([
        TorchCase(SWAP_MUON, "test_swap_muon_hsdp_logical_schedule_8p", None, 8),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_swap_muon_hsdp_packed_logical_schedule_8p():
    """Verify packed HSDP/DTensor state and checkpoint continuation on eight devices."""
    parallel_run([
        TorchCase(SWAP_MUON, "test_swap_muon_hsdp_packed_logical_schedule_8p", None, 8),
    ])
