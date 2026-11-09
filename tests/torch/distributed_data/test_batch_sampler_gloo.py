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
"""Thin launcher for native BatchSampler collective tests."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_native_batch_sampler_dp2_tp2_gloo() -> None:
    """Feature: Native HP BatchSampler integration.
    Description: Exercise DP2/TP2 online and metadata loading with both shuffle policies and prefetch.
    Expectation: Every round preserves native occurrences and checkpoint replay never skips a prefetched batch.
    """
    torchrun_case(
        file_name=str(Path(__file__).with_name("_test_batch_sampler_gloo.py")),
        case_name="test_native_batch_sampler_dp2_tp2_gloo", num_proc=4,
    )


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_changed_gbs_dp2_tp2_gloo() -> None:
    """Feature: Variable GBS and DP restoration.
    Description: Restore thirty consumed samples with new GBS/DP and replay the resulting offset cursor.
    Expectation: Constructors and TP peers preserve progress, full steps and the unconsumed global suffix.
    """
    torchrun_case(
        file_name=str(Path(__file__).with_name("_test_batch_sampler_gloo.py")),
        case_name="test_changed_gbs_dp2_tp2_gloo", num_proc=4,
    )


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_accumulation_window_dp2_tp2_gloo() -> None:
    """Feature: Accumulation-window native loading.
    Description: Balance windows with three or ten accumulation rounds using DP2/TP2 and replay checkpoints.
    Expectation: Samples remain inside their windows; both data paths and repacking replay exactly.
    """
    torchrun_case(
        file_name=str(Path(__file__).with_name("_test_batch_sampler_gloo.py")),
        case_name="test_accumulation_window_dp2_tp2_gloo", num_proc=4,
    )


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_host_window_device_queue_dp2_tp2_gloo() -> None:
    """Feature: Host windows and caller-side device prefetch.
    Description: Exercise DP2/TP2 loading with independent H2D lookahead and current-batch broadcasts.
    Expectation: Collective ordering, EOF delivery and checkpoint replay remain correct.
    """
    torchrun_case(
        file_name=str(Path(__file__).with_name("_test_batch_sampler_gloo.py")),
        case_name="test_host_window_device_queue_dp2_tp2_gloo", num_proc=4,
    )


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_trainer_batch_layout_dp2_tp2_gloo() -> None:
    """Feature: Trainer tensor-only model broadcast.
    Description: Load GPT, Online text and explicitly declared Omni batches on DP2/TP2.
    Expectation: Existing text collators and configured multimodal layouts preserve all model inputs.
    """
    torchrun_case(
        file_name=str(Path(__file__).with_name("_test_batch_sampler_gloo.py")),
        case_name="test_trainer_batch_layout_dp2_tp2_gloo", num_proc=4,
    )


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_full_pack_repair_dp2_tp2_gloo() -> None:
    """Feature: Feasible full-pack repair.
    Description: Repair DP2 packs with accumulation one or three, repeated records, and both plan modes.
    Expectation: Tokens, targets, boundaries and sample occurrences survive routing and checkpoint replay.
    """
    torchrun_case(
        file_name=str(Path(__file__).with_name("_test_batch_sampler_gloo.py")),
        case_name="test_full_pack_repair_dp2_tp2_gloo", num_proc=4,
    )


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_native_balance_logging_dp2_tp2_gloo() -> None:
    """Feature: Native balancing INFO logs.
    Description: Exercise both planning modes, metadata sources and a nonzero planner rank.
    Expectation: Only the planner logs consumed windows, with correct accepted and rejected costs.
    """
    torchrun_case(
        file_name=str(Path(__file__).with_name("_test_batch_sampler_gloo.py")),
        case_name="test_native_balance_logging_dp2_tp2_gloo", num_proc=4,
    )


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_single_distribution_dp2_cp2_tp2_gloo() -> None:
    """Feature: Single model-data distribution with DP, TP and CP.
    Description: Run native and balanced Text/Omni consumption across DP2/CP2/TP2.
    Expectation: CP fields, boundaries and EOF survive without duplicate TP broadcasts.
    """
    torchrun_case(
        file_name=str(Path(__file__).with_name("_test_batch_sampler_gloo.py")),
        case_name="test_single_distribution_dp2_cp2_tp2_gloo", num_proc=8,
    )
