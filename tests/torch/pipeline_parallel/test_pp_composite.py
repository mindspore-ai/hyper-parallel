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
"""Gate launchers for pipeline scheduling, HSDP, and tensor parallelism."""
from pathlib import Path

from tests.common.ascend_level0_cases import (
    run_pp_batch_cold_start_and_hsdp,
    run_pp_multi_stream_cold_start_and_composites,
)
from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark

_WORKER_DIR = Path(__file__).resolve().parent
_WORKER = str(_WORKER_DIR / "_test_pp_composite.py")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_pp_batch_cold_start_and_hsdp() -> None:
    """Start batch P2P on a fresh WORLD, then verify both PP/HSDP backward schedules."""
    run_pp_batch_cold_start_and_hsdp(_WORKER_DIR)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_pp_multi_stream_cold_start_and_composites() -> None:
    """Start multi-stream P2P on its own fresh WORLD before PP8 and PP/FSDP/TP."""
    run_pp_multi_stream_cold_start_and_composites(_WORKER_DIR)


@arg_mark(
    plat_marks=["platform_ascend910b", "platform_gpu"], level_mark="level1",
    card_mark="allcards", essential_mark="essential",
)
def test_pp_hsdp_checkpoint_concurrent_overlap():
    """Validate policies with checkpointing and a genuine backward worker thread."""
    torchrun_case(_WORKER, "test_pp_hsdp_policies_checkpoint_overlap_matches_reference", num_proc=8)


@arg_mark(
    plat_marks=["platform_ascend910b", "platform_gpu"], level_mark="level1",
    card_mark="allcards", essential_mark="essential",
)
def test_pp_hsdp_gpipe_and_1f1b(monkeypatch):
    """Validate ordinary schedules on PP=4/HSDP=(1,2)."""
    monkeypatch.setenv("HSDP_VPP_PP_SIZE", "4")
    monkeypatch.setenv("HSDP_VPP_DP_SIZE", "1")
    monkeypatch.setenv("HSDP_VPP_FSDP_SIZE", "2")
    torchrun_case(_WORKER, "test_pp_hsdp_gpipe_1f1b_matches_reference", num_proc=8)


@arg_mark(
    plat_marks=["platform_ascend910b", "platform_gpu"], level_mark="level1",
    card_mark="allcards", essential_mark="essential",
)
def test_pp_hsdp_nested_units_with_checkpoint():
    """Validate an empty HSDP root over multiple parameter-management units."""
    torchrun_case(_WORKER, "test_pp_hsdp_nested_units_checkpoint_matches_reference", num_proc=8)
