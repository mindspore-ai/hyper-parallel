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
"""Pytest launcher for PP + HSDP + Interleaved 1F1B composite vs single-card reference.

Spawns one ``torchrun`` worker on 8 NPUs exercising ``ScheduleInterleaved1F1B`` over a
3-D ``(pp=2, dp=2, fsdp=2)`` mesh, where the 2-D ``(dp, fsdp)`` HSDP submesh drives the
``fully_shard`` reduce (replicate all-reduce + shard reduce-scatter).  Per-step loss and
gradient parity are checked against a full-model serial reference inside the worker.
"""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.distributed_launcher import torchrun_case

_WORKER = str(Path(__file__).resolve().parent / "_test_pp_composite.py")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_pp_plus_fully_shard_vpp():
    """
    Feature: ``PP + HSDP`` Interleaved 1F1B composite vs full-model serial reference.
    Description:
        Launch :func:`test_pp_hsdp_vpp_interleaved_1f1b_matches_reference` on 8 ranks with
        mesh ``(pp=2, dp=2, fsdp=2)``.  Each step compares the HSDP-reduced global loss and
        every owned reduced gradient against a single-card full-model serial baseline,
        verifying the pipeline-driven HSDP gradient drain under ``ScheduleInterleaved1F1B``.
    Expectation: Worker exits with status 0 on every rank.
    """
    torchrun_case(
        _WORKER,
        "test_pp_hsdp_vpp_interleaved_1f1b_matches_reference",
        master_port=13830,
        num_proc=8,
    )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_pp_plus_fully_shard_vpp_dxdw_split():
    """Run Torch PP+HSDP with split input/weight backward on eight ranks."""
    torchrun_case(
        _WORKER,
        "test_pp_hsdp_vpp_dxdw_split_matches_reference",
        num_proc=8,
    )


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
