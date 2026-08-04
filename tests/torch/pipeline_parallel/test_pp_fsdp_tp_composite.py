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
"""Pytest launcher for Llama3 PP + FSDP + TP composite (1F1B) vs serial reference.

Follows ``tests/mindspore/st/pipeline_parallel/test_pp_composite.py`` launcher layout:
one ``torchrun`` worker on 8 NPUs exercising ``Schedule1F1B`` with per-step loss parity
against a full-model serial reference (``examples/torch/llama3`` Llama3 layout,
nested ``fully_shard``, ``micro_batch_num=4``).
"""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.distributed_launcher import torchrun_case

_WORKER = str(Path(__file__).resolve().parent / "_test_pp_fsdp_tp_composite.py")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_pp_fsdp_tp_1f1b_composite():
    """
    Feature: Llama3 ``PP + FSDP + TP`` 1F1B composite vs full-model serial reference.
    Description:
        Launch :func:`test_pp_fsdp_tp_1f1b_composite_matches_reference` on 8 ranks with
        mesh ``(pp=2, fsdp=2, tp=2)``.  Each of 10 steps compares global sum-loss from
        the last PP stage against a single-card ``Llama3Model`` baseline before
        ``optimizer.step()``, with a full 10-step trajectory check.
    Expectation: Worker exits with status 0 on every rank.
    """
    torchrun_case(
        _WORKER,
        "test_pp_fsdp_tp_1f1b_composite_matches_reference",
        master_port=13820,
        num_proc=8,
    )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_pp_fsdp_tp_1f1b_composite_100_steps():
    """
    Feature: Llama3 ``PP + FSDP + TP`` 1F1B composite vs single-card baseline (100 steps).
    Description:
        Launch :func:`test_pp_fsdp_tp_1f1b_composite_matches_reference_100_steps` on 8 ranks
        with mesh ``(pp=2, fsdp=2, tp=2)``.  Each of 100 steps asserts global sum-loss against
        a single-card ``Llama3Model`` baseline (``rtol=2e-3``, ``atol=2.0``); rank 0 prints
        abs/rel error drift statistics after all steps pass.
    Expectation: Worker exits with status 0 on every rank.
    """
    torchrun_case(
        _WORKER,
        "test_pp_fsdp_tp_1f1b_composite_matches_reference_100_steps",
        master_port=13821,
        num_proc=8,
    )
