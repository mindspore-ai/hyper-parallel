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
"""Launchers for distributed ``qwen3_vl_moe`` VL trainer smoke cases."""
import json
import math
import tempfile
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

_WORKER = str(Path(__file__).resolve().parent / "_test_qwen3_vl_moe_vl_trainer.py")
_MAX_PARALLEL_DEVICES = 8


def _capture_loss_path(master_port: int) -> Path:
    """Return the per-port path used to exchange a captured loss."""
    return Path(tempfile.gettempdir()) / f"hp_qwen3_vl_moe_loss_{master_port}.json"


def _run_case_waves(cases):
    """Run cases in disjoint batches that fit the 8-device smoke allocation."""
    wave = []
    wave_devices = 0
    for case in cases:
        if wave and wave_devices + case.num_proc > _MAX_PARALLEL_DEVICES:
            parallel_run(wave, global_num_proc=_MAX_PARALLEL_DEVICES)
            wave = []
            wave_devices = 0
        wave.append(case)
        wave_devices += case.num_proc
    if wave:
        parallel_run(wave, global_num_proc=_MAX_PARALLEL_DEVICES)


def _collect_captured_losses(specs):
    """Run loss-capture cases in disjoint batches and return their losses."""
    cases = [
        TorchCase(_WORKER, worker_case, master_port, num_proc)
        for worker_case, master_port, num_proc in specs
    ]
    for _, master_port, _ in specs:
        _capture_loss_path(master_port).unlink(missing_ok=True)

    _run_case_waves(cases)

    losses = []
    for _, master_port, _ in specs:
        capture_path = _capture_loss_path(master_port)
        try:
            with capture_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            losses.append(float(payload["loss"]))
        finally:
            capture_path.unlink(missing_ok=True)
    return losses


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_qwen3_vl_moe_vl_dummy_smoke_group1():
    """
    Feature: grouped VL trainer smoke cases for ``qwen3_vl_moe``.
    Description: Run independent smoke cases in one device-isolated wave.
    Expectation: Run success.
    """
    _run_case_waves([
        TorchCase(_WORKER, "test_qwen3_vl_moe_vl_dummy_smoke_2card_dp", 13901, 2),
        TorchCase(_WORKER, "test_qwen3_vl_moe_vl_dummy_smoke_1card", 13900, 1),
        TorchCase(_WORKER, "test_qwen3_vl_moe_vl_dummy_smoke_2card_vision_dp1", 13908, 2),
    ])


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_qwen3_vl_moe_vl_dummy_smoke_group2():
    """
    Feature: grouped visual Encoder CP smoke cases for ``qwen3_vl_moe``.
    Description: Run independent smoke cases in one device-isolated wave.
    Expectation: Run success.
    """
    _run_case_waves([
        TorchCase(_WORKER, "test_qwen3_vl_moe_vl_dummy_smoke_2card_vision_cp_colossal", 13902, 2),
        TorchCase(_WORKER, "test_qwen3_vl_moe_vl_dummy_smoke_2card_vision_cp_ulysses", 13903, 2),
        TorchCase(_WORKER, "test_qwen3_vl_moe_vl_dummy_smoke_2card_vision_async_cp_colossal", 13914, 2),
    ])


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_qwen3_vl_moe_vl_dummy_loss_alignment():
    """
    Feature: Qwen3-VL-MoE first-step loss alignment regression.
    Description: Compare independent distributed modes using batched captures.
    Expectation: Compared losses stay within tolerance.
    """
    losses = _collect_captured_losses([
        ("test_qwen3_vl_moe_vl_dummy_capture_loss_2card_dp", 13904, 2),
        ("test_qwen3_vl_moe_vl_dummy_capture_loss_2card_vision_dp1", 13909, 2),
        ("test_qwen3_vl_moe_vl_dummy_capture_loss_2card_vision_cp_colossal", 13905, 2),
        ("test_qwen3_vl_moe_vl_dummy_capture_loss_2card_vision_cp_ulysses", 13906, 2),
        ("test_qwen3_vl_moe_vl_dummy_capture_loss_2card_vision_async_cp_colossal", 13915, 2),
    ])
    baseline_loss = losses[0]
    vision_dp1_loss = losses[1]
    colossal_loss = losses[2]
    ulysses_loss = losses[3]
    async_colossal_loss = losses[4]

    tolerance = 1.0e-5
    assert math.isclose(
        baseline_loss, vision_dp1_loss, rel_tol=1.0e-6, abs_tol=tolerance
    ), (
        "Baseline DP/FSDP and visual DP first-step losses diverged: "
        f"{baseline_loss} vs {vision_dp1_loss}"
    )
    assert math.isclose(
        baseline_loss, colossal_loss, rel_tol=1.0e-6, abs_tol=tolerance
    ), (
        "Baseline DP/FSDP and visual CP (Pure Colossal) first-step losses diverged: "
        f"{baseline_loss} vs {colossal_loss}"
    )
    assert math.isclose(
        baseline_loss, ulysses_loss, rel_tol=1.0e-6, abs_tol=tolerance
    ), (
        "Baseline DP/FSDP and visual CP (Pure Ulysses) first-step losses diverged: "
        f"{baseline_loss} vs {ulysses_loss}"
    )
    assert math.isclose(
        baseline_loss, async_colossal_loss, rel_tol=1.0e-6, abs_tol=tolerance
    ), (
        "Baseline DP/FSDP and visual async CP (Pure Colossal) first-step losses diverged: "
        f"{baseline_loss} vs {async_colossal_loss}"
    )
