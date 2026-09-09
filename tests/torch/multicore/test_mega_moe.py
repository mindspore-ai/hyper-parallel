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

"""Lightweight launchers for public MegaMoe layer system tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.common.mark_utils import arg_mark
from tests.torch.multicore._test_env import (
    multicore_adapter_is_available,
    prepare_multicore_test_environment,
    without_inherited_rank_environment,
)
from tests.common.parallel_case import TorchCase, parallel_run
from tests.common.port_utils import allocate_port

_WORKER = str(Path(__file__).resolve().parent / "_test_mega_moe.py")
_PRECISION_WORLD_SIZE = 2
_PERFORMANCE_WORLD_SIZE = 4


def _prepare_torch_multicore_test_environment() -> None:
    """Activate multicore and require a payload built for the Torch framework."""
    prepare_multicore_test_environment()
    if not multicore_adapter_is_available():
        raise RuntimeError(
            "MegaMoe Torch ST requires a wheel or PYTHONPATH payload built with "
            "--multicore on"
        )


def _run_precision_worker(monkeypatch) -> None:
    """Run the two-rank precision worker with an isolated SHMEM endpoint."""
    _prepare_torch_multicore_test_environment()
    monkeypatch.setenv("SYMMETRIC_MEMORY_HEAP_SIZE", str(64 * 1024 * 1024))
    monkeypatch.setenv("HP_MEGA_MOE_WORLD_SIZE", str(_PRECISION_WORLD_SIZE))
    monkeypatch.setenv("SHMEM_IP_PORT", f"tcp://127.0.0.1:{allocate_port()}")
    with without_inherited_rank_environment():
        parallel_run(
            [
                TorchCase(
                    _WORKER,
                    "test_mega_moe_level0_balanced",
                    num_proc=_PRECISION_WORLD_SIZE,
                )
            ],
            global_num_proc=_PRECISION_WORLD_SIZE,
        )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_mega_moe_level0_precision(monkeypatch) -> None:
    """Compare output, dX, route dW, W1 dW, and W2 dW with common MoE."""
    _run_precision_worker(monkeypatch)


def _run_performance_worker(monkeypatch, result_dir: Path) -> dict:
    """Run common/MegaMoe A/B in one four-rank process group."""
    result_path = result_dir / "routed_expert_ab.json"
    monkeypatch.delenv("SYMMETRIC_MEMORY_HEAP_SIZE", raising=False)
    monkeypatch.setenv("HP_MEGA_MOE_WORLD_SIZE", str(_PERFORMANCE_WORLD_SIZE))
    monkeypatch.setenv("HP_MEGA_MOE_PERF_WARMUP", "3")
    monkeypatch.setenv("HP_MEGA_MOE_PERF_MEASURED", "5")
    monkeypatch.setenv("HP_MEGA_MOE_MAX_PERF_RATIO", "1.10")
    monkeypatch.setenv("HP_MEGA_MOE_PERF_RESULT", str(result_path))
    monkeypatch.setenv("HCCL_NPU_SOCKET_PORT_RANGE", "62000-62063")
    monkeypatch.setenv("SHMEM_IP_PORT", f"tcp://127.0.0.1:{allocate_port()}")
    with without_inherited_rank_environment():
        parallel_run(
            [
                TorchCase(
                    _WORKER,
                    "test_mega_moe_fwd_bwd_performance",
                    num_proc=_PERFORMANCE_WORLD_SIZE,
                )
            ],
            global_num_proc=_PERFORMANCE_WORLD_SIZE,
        )
    return json.loads(result_path.read_text(encoding="utf-8"))


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="unessential",
)
def test_mega_moe_representative_performance(monkeypatch, tmp_path: Path) -> None:
    """Enforce one steady-state common/MegaMoe routed-expert A/B threshold."""
    _prepare_torch_multicore_test_environment()
    comparison = _run_performance_worker(monkeypatch, tmp_path)
    results = comparison["results"]
    reference = results["common"]
    for tag, backend_result in results.items():
        assert backend_result["shape"] == reference["shape"], (
            f"performance shape drifted for {tag}: "
            f"expected={reference['shape']}, got={backend_result['shape']}"
        )
        assert backend_result["topology"] == reference["topology"], (
            f"performance topology drifted for {tag}: "
            f"expected={reference['topology']}, got={backend_result['topology']}"
        )
        assert backend_result["device_healthy"], (
            f"performance device health failed for {tag}: result={backend_result}"
        )
        assert all(backend_result["validation"].values()), (
            f"performance validation failed for {tag}: "
            f"validation={backend_result['validation']}"
        )

    common_median = results["common"]["rank_max_fwd_bwd_ms"]["median"]
    mega_median = results["mega_moe"]["rank_max_fwd_bwd_ms"]["median"]
    assert common_median > 0.0, (
        f"common median latency must be positive, got {common_median}."
    )
    assert mega_median > 0.0, f"MegaMoe median latency must be positive, got {mega_median}."
    ratio = mega_median / common_median
    assert ratio <= comparison["maximum_allowed_ratio"], (
        "MegaMoe routed-expert performance exceeded the A/B threshold: "
        f"ratio={ratio}, maximum={comparison['maximum_allowed_ratio']}."
    )
    print(
        "MegaMoe representative A/B: "
        f"common_ms={common_median}, mega_ms={mega_median}, ratio={ratio:.6f}"
    )


def _run_acceptance_worker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str, cards: int) -> dict:
    """Launch one real-device acceptance worker and require its rank evidence."""
    _prepare_torch_multicore_test_environment()
    result_dir = Path(os.getenv("HP_MEGA_MOE_EVIDENCE_DIR", str(tmp_path)))
    result_path = result_dir / f"{case}.json"
    worker = "_test_mega_moe_resources.py" if cards == 2 else "_test_mega_moe_router_interface.py"
    monkeypatch.setenv("HP_MEGA_MOE_WORLD_SIZE", str(cards))
    monkeypatch.setenv("HP_MEGA_MOE_LEVEL1_RESULT", str(result_path))
    monkeypatch.setenv("SHMEM_IP_PORT", f"tcp://127.0.0.1:{allocate_port()}")
    if cards == 2:
        monkeypatch.setenv("SYMMETRIC_MEMORY_HEAP_SIZE", str(64 * 1024 * 1024))
    else:
        monkeypatch.delenv("SYMMETRIC_MEMORY_HEAP_SIZE", raising=False)
    with without_inherited_rank_environment():
        parallel_run(
            [TorchCase(str(Path(__file__).with_name(worker)), case, num_proc=cards)],
            global_num_proc=cards,
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert len(result["ranks"]) == cards, f"expected {cards} rank records, got {len(result['ranks'])}."
    return result


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="unessential",
)
def test_mega_moe_shared_resources(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Feature: Shared execution resources and private SHMEM ownership.

    Description: Compare real serial stacks and alternate streams across repeated steps.
    Expectation: Outputs, gradients and updates match, memory stabilizes and the last owner finalizes.
    """
    _run_acceptance_worker(monkeypatch, tmp_path, "test_mega_moe_shared_resource_acceptance", 2)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="unessential",
)
def test_mega_moe_router_interface(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Feature: Router interface without caller-provided expert counts.

    Description: Connect a trainable Router and run one common/MegaMoe step at the representative shape.
    Expectation: Outputs, input/route/expert/Router gradients and one SGD update agree without supplied counts.
    """
    _run_acceptance_worker(monkeypatch, tmp_path, "test_mega_moe_router_interface", 4)
